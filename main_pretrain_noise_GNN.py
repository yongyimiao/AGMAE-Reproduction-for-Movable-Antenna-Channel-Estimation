# --------------------------------------------------------
# MAE noise-aware pre-training with GAT decoder
# Modified from main_pretrain_noise.py + main_pretrain_channel_GNN.py
# 输入：训练集/测试集各一对 npy (noisy, clean)
# 支持渐进式多 mask 比例训练，每阶段结束后输出测试集 NMSE
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_fas_noise_GNN

from engine_pretrain_noise import train_one_epoch


# ---------------------------------------------------------------------------
# Dataset：有噪 / 无噪 配对
# ---------------------------------------------------------------------------

class PairedCSIDataset(torch.utils.data.Dataset):
    """
    加载一对 .npy 文件：noisy_path (带噪) 和 clean_path (无噪)。
    形状均为 (N, 2, H, W)。
    """
    def __init__(self, noisy_path: str, clean_path: str, normalize: bool = False):
        noisy = np.load(noisy_path)
        clean = np.load(clean_path)

        assert noisy.shape == clean.shape, \
            f"noisy {noisy.shape} != clean {clean.shape}"
        assert noisy.ndim == 4 and noisy.shape[1] == 2, \
            f"Expected (N, 2, H, W), got {noisy.shape}"

        if normalize:
            max_val = np.max(np.abs(clean)) + 1e-6
            noisy = noisy / max_val
            clean = clean / max_val

        self.noisy = torch.tensor(noisy, dtype=torch.float32)
        self.clean = torch.tensor(clean, dtype=torch.float32)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


# ---------------------------------------------------------------------------
# NMSE 评估（测试集，masked 区域，以 clean 为目标）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nmse(model, data_loader, mask_ratio: float,
                  alpha: float, device: torch.device) -> dict:
    """
    在测试集上跑一遍，返回：
      - nmse_masked:   仅掩码区域的 NMSE（主要指标）
      - nmse_full:     全部端口的 NMSE（去噪质量）
    """
    model.eval()
    numer_masked = 0.0;  denom_masked = 0.0
    numer_full   = 0.0;  denom_full   = 0.0

    _model = model.module if hasattr(model, 'module') else model

    for noisy, clean in data_loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        _, pred, mask, _, _ = model(noisy, clean,
                                    mask_ratio=mask_ratio, alpha=alpha)

        target = _model.patchify(clean)          # (B, L, 2)

        se     = ((pred - target) ** 2).sum(dim=-1)   # (B, L)
        energy = (target ** 2).sum(dim=-1)             # (B, L)

        # masked
        numer_masked += (se * mask).sum().item()
        denom_masked += (energy * mask).sum().item()
        # full
        numer_full   += se.sum().item()
        denom_full   += energy.sum().item()

    model.train()
    return {
        'nmse_masked': numer_masked / (denom_masked + 1e-12),
        'nmse_full':   numer_full   / (denom_full   + 1e-12),
    }


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE noise-aware pre-training (multi-mask progressive)', add_help=False)

    # ---- 数据（训练集 + 测试集，各一对 noisy/clean npy）----
    parser.add_argument('--train_noisy_path', default='', type=str,
                        help='训练集有噪 .npy，形状 (N, 2, H, W)')
    parser.add_argument('--train_clean_path', default='', type=str,
                        help='训练集无噪 .npy，形状 (N, 2, H, W)')
    parser.add_argument('--test_noisy_path',  default='', type=str,
                        help='测试集有噪 .npy（可选）')
    parser.add_argument('--test_clean_path',  default='', type=str,
                        help='测试集无噪 .npy（可选）')
    parser.add_argument('--normalize', action='store_true',
                        help='用 clean 数据的全局 max(abs) 做归一化')

    # ---- 模型 ----
    parser.add_argument('--model', default='mae_fas_channel_model', type=str,
                        metavar='MODEL',
                        help='models_mae_fas_noise_GNN 中的工厂函数名')
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)

    # ---- 渐进式 mask 训练 ----
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='当前阶段工作 mask 比例（由 train_stage 覆写）')
    parser.add_argument('--mask_ratios', default='0.50,0.75,0.90', type=str,
                        help='逗号分隔的渐进 mask 比例列表，如 "0.50,0.75,0.90"')
    parser.add_argument('--epochs_per_stage', default='200,200,200', type=str,
                        help='各阶段 epoch 数，与 --mask_ratios 等长；'
                             '也可填单个值自动广播')

    # ---- 损失权重 ----
    parser.add_argument('--alpha', type=float, default=0.84,
                        help='loss = alpha*masked_nmse + (1-alpha)*unmasked_nmse')

    # ---- 优化器 ----
    parser.add_argument('--batch_size',    default=64,   type=int)
    parser.add_argument('--accum_iter',    default=1,    type=int)
    parser.add_argument('--weight_decay',  default=0.05, type=float)
    parser.add_argument('--lr',    default=None, type=float, metavar='LR',
                        help='绝对学习率，None 时由 --blr 计算')
    parser.add_argument('--blr',   default=1e-3, type=float, metavar='LR',
                        help='基础学习率: lr = blr * total_batch / 256')
    parser.add_argument('--min_lr',        default=0.,   type=float)
    parser.add_argument('--warmup_epochs', default=40,   type=int)

    # ---- 杂项 ----
    parser.add_argument('--output_dir',  default='./output_dir')
    parser.add_argument('--log_dir',     default='./output_dir')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=0,  type=int)
    parser.add_argument('--resume',      default='',
                        help='第一阶段的恢复 checkpoint')
    parser.add_argument('--start_epoch', default=0,  type=int)
    parser.add_argument('--num_workers', default=8,  type=int)
    parser.add_argument('--pin_mem',     action='store_true')
    parser.add_argument('--no_pin_mem',  action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ---- 分布式 ----
    parser.add_argument('--world_size',  default=1,  type=int)
    parser.add_argument('--local_rank',  default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',    default='env://')

    return parser


# ---------------------------------------------------------------------------
# 单阶段训练
# ---------------------------------------------------------------------------

def train_stage(model, data_loader_train, data_loader_test,
                mask_ratio: float, n_epochs: int, stage_idx: int,
                args, device, global_rank):
    """
    以固定 mask_ratio 训练 n_epochs，结束后在测试集上评估 NMSE。
    模型权重在阶段间共享（渐进迁移）。
    """
    stage_output_dir = os.path.join(
        args.output_dir, f'stage{stage_idx}_mask{mask_ratio:.2f}')
    stage_log_dir    = os.path.join(
        args.log_dir,    f'stage{stage_idx}_mask{mask_ratio:.2f}')
    os.makedirs(stage_output_dir, exist_ok=True)

    log_writer = None
    if global_rank == 0:
        os.makedirs(stage_log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=stage_log_dir)

    # ---- 优化器（每阶段重建）----
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    lr = args.blr * eff_batch_size / 256 if args.lr is None else args.lr
    args.lr = lr   # 写回，供 lr_sched 读取

    model_without_ddp = model.module if args.distributed else model
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler  = NativeScaler()

    # 仅第一阶段尝试恢复 checkpoint
    if stage_idx == 0 and args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)

    # 临时覆写 args 供 engine 读取
    _orig_mask_ratio   = args.mask_ratio
    _orig_epochs       = getattr(args, "epochs", None)
    _orig_output_dir   = args.output_dir
    _orig_warmup       = args.warmup_epochs
    args.mask_ratio    = mask_ratio
    args.output_dir    = stage_output_dir
    args.warmup_epochs = min(args.warmup_epochs, n_epochs // 5 + 1)
    args.epochs        = n_epochs

    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx}  |  mask_ratio={mask_ratio:.2f}  "
          f"|  alpha={args.alpha:.2f}  |  epochs={n_epochs}")
    print(f"{'='*60}")

    start = time.time()
    for epoch in range(n_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        # 保存 checkpoint
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == n_epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'stage': stage_idx,
                     'mask_ratio': mask_ratio, 'alpha': args.alpha}

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(stage_output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    print(f"  Stage {stage_idx} training time: {elapsed}")

    # ---- 测试集 NMSE ----
    if data_loader_test is not None and misc.is_main_process():
        nmse_dict = evaluate_nmse(model, data_loader_test,
                                  mask_ratio, args.alpha, device)
        nmse_masked = nmse_dict['nmse_masked']
        nmse_full   = nmse_dict['nmse_full']
        print(f"  [Stage {stage_idx}] Test NMSE (masked) @ mask={mask_ratio:.2f}: "
              f"{nmse_masked:.6e}")
        print(f"  [Stage {stage_idx}] Test NMSE (full)   @ mask={mask_ratio:.2f}: "
              f"{nmse_full:.6e}")

        nmse_path = os.path.join(args.output_dir, 'nmse_results.txt')
        with open(nmse_path, 'a') as f:
            f.write(json.dumps({
                'stage': stage_idx, 'mask_ratio': mask_ratio,
                'nmse_masked': nmse_masked, 'nmse_full': nmse_full,
            }) + '\n')

        if log_writer is not None:
            log_writer.add_scalar('test/nmse_masked', nmse_masked, stage_idx)
            log_writer.add_scalar('test/nmse_full',   nmse_full,   stage_idx)

    # 恢复 args
    args.mask_ratio    = _orig_mask_ratio
    if _orig_epochs is None:
        del args.epochs
    else:
        args.epochs = _orig_epochs
    args.output_dir    = _orig_output_dir
    args.warmup_epochs = _orig_warmup

    if log_writer is not None:
        log_writer.close()

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device      = torch.device(args.device)
    global_rank = misc.get_rank()

    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- 解析渐进训练计划 ----
    mask_ratios = [float(x) for x in args.mask_ratios.split(',')]
    epochs_list = [int(x)   for x in args.epochs_per_stage.split(',')]
    if len(epochs_list) == 1:
        epochs_list = epochs_list * len(mask_ratios)
    assert len(mask_ratios) == len(epochs_list), \
        "--mask_ratios 和 --epochs_per_stage 长度必须一致。"

    print("\n渐进训练计划:")
    for i, (mr, ep) in enumerate(zip(mask_ratios, epochs_list)):
        print(f"  Stage {i}: mask_ratio={mr:.2f}, epochs={ep}")

    # ---- 数据集 ----
    assert args.train_noisy_path and args.train_clean_path, \
        "必须同时指定 --train_noisy_path 和 --train_clean_path"

    dataset_train = PairedCSIDataset(
        args.train_noisy_path, args.train_clean_path,
        normalize=args.normalize)
    print(f"Train: {len(dataset_train)} 样本, "
          f"noisy shape={dataset_train.noisy.shape}")

    dataset_test = None
    if args.test_noisy_path and args.test_clean_path:
        dataset_test = PairedCSIDataset(
            args.test_noisy_path, args.test_clean_path,
            normalize=args.normalize)
        print(f"Test:  {len(dataset_test)} 样本, "
              f"noisy shape={dataset_test.noisy.shape}")
    else:
        print("未指定测试集，跳过 NMSE 评估。")

    # ---- Sampler & DataLoader ----
    num_tasks     = misc.get_world_size()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks,
        rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = None
    if dataset_test is not None:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # ---- 构建模型 ----
    model = models_mae_fas_noise_GNN.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    print("Model = %s" % str(model.module if args.distributed else model))

    # ---- 渐进式训练 ----
    total_start = time.time()
    for stage_idx, (mask_ratio, n_epochs) in enumerate(
            zip(mask_ratios, epochs_list)):
        model = train_stage(
            model, data_loader_train, data_loader_test,
            mask_ratio=mask_ratio,
            n_epochs=n_epochs,
            stage_idx=stage_idx,
            args=args,
            device=device,
            global_rank=global_rank,
        )

    total_time_str = str(
        datetime.timedelta(seconds=int(time.time() - total_start)))
    print(f'\n总训练时间: {total_time_str}')

    # ---- 最终汇总 NMSE ----
    if data_loader_test is not None and misc.is_main_process():
        print("\n===== 最终测试集 NMSE（所有 mask 比例）=====")
        for mr in mask_ratios:
            nmse_dict = evaluate_nmse(model, data_loader_test,
                                      mr, args.alpha, device)
            print(f"  mask_ratio={mr:.2f}  "
                  f"NMSE_masked={nmse_dict['nmse_masked']:.6e}  "
                  f"NMSE_full={nmse_dict['nmse_full']:.6e}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
