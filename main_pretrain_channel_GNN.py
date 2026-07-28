# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# Modified: npy data interface, progressive multi-mask training, test NMSE eval
# --------------------------------------------------------
import argparse
import copy
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

import models_mae_fas_GNN

from engine_pretrain import train_one_epoch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FloatTensorDataset(torch.utils.data.Dataset):
    """
    Load a .npy file of shape (N, 2, H, W).
    Optional per-channel z-score normalization.
    """
    def __init__(self, npy_path: str, normalize: bool = False):
        data = np.load(npy_path)
        assert data.ndim == 4 and data.shape[1] == 2, \
            f"Expected shape (N, 2, H, W), got {data.shape}"
        if normalize:
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
            std  = data.std(axis=(0, 2, 3),  keepdims=True) + 1e-6
            data = (data - mean) / std
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0   # dummy label for MAE


# ---------------------------------------------------------------------------
# NMSE evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nmse(model, data_loader, mask_ratio: float, device: torch.device) -> float:
    """
    Run one pass over data_loader and return the NMSE on **masked** positions.
    NMSE = Σ‖pred - target‖² / Σ‖target‖²  (per sample, then averaged)
    """
    model.eval()
    total_num   = 0
    total_numer = 0.0
    total_denom = 0.0

    for samples, _ in data_loader:
        samples = samples.to(device, non_blocking=True)
        _, pred, mask = model(samples, mask_ratio=mask_ratio)

        # patchify ground truth to match pred shape  (B, L, C)
        target = model.module.patchify(samples) \
            if hasattr(model, 'module') else model.patchify(samples)

        # per-sample squared error on masked patches only
        # pred, target: (B, L, C);  mask: (B, L)  1=masked
        se     = ((pred - target) ** 2).sum(dim=-1)   # (B, L)
        energy = (target ** 2).sum(dim=-1)             # (B, L)

        masked_se     = (se     * mask).sum(dim=-1)    # (B,)
        masked_energy = (energy * mask).sum(dim=-1)    # (B,)

        total_numer += masked_se.sum().item()
        total_denom += masked_energy.sum().item()
        total_num   += samples.shape[0]

    nmse = total_numer / (total_denom + 1e-12)
    model.train()
    return nmse


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training (multi-mask progressive)', add_help=False)

    # ---- data ----
    parser.add_argument('--train_data_path', default='', type=str,
                        help='Path to training .npy file, shape (N, 2, H, W)')
    parser.add_argument('--test_data_path', default='', type=str,
                        help='Path to test .npy file (optional). If given, NMSE is reported after each stage.')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply per-channel z-score normalization to the loaded data.')

    # ---- model ----
    parser.add_argument('--model', default='mae_fas_channel_modelv2', type=str, metavar='MODEL',
                        help='Model factory name defined in models_mae_fas_GNN.')
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)

    # ---- progressive mask training ----
    # mask_ratio is a working attribute set internally by train_stage; initialised here
    # so that engine_pretrain.train_one_epoch can read args.mask_ratio at any time.
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Internal working mask ratio (overridden per stage). '
                             'Use --mask_ratios to set the full schedule.')
    parser.add_argument('--mask_ratios', default='0.50,0.75,0.90', type=str,
                        help='Comma-separated list of mask ratios to train progressively, '
                             'e.g. "0.50,0.75,0.90". A separate model is trained for each ratio, '
                             'initialised from the previous stage\'s weights.')
    parser.add_argument('--epochs_per_stage', default='200,200,200', type=str,
                        help='Comma-separated epochs for each mask-ratio stage. '
                             'Must have the same length as --mask_ratios. '
                             'A single value is broadcast to all stages.')

    # ---- optimiser ----
    parser.add_argument('--batch_size',   default=128,  type=int)
    parser.add_argument('--accum_iter',   default=1,    type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--lr',    default=None,  type=float, metavar='LR',
                        help='Absolute learning rate. If None, computed from --blr.')
    parser.add_argument('--blr',   default=1e-3,  type=float, metavar='LR',
                        help='Base LR: lr = blr * total_batch_size / 256')
    parser.add_argument('--min_lr',         default=0.,  type=float)
    parser.add_argument('--warmup_epochs',  default=40,  type=int)

    # ---- misc ----
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir',    default='./output_dir')
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--seed',       default=0, type=int)
    parser.add_argument('--resume',     default='',
                        help='Resume checkpoint for the FIRST stage only.')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=8,  type=int)
    parser.add_argument('--pin_mem',     action='store_true')
    parser.add_argument('--no_pin_mem',  action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ---- distributed ----
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


# ---------------------------------------------------------------------------
# Training one stage (one mask ratio)
# ---------------------------------------------------------------------------

def train_stage(model, data_loader_train, data_loader_test,
                mask_ratio: float, n_epochs: int, stage_idx: int,
                args, device, global_rank):
    """
    Fine-tune / train `model` for `n_epochs` at a fixed `mask_ratio`.
    Returns the trained model (in-place mutation, also returned for clarity).
    """
    stage_output_dir = os.path.join(args.output_dir, f'stage{stage_idx}_mask{mask_ratio:.2f}')
    stage_log_dir    = os.path.join(args.log_dir,    f'stage{stage_idx}_mask{mask_ratio:.2f}')
    os.makedirs(stage_output_dir, exist_ok=True)

    log_writer = None
    if global_rank == 0:
        os.makedirs(stage_log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=stage_log_dir)

    # ---- optimiser (re-created fresh each stage) ----
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    lr = args.blr * eff_batch_size / 256 if args.lr is None else args.lr
    args.lr = lr   # write back so lr_sched.adjust_learning_rate can read args.lr

    model_without_ddp = model.module if args.distributed else model
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler  = NativeScaler()

    # resume only for the very first stage
    if stage_idx == 0 and args.resume:
        # temporarily point args.output_dir to stage dir so load_model finds the ckpt
        _orig_resume = args.resume
        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)
        args.resume = _orig_resume

    # patch args so train_one_epoch sees the right mask_ratio & output_dir
    _orig_mask_ratio  = args.mask_ratio
    _orig_epochs      = getattr(args, "epochs", None)
    _orig_output_dir  = args.output_dir
    _orig_warmup      = args.warmup_epochs
    args.mask_ratio   = mask_ratio
    args.output_dir   = stage_output_dir
    args.warmup_epochs = min(args.warmup_epochs, n_epochs // 5 + 1)
    args.epochs        = n_epochs

    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx}  |  mask_ratio={mask_ratio:.2f}  |  epochs={n_epochs}")
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

        # save checkpoint
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == n_epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'stage': stage_idx, 'mask_ratio': mask_ratio}

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(stage_output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    print(f"  Stage {stage_idx} training time: {elapsed}")

    # ---- test NMSE after this stage ----
    if data_loader_test is not None and misc.is_main_process():
        nmse = evaluate_nmse(model, data_loader_test, mask_ratio, device)
        print(f"  [Stage {stage_idx}] Test NMSE @ mask={mask_ratio:.2f}: {nmse:.6e}")
        nmse_path = os.path.join(args.output_dir, "nmse_results.txt")
        with open(nmse_path, "a") as f:
            f.write(json.dumps({
                'stage': stage_idx, 'mask_ratio': mask_ratio, 'test_nmse': nmse
            }) + "\n")
        if log_writer is not None:
            log_writer.add_scalar('test/nmse', nmse, stage_idx)

    # restore patched args
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

    # ---- parse progressive training schedule ----
    mask_ratios = [float(x) for x in args.mask_ratios.split(',')]
    epochs_list = [int(x)   for x in args.epochs_per_stage.split(',')]
    if len(epochs_list) == 1:
        epochs_list = epochs_list * len(mask_ratios)
    assert len(mask_ratios) == len(epochs_list), \
        "--mask_ratios and --epochs_per_stage must have the same number of entries."

    print(f"\nProgressive training schedule:")
    for i, (mr, ep) in enumerate(zip(mask_ratios, epochs_list)):
        print(f"  Stage {i}: mask_ratio={mr:.2f}, epochs={ep}")

    # ---- datasets ----
    assert args.train_data_path, "--train_data_path must be specified."
    dataset_train = FloatTensorDataset(args.train_data_path, normalize=args.normalize)
    print(f"Train dataset: {len(dataset_train)} samples, shape={dataset_train.data.shape}")

    dataset_test = None
    if args.test_data_path:
        dataset_test = FloatTensorDataset(args.test_data_path, normalize=args.normalize)
        print(f"Test  dataset: {len(dataset_test)} samples, shape={dataset_test.data.shape}")

    # ---- samplers & loaders ----
    num_tasks = misc.get_world_size()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
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

    # ---- build model once ----
    model = models_mae_fas_GNN.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    print("Model = %s" % str(model.module if args.distributed else model))

    # ---- progressive training: iterate over stages ----
    total_start = time.time()
    for stage_idx, (mask_ratio, n_epochs) in enumerate(zip(mask_ratios, epochs_list)):
        model = train_stage(
            model, data_loader_train, data_loader_test,
            mask_ratio=mask_ratio,
            n_epochs=n_epochs,
            stage_idx=stage_idx,
            args=args,
            device=device,
            global_rank=global_rank,
        )

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - total_start)))
    print(f'\nTotal training time: {total_time_str}')

    # ---- final NMSE summary ----
    if data_loader_test is not None and misc.is_main_process():
        print("\n===== Final NMSE on test set at all mask ratios =====")
        for mr in mask_ratios:
            nmse = evaluate_nmse(model, data_loader_test, mr, device)
            print(f"  mask_ratio={mr:.2f}  NMSE={nmse:.6e}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
