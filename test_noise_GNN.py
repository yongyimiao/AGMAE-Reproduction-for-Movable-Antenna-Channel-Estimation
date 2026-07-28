#!/usr/bin/env python3
"""
batch_eval_nmse_noise.py
批量计算各 stage checkpoint 在测试集上的标准 NMSE（noisy/clean 配对输入）。

标准 NMSE = Σ‖ŷ - y‖² / Σ‖y‖²   (在所有 masked patch 上累加)

用法:
    python batch_eval_nmse_noise.py \
        --test_noisy_path /path/to/test_noisy.npy \
        --test_clean_path /path/to/test_clean.npy \
        --output_dir      ./output_R2P6_noise \
        --model           mae_fas_channel_modelv2 \
        --device          cuda
"""

import argparse
import os
import glob
import json
import re

import numpy as np
import torch

import models_mae_fas_noise_GNN


# ──────────────────────────────────────────────
# 数据集（noisy/clean 配对，与训练脚本保持一致）
# ──────────────────────────────────────────────
class PairedCSIDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_path: str, clean_path: str, normalize: bool = False):
        noisy = np.load(noisy_path)
        clean = np.load(clean_path)
        assert noisy.shape == clean.shape, \
            f"noisy {noisy.shape} != clean {clean.shape}"
        assert noisy.ndim == 4 and noisy.shape[1] == 2, \
            f"Expected (N,2,H,W), got {noisy.shape}"
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


# ──────────────────────────────────────────────
# 标准 NMSE 评估
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_nmse(model, data_loader, mask_ratio: float,
                  device: torch.device) -> float:
    """
    标准 NMSE = Σ‖pred - target‖² / Σ‖target‖²
    分子分母均在全部样本的 masked patch 上累加后再做除法，
    避免 per-sample 平均带来的偏差。
    alpha 不参与任何计算，forward 传 alpha=0.0 仅为满足接口签名，
    返回的 loss（第0项）直接丢弃。
    """
    model.eval()
    total_numer = 0.0
    total_denom = 0.0

    _model = model.module if hasattr(model, 'module') else model

    for noisy, clean in data_loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        # alpha=0.0 仅满足接口，返回的 loss 直接丢弃
        _, pred, mask, _, _ = model(noisy, clean,
                                    mask_ratio=mask_ratio, alpha=0.0)

        # patchify ground truth → (B, L, patch_dim)
        target = _model.patchify(clean)

        # pred, target: (B, L, C);  mask: (B, L)  1=masked 0=visible
        se     = ((pred - target) ** 2).sum(dim=-1)   # (B, L)
        energy = (target          ** 2).sum(dim=-1)   # (B, L)

        # 只统计 masked 位置
        total_numer += (se     * mask).sum().item()
        total_denom += (energy * mask).sum().item()

    nmse = total_numer / (total_denom + 1e-12)
    return nmse


# ──────────────────────────────────────────────
# 加载 checkpoint
# ──────────────────────────────────────────────
def load_checkpoint(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容 misc.save_model 写入的格式：{'model': state_dict, ...}
    state_dict = ckpt.get('model', ckpt)

    # 去掉 DDP 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    [warn] missing keys  : {missing[:5]} ...")
    if unexpected:
        print(f"    [warn] unexpected keys: {unexpected[:5]} ...")


# ──────────────────────────────────────────────
# 扫描所有 stage 目录，找最新 checkpoint
# ──────────────────────────────────────────────
def find_stage_checkpoints(output_dir: str):
    """
    扫描 output_dir/stage*_mask*/checkpoint-*.pth
    返回按 stage 编号排序的列表：
        [(stage_idx, mask_ratio, ckpt_path), ...]
    取每个 stage 最大 epoch 的 checkpoint。
    """
    pattern = os.path.join(output_dir, 'stage*_mask*')
    stage_dirs = sorted(glob.glob(pattern))

    results = []
    for sd in stage_dirs:
        basename = os.path.basename(sd)  # e.g. stage3_mask0.65
        m = re.match(r'stage(\d+)_mask([\d.]+)', basename)
        if m is None:
            continue
        stage_idx  = int(m.group(1))
        mask_ratio = float(m.group(2))

        # 找该 stage 下所有 checkpoint，取最大 epoch
        ckpts = sorted(glob.glob(os.path.join(sd, 'checkpoint-*.pth')))
        if not ckpts:
            print(f"  [skip] no checkpoint found in {sd}")
            continue

        def epoch_num(p):
            n = re.search(r'checkpoint-(\d+)\.pth', os.path.basename(p))
            return int(n.group(1)) if n else -1

        best_ckpt = max(ckpts, key=epoch_num)
        results.append((stage_idx, mask_ratio, best_ckpt))

    results.sort(key=lambda x: x[0])
    return results


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser('Batch NMSE evaluation (paired noisy/clean)')
    parser.add_argument('--test_noisy_path', required=True,
                        help='测试集有噪 .npy，形状 (N,2,H,W)')
    parser.add_argument('--test_clean_path', required=True,
                        help='测试集无噪 .npy，形状 (N,2,H,W)')
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--model',  default='mae_fas_channel_modelv2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size',    default=128, type=int)
    parser.add_argument('--num_workers',   default=4,   type=int)
    parser.add_argument('--normalize',     action='store_true',
                        help='与训练时一致：是否用 clean 全局 max(abs) 归一化')
    parser.add_argument('--norm_pix_loss', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 测试集 ----
    dataset_test = PairedCSIDataset(
        args.test_noisy_path, args.test_clean_path, normalize=args.normalize)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=torch.utils.data.SequentialSampler(dataset_test),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Test samples: {len(dataset_test)}, shape: {dataset_test.noisy.shape}")

    # ---- 找所有 stage checkpoints ----
    stages = find_stage_checkpoints(args.output_dir)
    if not stages:
        raise RuntimeError(f"No valid stage directories found under {args.output_dir}")

    print(f"\nFound {len(stages)} stage(s):\n")
    for si, mr, cp in stages:
        print(f"  stage={si}  mask={mr:.2f}  ckpt={cp}")

    # ---- 逐 stage 评估 ----
    print("\n" + "="*60)
    results = []

    for stage_idx, mask_ratio, ckpt_path in stages:
        print(f"\n[Stage {stage_idx}]  mask_ratio={mask_ratio:.2f}")
        print(f"  checkpoint : {ckpt_path}")

        # 每次重新构建模型（防止权重污染）
        model = models_mae_fas_noise_GNN.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss)
        model.to(device)
        model.eval()

        load_checkpoint(model, ckpt_path, device)

        nmse = evaluate_nmse(model, loader_test, mask_ratio, device)
        print(f"  NMSE = {nmse:.6e}   ({10*np.log10(nmse + 1e-12):.2f} dB)")

        results.append({
            'stage':      stage_idx,
            'mask_ratio': mask_ratio,
            'ckpt':       ckpt_path,
            'nmse':       nmse,
            'nmse_dB':    float(10 * np.log10(nmse + 1e-12)),
        })

    # ---- 汇总打印 ----
    print("\n" + "="*60)
    print(f"{'Stage':>6} {'MaskRatio':>10} {'NMSE':>14} {'NMSE(dB)':>10}")
    print("-"*44)
    for r in results:
        print(f"  {r['stage']:>4}   {r['mask_ratio']:>8.2f}   "
              f"{r['nmse']:>12.6e}   {r['nmse_dB']:>8.2f} dB")
    print("="*60)

    # ---- 保存 JSON ----
    out_json = os.path.join(args.output_dir, 'nmse_eval_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_json}")


if __name__ == '__main__':
    main()
