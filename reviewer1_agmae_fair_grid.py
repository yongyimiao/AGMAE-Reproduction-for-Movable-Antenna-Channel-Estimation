import argparse
import csv
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models_mae_fas_GNN import MaskedAutoencoderViT


ARCH_PRESETS = {
    # Paper-faithful AGMAE dimensions adapted to the MA 10x10, Nt=1 input.
    # Original Table I: D=256, L=12, K=4, F=32, Q=24, Ks=8.
    "paper_ma": {
        "embed_dim": 256,
        "depth": 12,
        "num_heads": 4,
        "decoder_embed_dim": 32,
        "decoder_depth": 24,
        "gat_k_neighbors": 8,
    },
    # The lightweight setting used in the previous local reproduction.
    "ma_v2": {
        "embed_dim": 256,
        "depth": 12,
        "num_heads": 8,
        "decoder_embed_dim": 128,
        "decoder_depth": 4,
        "gat_k_neighbors": 8,
    },
    # A stronger MA-adapted setting with a deeper encoder/decoder.
    "ma_v3": {
        "embed_dim": 256,
        "depth": 16,
        "num_heads": 8,
        "decoder_embed_dim": 128,
        "decoder_depth": 8,
        "gat_k_neighbors": 8,
    },
    # Middle point between the paper-faithful decoder depth and the local v2.
    "balanced": {
        "embed_dim": 256,
        "depth": 12,
        "num_heads": 4,
        "decoder_embed_dim": 64,
        "decoder_depth": 12,
        "gat_k_neighbors": 8,
    },
}


class NpyCSIDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, normalize=False):
        data = np.load(npy_path)
        if data.ndim != 4 or data.shape[1] != 2:
            raise ValueError(f"Expected data shape (N,2,H,W), got {data.shape}")
        if data.shape[2] != data.shape[3]:
            raise ValueError(f"Expected a square MA grid, got {data.shape[2:]}")
        if normalize:
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
            std = data.std(axis=(0, 2, 3), keepdims=True) + 1e-6
            data = (data - mean) / std
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], 0


def parse_float_list(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text):
    return [int(item) for item in text.split(",") if item.strip()]


def parse_str_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def observed_ratio(mask_ratio):
    return 1.0 - mask_ratio


def mask_tag(mask_ratio):
    return f"{int(round(mask_ratio * 100)):02d}"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val(dataset, val_fraction, seed):
    n_total = len(dataset)
    n_val = int(round(n_total * val_fraction))
    n_val = max(1, min(n_total - 1, n_val))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)


def make_loader(dataset, batch_size, num_workers, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_model(arch_name, img_size, norm_pix_loss=False):
    if arch_name not in ARCH_PRESETS:
        raise KeyError(f"Unknown arch '{arch_name}'. Available: {sorted(ARCH_PRESETS)}")
    cfg = dict(ARCH_PRESETS[arch_name])
    return MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=1,
        in_chans=2,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=norm_pix_loss,
        **cfg,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, progress_epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    if progress_epoch < warmup_epochs:
        lr = base_lr * progress_epoch / max(1.0, warmup_epochs)
    else:
        denom = max(1.0, total_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (progress_epoch - warmup_epochs) / denom)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_one_stage(model, loader, device, mask_ratio, epochs, lr, min_lr,
                    warmup_epochs, weight_decay, beta2, log_path):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, beta2),
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    model.train()
    total_steps = max(1, len(loader))
    warmup_epochs = min(warmup_epochs, max(1, epochs // 5 + 1))

    with open(log_path, "a", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = 0
            for step, (samples, _) in enumerate(loader):
                progress_epoch = epoch + step / total_steps
                current_lr = adjust_learning_rate(
                    optimizer, progress_epoch, epochs, lr, min_lr, warmup_epochs
                )
                samples = samples.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    loss, _, _ = model(samples, mask_ratio=mask_ratio)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_size = samples.shape[0]
                epoch_loss += float(loss.item()) * batch_size
                n_samples += batch_size
            row = {
                "epoch": epoch,
                "mask_ratio": mask_ratio,
                "observed_ratio": observed_ratio(mask_ratio),
                "train_mse_loss": epoch_loss / max(1, n_samples),
                "lr": current_lr,
            }
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()


@torch.no_grad()
def evaluate(model, loader, device, mask_ratio, mc_times, seed):
    model.eval()
    totals = {
        "masked_num": 0.0,
        "masked_den": 0.0,
        "completed_num": 0.0,
        "completed_den": 0.0,
        "full_pred_num": 0.0,
        "full_pred_den": 0.0,
        "observed_sum": 0.0,
        "observed_count": 0,
    }
    for mc in range(mc_times):
        torch.manual_seed(seed + mc)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + mc)
        for samples, _ in loader:
            samples = samples.to(device, non_blocking=True)
            _, pred, mask = model(samples, mask_ratio=mask_ratio)
            target = model.patchify(samples)
            se = ((pred - target) ** 2).sum(dim=-1)
            energy = (target ** 2).sum(dim=-1)

            totals["masked_num"] += float((se * mask).sum().item())
            totals["masked_den"] += float((energy * mask).sum().item())
            # Completion metric assumes observed ports are known and only masked
            # positions contribute to the numerator over the full-channel energy.
            totals["completed_num"] += float((se * mask).sum().item())
            totals["completed_den"] += float(energy.sum().item())
            # This diagnostic penalizes prediction errors also on visible ports.
            totals["full_pred_num"] += float(se.sum().item())
            totals["full_pred_den"] += float(energy.sum().item())
            totals["observed_sum"] += float((mask == 0).float().mean().item())
            totals["observed_count"] += 1

    def div(num, den):
        return num / (den + 1e-12)

    out = {
        "nmse_masked": div(totals["masked_num"], totals["masked_den"]),
        "nmse_completed": div(totals["completed_num"], totals["completed_den"]),
        "nmse_full_pred": div(totals["full_pred_num"], totals["full_pred_den"]),
        "observed_fraction": totals["observed_sum"] / max(1, totals["observed_count"]),
    }
    for key in ("nmse_masked", "nmse_completed", "nmse_full_pred"):
        out[key + "_db"] = 10.0 * math.log10(max(out[key], 1e-12))
    model.train()
    return out


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_best_by_mask(rows, metric):
    best = {}
    for row in rows:
        mask_ratio = row["stage_mask_ratio"]
        if mask_ratio not in best or row[metric] < best[mask_ratio][metric]:
            best[mask_ratio] = row
    return [best[key] for key in sorted(best)]


def select_default_by_mask(rows, default_arch, default_lr, default_weight_decay):
    default_rows = {}
    for row in rows:
        if row["arch"] != default_arch:
            continue
        if not math.isclose(float(row["lr"]), float(default_lr), rel_tol=1e-9, abs_tol=1e-12):
            continue
        if not math.isclose(
            float(row["weight_decay"]), float(default_weight_decay), rel_tol=1e-9, abs_tol=1e-12
        ):
            continue
        default_rows[row["stage_mask_ratio"]] = row
    return default_rows


def make_default_vs_retuned_rows(best_rows, default_rows, selection_metric):
    rows = []
    for best in best_rows:
        mask_ratio = best["stage_mask_ratio"]
        default = default_rows.get(mask_ratio)
        row = {
            "observed_ratio": best["observed_ratio"],
            "mask_ratio": mask_ratio,
            "selection_metric": selection_metric,
            "retuned_arch": best["arch"],
            "retuned_lr": best["lr"],
            "retuned_weight_decay": best["weight_decay"],
            "retuned_val_db": best[selection_metric + "_db"],
            "retuned_param_count": best["param_count"],
        }
        if "test_nmse_completed_db" in best:
            row.update({
                "retuned_test_completed_db": best["test_nmse_completed_db"],
                "retuned_test_masked_db": best["test_nmse_masked_db"],
                "retuned_test_full_pred_db": best["test_nmse_full_pred_db"],
            })

        if default is None:
            row.update({
                "default_arch": "",
                "default_lr": "",
                "default_weight_decay": "",
                "default_val_db": "",
                "retuned_gain_val_db": "",
            })
        else:
            row.update({
                "default_arch": default["arch"],
                "default_lr": default["lr"],
                "default_weight_decay": default["weight_decay"],
                "default_val_db": default[selection_metric + "_db"],
                # Positive means the MA-retuned configuration is better.
                "retuned_gain_val_db": default[selection_metric + "_db"] - best[selection_metric + "_db"],
            })
            if "test_nmse_completed_db" in default:
                row.update({
                    "default_test_completed_db": default["test_nmse_completed_db"],
                    "default_test_masked_db": default["test_nmse_masked_db"],
                    "default_test_full_pred_db": default["test_nmse_full_pred_db"],
                    "retuned_gain_test_completed_db": (
                        default["test_nmse_completed_db"] - best["test_nmse_completed_db"]
                    ) if "test_nmse_completed_db" in best else "",
                })
        rows.append(row)
    return rows


def run_config(config_id, arch, lr, weight_decay, args, datasets, device):
    train_set, val_set, test_set = datasets
    config_dir = Path(args.output_dir) / f"cfg{config_id:03d}_{arch}_lr{lr:g}_wd{weight_decay:g}"
    config_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "config_id": config_id,
        "arch": arch,
        "lr": lr,
        "weight_decay": weight_decay,
        "arch_config": ARCH_PRESETS[arch],
        "args": vars(args),
    }
    with open(config_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    train_loader = make_loader(
        train_set, args.batch_size, args.num_workers, shuffle=True, seed=args.seed + config_id
    )
    val_loader = make_loader(
        val_set, args.eval_batch_size, args.num_workers, shuffle=False, seed=args.seed
    )
    test_loader = None
    if test_set is not None:
        test_loader = make_loader(
            test_set, args.eval_batch_size, args.num_workers, shuffle=False, seed=args.seed
        )

    model = build_model(arch, args.img_size, norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    param_count = count_parameters(model)
    print(
        f"[cfg={config_id:03d}] arch={arch}, lr={lr:g}, wd={weight_decay:g}, "
        f"params={param_count/1e6:.2f}M"
    )

    mask_ratios = parse_float_list(args.mask_ratios)
    epochs_list = parse_int_list(args.epochs_per_stage)
    if len(epochs_list) == 1:
        epochs_list = epochs_list * len(mask_ratios)
    if len(mask_ratios) != len(epochs_list):
        raise ValueError("--mask_ratios and --epochs_per_stage length mismatch")

    rows = []
    start_time = time.time()
    for stage_idx, (stage_mask, epochs) in enumerate(zip(mask_ratios, epochs_list)):
        stage_dir = config_dir / f"stage{stage_idx}_mask{stage_mask:.2f}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[cfg={config_id:03d}] stage={stage_idx}, mask={stage_mask:.6f}, "
            f"rho={observed_ratio(stage_mask):.4f}, epochs={epochs}"
        )
        train_one_stage(
            model=model,
            loader=train_loader,
            device=device,
            mask_ratio=stage_mask,
            epochs=epochs,
            lr=lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            weight_decay=weight_decay,
            beta2=args.beta2,
            log_path=str(stage_dir / "train_log.jsonl"),
        )
        if args.save_checkpoints:
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": arch,
                    "arch_config": ARCH_PRESETS[arch],
                    "stage": stage_idx,
                    "mask_ratio": stage_mask,
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                stage_dir / "checkpoint-final.pth",
            )

        val_metrics = evaluate(
            model, val_loader, device, stage_mask, args.mc_eval, args.seed + 10000 + stage_idx
        )
        test_metrics = {}
        if test_loader is not None:
            test_metrics = evaluate(
                model, test_loader, device, stage_mask, args.mc_eval, args.seed + 20000 + stage_idx
            )

        row = {
            "config_id": config_id,
            "arch": arch,
            "lr": lr,
            "weight_decay": weight_decay,
            "param_count": param_count,
            "stage": stage_idx,
            "stage_mask_ratio": stage_mask,
            "observed_ratio": observed_ratio(stage_mask),
            "epochs": epochs,
            "val_nmse_masked": val_metrics["nmse_masked"],
            "val_nmse_masked_db": val_metrics["nmse_masked_db"],
            "val_nmse_completed": val_metrics["nmse_completed"],
            "val_nmse_completed_db": val_metrics["nmse_completed_db"],
            "val_nmse_full_pred": val_metrics["nmse_full_pred"],
            "val_nmse_full_pred_db": val_metrics["nmse_full_pred_db"],
            "observed_fraction": val_metrics["observed_fraction"],
        }
        if test_metrics:
            row.update({
                "test_nmse_masked": test_metrics["nmse_masked"],
                "test_nmse_masked_db": test_metrics["nmse_masked_db"],
                "test_nmse_completed": test_metrics["nmse_completed"],
                "test_nmse_completed_db": test_metrics["nmse_completed_db"],
                "test_nmse_full_pred": test_metrics["nmse_full_pred"],
                "test_nmse_full_pred_db": test_metrics["nmse_full_pred_db"],
                "test_observed_fraction": test_metrics["observed_fraction"],
            })
        rows.append(row)
        write_csv(config_dir / "stage_results.csv", rows)
        print(
            f"[cfg={config_id:03d}] val completed={row['val_nmse_completed_db']:.2f} dB, "
            f"masked={row['val_nmse_masked_db']:.2f} dB, "
            f"obs_frac={row['observed_fraction']:.4f}"
        )

    elapsed = time.time() - start_time
    with open(config_dir / "done.json", "w", encoding="utf-8") as f:
        json.dump({"elapsed_minutes": elapsed / 60.0}, f, indent=2)
    return rows


def get_args_parser():
    parser = argparse.ArgumentParser("Fair AGMAE retuning on the MA dataset")
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--test_data_path", default="")
    parser.add_argument("--val_data_path", default="")
    parser.add_argument("--output_dir", default="./reviewer1_results/agmae_fair_grid")
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--img_size", default=10, type=int)
    parser.add_argument("--arch_grid", default="paper_ma,ma_v2,balanced,ma_v3")
    parser.add_argument("--lr_grid", default="0.0003,0.001")
    parser.add_argument("--weight_decay_grid", default="0.0001,0.001,0.05")
    parser.add_argument("--mask_ratios", default="0.50,0.55,0.60,0.65,0.70,0.75,0.799999,0.85,0.90")
    parser.add_argument("--epochs_per_stage", default="200")
    parser.add_argument("--selection_metric", default="val_nmse_completed",
                        choices=["val_nmse_completed", "val_nmse_masked", "val_nmse_full_pred"])
    parser.add_argument("--default_arch", default="paper_ma",
                        help="Paper-faithful AGMAE configuration used as the FAS-default reference.")
    parser.add_argument("--default_lr", default=1e-3, type=float)
    parser.add_argument("--default_weight_decay", default=1e-4, type=float)

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--warmup_epochs", default=40, type=int)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--mc_eval", default=3, type=int)
    parser.add_argument("--val_fraction", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--max_configs", default=0, type=int,
                        help="For smoke tests. 0 means run all grid configurations.")
    return parser


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_full = NpyCSIDataset(args.train_data_path, normalize=args.normalize)
    if args.val_data_path:
        train_set = train_full
        val_set = NpyCSIDataset(args.val_data_path, normalize=args.normalize)
    else:
        train_set, val_set = split_train_val(train_full, args.val_fraction, args.seed)
    test_set = NpyCSIDataset(args.test_data_path, normalize=args.normalize) if args.test_data_path else None

    sample = train_full[0][0]
    if sample.shape[-1] != args.img_size or sample.shape[-2] != args.img_size:
        raise ValueError(f"--img_size={args.img_size} but data sample has shape {tuple(sample.shape)}")

    print(f"train={len(train_set)}, val={len(val_set)}, test={len(test_set) if test_set else 0}")
    print(f"arch_grid={args.arch_grid}")
    print(f"lr_grid={args.lr_grid}")
    print(f"weight_decay_grid={args.weight_decay_grid}")
    print(f"mask_ratios={args.mask_ratios}")

    arch_grid = parse_str_list(args.arch_grid)
    lr_grid = parse_float_list(args.lr_grid)
    wd_grid = parse_float_list(args.weight_decay_grid)

    all_rows = []
    config_id = 0
    for arch in arch_grid:
        for lr in lr_grid:
            for weight_decay in wd_grid:
                if args.max_configs and config_id >= args.max_configs:
                    break
                rows = run_config(
                    config_id=config_id,
                    arch=arch,
                    lr=lr,
                    weight_decay=weight_decay,
                    args=args,
                    datasets=(train_set, val_set, test_set),
                    device=device,
                )
                all_rows.extend(rows)
                write_csv(Path(args.output_dir) / "grid_results.csv", all_rows)
                best_rows = select_best_by_mask(all_rows, args.selection_metric)
                write_csv(Path(args.output_dir) / "best_by_observed_ratio.csv", best_rows)
                config_id += 1
            if args.max_configs and config_id >= args.max_configs:
                break
        if args.max_configs and config_id >= args.max_configs:
            break

    best_rows = select_best_by_mask(all_rows, args.selection_metric)
    default_rows = select_default_by_mask(
        all_rows, args.default_arch, args.default_lr, args.default_weight_decay
    )
    comparison_rows = make_default_vs_retuned_rows(best_rows, default_rows, args.selection_metric)
    write_csv(Path(args.output_dir) / "default_vs_retuned_by_observed_ratio.csv", comparison_rows)
    summary = {
        "selection_metric": args.selection_metric,
        "num_grid_rows": len(all_rows),
        "num_configs": config_id,
        "default_reference": {
            "arch": args.default_arch,
            "lr": args.default_lr,
            "weight_decay": args.default_weight_decay,
            "present_for_all_masks": len(default_rows) == len(best_rows),
        },
        "best_by_observed_ratio": best_rows,
        "default_vs_retuned_by_observed_ratio": comparison_rows,
    }
    with open(Path(args.output_dir) / "fairness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if len(default_rows) != len(best_rows):
        print(
            "[warn] The paper-default AGMAE reference was not present for all masks. "
            "Make sure arch_grid/lr_grid/weight_decay_grid include "
            f"{args.default_arch}, {args.default_lr:g}, {args.default_weight_decay:g}."
        )

    print("\nBest AGMAE configurations selected by validation metric:")
    for row in best_rows:
        msg = (
            f"rho={row['observed_ratio']:.3f}, arch={row['arch']}, "
            f"lr={row['lr']:g}, wd={row['weight_decay']:g}, "
            f"val={row[args.selection_metric + '_db']:.2f} dB"
        )
        if "test_nmse_completed_db" in row:
            msg += (
                f", test_completed={row['test_nmse_completed_db']:.2f} dB, "
                f"test_masked={row['test_nmse_masked_db']:.2f} dB"
            )
        print(msg)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
