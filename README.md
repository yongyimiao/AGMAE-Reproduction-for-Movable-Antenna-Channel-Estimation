# AGMAE Baseline Reproduction for MA Channel Estimation

This folder contains the AGMAE baseline reproduction used for the movable-antenna
(MA) channel-estimation experiments in our WCL revision.

The code is a clean, self-contained PyTorch reimplementation of the AGMAE
baseline for the considered MA dataset. It is not the official AGMAE
implementation. It is provided to make the baseline comparison auditable.

## What Is Included

- `reviewer1_agmae_fair_grid.py`: main entry point for the AGMAE fairness check.
- `models_mae_fas_GNN.py`: AGMAE-style masked autoencoder with a Transformer
  encoder and graph-attention decoder.
- `util/pos_embed.py`: 2-D sinusoidal positional embedding helper.
- `scripts/run_agmae_fair_grid_noiseless.sh`: no-noise fair-grid command.
- `environment.yml` and `requirements.txt`: tested environment notes.
- `docs/reviewer_response_snippet.md`: text that can be adapted in the rebuttal.

The channel datasets, checkpoints, logs, paper PDFs, and temporary Python cache
files are intentionally not included.

## Environment

The experiments were run with the following environment:

- Python 3.8
- PyTorch 1.11.0
- CUDA-enabled GPU
- `timm==0.3.2`
- NumPy

Recommended setup:

```bash
conda env create -f environment.yml
conda activate torch111
```

If PyTorch is already installed on the server, install only the Python-level
dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The script expects normalized CSI arrays in NumPy format:

```text
train_norm.npy: shape (num_train, 2, H, W)
test_norm.npy:  shape (num_test,  2, H, W)
```

For the MA experiments in the paper, `H=W=10`, and the two channels correspond
to the real and imaginary parts of the channel.

Data are not committed to this repository. Set the paths before running:

```bash
export TRAIN_DATA=/path/to/H_SV_f28_Nr100_Nt1_P5_alpha_train_norm.npy
export TEST_DATA=/path/to/H_SV_f28_Nr100_Nt1_P5_alpha_test_norm.npy
```

## Fairness Protocol

The script is designed to address the reviewer concern that AGMAE might have
been evaluated with FAS-default hyperparameters only.

It uses the following fair-comparison protocol:

1. The same MA train/test data are used for AGMAE and HybridMAE.
2. The same observation ratios are evaluated. The observation ratio is
   `rho = 1 - mask_ratio`.
3. The same no-noise channel setting is used for the no-noise comparison.
4. The same NMSE-style evaluation is reported.
5. A paper-faithful AGMAE setting is included:
   `embed_dim=256`, encoder depth `12`, encoder heads `4`, decoder feature
   dimension `32`, GAT decoder depth `24`, and `K_s=8` nearest neighbors.
6. MA-retuned alternatives are also swept over architecture, learning rate, and
   weight decay.
7. Hyperparameter selection is performed using a split from the training set.
   The test set is not used for model selection.

The default AGMAE reference in the script is:

```text
arch = paper_ma
learning rate = 1e-3
weight decay = 1e-4
```

This matches the paper-style AGMAE configuration adapted to the 10-by-10 MA
grid. The retuned AGMAE result is selected by validation NMSE.

## No-Noise Fair-Grid Run

From this folder, run:

```bash
export TRAIN_DATA=/path/to/H_SV_f28_Nr100_Nt1_P5_alpha_train_norm.npy
export TEST_DATA=/path/to/H_SV_f28_Nr100_Nt1_P5_alpha_test_norm.npy
export GPU=1
export OUTPUT_DIR=./reviewer1_results/agmae_fair_grid_full

mkdir -p "${OUTPUT_DIR}"
nohup bash scripts/run_agmae_fair_grid_noiseless.sh \
  > ${OUTPUT_DIR}/train.log 2>&1 &
```

For a quick smoke test:

```bash
python reviewer1_agmae_fair_grid.py \
  --train_data_path "$TRAIN_DATA" \
  --test_data_path "$TEST_DATA" \
  --output_dir ./reviewer1_results/smoke_test \
  --arch_grid paper_ma \
  --lr_grid 0.001 \
  --weight_decay_grid 0.0001 \
  --mask_ratios 0.799999 \
  --epochs_per_stage 1 \
  --mc_eval 1 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --max_configs 1
```

## Monitoring

```bash
tail -f ./reviewer1_results/agmae_fair_grid_full/train.log
find ./reviewer1_results/agmae_fair_grid_full -maxdepth 2 -name done.json | wc -l
nvidia-smi
```

## Output Files

The main output files are:

- `grid_results.csv`: every evaluated architecture/hyperparameter/stage row.
- `best_by_observed_ratio.csv`: validation-selected AGMAE result per observed
  ratio.
- `default_vs_retuned_by_observed_ratio.csv`: direct comparison between the
  paper-faithful AGMAE default and the MA-retuned AGMAE setting.
- `fairness_summary.json`: compact summary for the response letter.
- `cfg*/stage_results.csv`: per-configuration stage results.
- `cfg*/stage*/train_log.jsonl`: per-epoch training loss for each stage.

For the WCL response, the most useful file is usually
`default_vs_retuned_by_observed_ratio.csv`, because it directly answers whether
AGMAE was evaluated only with its default FAS hyperparameters or was also
retuned for the MA setting.

## Metrics

The script reports three NMSE variants:

- `nmse_masked`: NMSE over masked/unobserved ports only.
- `nmse_completed`: full-channel completion NMSE assuming the observed ports are
  already known, so only unobserved-port errors contribute to the numerator.
- `nmse_full_pred`: diagnostic NMSE over all ports using the raw model output.

For channel completion, `nmse_completed` is the recommended comparison metric.
The validation selector defaults to `val_nmse_completed`.

## Notes for Reproducibility

- Random seed is controlled by `--seed`.
- The default training split uses `--val_fraction 0.1`, i.e., 90 percent of the
  training file for training and 10 percent for validation. A separate
  validation file can be supplied via `--val_data_path`.
- Multi-mask progressive training is used inside each configuration, following
  the same observed-ratio schedule used in the baseline comparison.
- The test set is evaluated after each stage but is not used to select the
  retuned configuration.
