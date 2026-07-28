# AGMAE Baseline Reproduction for MA Channel Estimation

This folder contains the AGMAE baseline reproduction used for the movable-antenna
(MA) channel-estimation experiments in our WCL revision.

The code is a PyTorch reimplementation of the AGMAE
baseline for the considered MA dataset. It is not the official AGMAE
implementation. It is provided to make the baseline comparison auditable.

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
