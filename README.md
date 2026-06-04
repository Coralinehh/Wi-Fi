# SWS-Net

This repository contains the cleaned training code for SWS-Net WiFi sensing experiments.

## Datasets

Large datasets are not tracked by git. Put them under:

```text
dataset/
  BVP_22guesture_all/
  NTU-Fi_HAR/
  NTU-Fi-HumanID/
```

The current code organizes the BVP 22-gesture experiments first.

## Structure

```text
data/       Dataset loading and preprocessing
model/      Model definitions
train/      Training and evaluation entry points
dataset/    Local datasets, ignored by git
```

## Train on BVP

From this directory:

```bash
python -m train.train_bvp --model cnn --data-root dataset/BVP_22guesture_all
```

Available models:

```text
cnn
cnnres
lstm
snngru
transformer
```

Example:

```bash
python -m train.train_bvp --model snngru --data-root dataset/BVP_22guesture_all --epochs 50 --batch-size 32
```

Checkpoints are written to `checkpoints/`.

BVP SNN-GRU threshold and timestep ablations:

```bash
# Full-time baseline, equivalent to K=None.
python -m train.train_bvp --model snngru --time-groups none --v-threshold 0.5

# Grouped-time comparison with K=4.
python -m train.train_bvp --model snngru --time-groups 4 --v-threshold 0.5

# Sweep thresholds.
python -m train.train_bvp --model snngru --time-groups none --v-thresholds 0.25,0.5,0.75,1.0

# Sweep thresholds and grouped-time settings together.
python -m train.train_bvp --model snngru --time-groups-list none,4,8 --v-thresholds 0.25,0.5,0.75,1.0
```

## Train on NTU-Fi CSI

HumanID and HAR share the same CSI data format, so they use one training entry.

```bash
python -m train.train_csi --dataset humanid --model cnn
python -m train.train_csi --dataset har --model cnn
```

Default dataset paths:

```text
humanid -> dataset/NTU-Fi-HumanID
har     -> dataset/NTU-Fi_HAR
```

Available CSI models:

```text
cnn
cnnres
lstm
snn
transformer
```

Example:

```bash
python -m train.train_csi --dataset har --model snn --time-steps 10 --channels 32
```

CSI SNN threshold and timestep ablations:

```bash
python -m train.train_csi --dataset humanid --model snn --time-steps 10 --v-threshold 0.5
python -m train.train_csi --dataset har --model snn --time-steps 10 --v-thresholds 0.25,0.5,0.75,1.0
python -m train.train_csi --dataset humanid --model snn --time-steps-list 5,10,15 --v-thresholds 0.25,0.5,0.75,1.0
```
