from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class BvpDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels - 1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def normalize_data(sample):
    sample_max = np.concatenate((sample.max(axis=0), sample.max(axis=1)), axis=0).max(axis=0)
    sample_min = np.concatenate((sample.min(axis=0), sample.min(axis=1)), axis=0).min(axis=0)
    denom = sample_max - sample_min
    if np.any(denom == 0):
        return sample
    sample_max = np.tile(sample_max, (sample.shape[0], sample.shape[1], 1))
    sample_min = np.tile(sample_min, (sample.shape[0], sample.shape[1], 1))
    return (sample - sample_min) / (sample_max - sample_min)


def zero_padding(samples, max_steps):
    padded = []
    for sample in samples:
        steps = np.asarray(sample).shape[2]
        padded.append(np.pad(sample, ((0, 0), (0, 0), (max_steps - steps, 0)), "constant"))
    return np.asarray(padded)


def _label_from_filename(path):
    return int(path.name.split("-")[1])


def load_bvp_data(data_root, motions=None, mat_key="velocity_spectrum_ro"):
    data_root = Path(data_root)
    motions = set(motions or range(1, 23))
    samples = []
    labels = []
    max_steps = 0

    if not data_root.exists():
        raise FileNotFoundError(f"BVP dataset directory does not exist: {data_root}")

    for mat_path in sorted(data_root.rglob("*.mat")):
        try:
            label = _label_from_filename(mat_path)
            if label not in motions:
                continue
            sample = scio.loadmat(mat_path)[mat_key]
        except Exception:
            continue

        max_steps = max(max_steps, np.asarray(sample).shape[2])
        samples.append(normalize_data(sample))
        labels.append(label)

    if not samples:
        raise RuntimeError(f"No BVP samples were loaded from {data_root}")

    data = zero_padding(samples, max_steps)
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
    data = np.expand_dims(data, axis=-1)
    return data, np.asarray(labels), max_steps


def build_bvp_loaders(
    data_root,
    batch_size=32,
    test_size=0.1,
    val_size=0.1,
    seed=1,
    num_workers=0,
):
    data, labels, max_steps = load_bvp_data(data_root)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=val_size, stratify=train_labels, random_state=seed
    )

    loaders = {
        "train": DataLoader(BvpDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(BvpDataset(val_data, val_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(BvpDataset(test_data, test_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    input_shape = (None, max_steps, data.shape[2], data.shape[3], data.shape[4])
    return loaders, input_shape

