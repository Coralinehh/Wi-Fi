from pathlib import Path

import scipy.io as sio
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class NTUFiCSIDataset(data.Dataset):
    def __init__(self, root_dir, modal="CSIamp", category=None, transform=None):
        self.root_dir = Path(root_dir)
        self.modal = modal
        self.transform = transform
        self.data_list = sorted(self.root_dir.rglob("*.mat"))

        if not self.root_dir.exists():
            raise FileNotFoundError(f"CSI dataset directory does not exist: {self.root_dir}")
        if not self.data_list:
            raise RuntimeError(f"No .mat files were found under {self.root_dir}")

        folders = sorted({path.parent.name for path in self.data_list})
        self.category = category or {folder: index for index, folder in enumerate(folders)}
        self.labels = [self.category[path.parent.name] for path in self.data_list]

        print(f"[INFO] Found {len(self.data_list)} samples in {len(self.category)} classes at {self.root_dir}.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample_path = self.data_list[index]
        label = self.category[sample_path.parent.name]
        csi = sio.loadmat(sample_path)[self.modal]
        csi = (csi - 42.3199) / 4.9802
        csi = csi[:, ::4]
        csi = csi.reshape(3, 114, 500)

        if self.transform:
            csi = self.transform(csi)

        return torch.as_tensor(csi, dtype=torch.float32), label


def build_csi_loaders(
    data_root,
    batch_size=16,
    val_size=0.1,
    seed=42,
    num_workers=8,
    modal="CSIamp",
    pin_memory=True,
):
    data_root = Path(data_root)
    train_set_full = NTUFiCSIDataset(data_root / "train_amp", modal=modal)
    labels = train_set_full.labels
    train_indices, val_indices = train_test_split(
        range(len(train_set_full)),
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    train_set = data.Subset(train_set_full, train_indices)
    val_set = data.Subset(train_set_full, val_indices)
    test_set = NTUFiCSIDataset(data_root / "test_amp", modal=modal, category=train_set_full.category)

    loaders = {
        "train": data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders, len(train_set_full.category)

