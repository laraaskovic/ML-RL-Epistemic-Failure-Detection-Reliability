import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x - MNIST_MEAN) / MNIST_STD


def base_transform() -> transforms.Compose:
    return transforms.ToTensor()


def classification_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def get_classification_loaders(
    batch_size: int = 128, data_root: str = "data"
) -> Tuple[DataLoader, DataLoader]:
    train_set = datasets.MNIST(
        root=data_root, train=True, download=True, transform=classification_transforms()
    )
    test_set = datasets.MNIST(
        root=data_root, train=False, download=True, transform=classification_transforms()
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def apply_corruption(x: torch.Tensor, kind: str, severity: float) -> torch.Tensor:
    if kind == "rotate":
        angle = 20 + severity * 60
        x = TF.rotate(x, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    elif kind == "noise":
        noise = torch.randn_like(x) * (0.3 + 0.5 * severity)
        x = torch.clamp(x + noise, 0.0, 1.0)
    elif kind == "occlude":
        size = int(6 + severity * 12)
        cx = random.randint(0, x.shape[-1] - 1)
        cy = random.randint(0, x.shape[-2] - 1)
        x = x.clone()
        x[..., max(0, cy - size) : cy + size, max(0, cx - size) : cx + size] = 0.0
    elif kind == "shift":
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        grid = torch.meshgrid(
            torch.arange(x.shape[-2]), torch.arange(x.shape[-1]), indexing="ij"
        )
        gy = torch.clamp(grid[0] - dy, 0, x.shape[-2] - 1)
        gx = torch.clamp(grid[1] - dx, 0, x.shape[-1] - 1)
        x = x[..., gy, gx]
    return x


def choose_corruption(kinds: List[str]) -> Tuple[str, float]:
    kind = random.choice(kinds)
    severity = random.random()
    return kind, severity


class CorruptedMNIST(Dataset):
    def __init__(
        self,
        split: str,
        data_root: str = "data",
        corruption_pool: List[str] = None,
        corruption_chance: float = 0.7,
    ) -> None:
        self.base = datasets.MNIST(
            root=data_root, train=split == "train", download=True, transform=base_transform()
        )
        self.corruption_pool = corruption_pool or ["clean", "rotate", "noise", "occlude", "shift"]
        self.corruption_chance = corruption_chance

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        corruption = "clean"
        severity = 0.0
        if random.random() < self.corruption_chance:
            corruption = random.choice([c for c in self.corruption_pool if c != "clean"])
            severity = random.random()
            x = apply_corruption(x, corruption, severity)
        x = normalize_tensor(x)
        return x, y, corruption, severity


def get_corrupted_loader(
    split: str, batch_size: int = 256, data_root: str = "data", corruption_chance: float = 0.7
) -> DataLoader:
    dataset = CorruptedMNIST(split=split, data_root=data_root, corruption_chance=corruption_chance)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=2)
