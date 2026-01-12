import torch
from torch import nn
import torch.nn.functional as F


class PrimaryCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, capture: bool = False):
        x = self.conv1(x)
        a1 = F.relu(self.bn1(x))
        x = F.max_pool2d(a1, 2)

        x = self.conv2(x)
        a2 = F.relu(self.bn2(x))
        x = F.max_pool2d(a2, 2)

        x = torch.flatten(x, 1)
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)


        #allows for more inputs for the auditor nn
        if capture:
            pooled1 = F.adaptive_avg_pool2d(a1, 1).flatten(1)
            pooled2 = F.adaptive_avg_pool2d(a2, 1).flatten(1)
            latent = torch.cat([pooled1, pooled2, hidden], dim=1)
            return logits, latent
        return logits

    @property
    def latent_dim(self) -> int:
        return 32 + 64 + 128
