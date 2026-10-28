from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VisualEncoderConfig:
    """Hyperparameters for visual representation extractor."""

    in_channels: int = 3
    base_dim: int = 32
    embedding_dim: int = 128
    num_stages: int = 3
    normalize_embeddings: bool = True


class ResConvBlock(nn.Module):
    """Residual convolutional block with BatchNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class VisualEncoder(nn.Module):
    """Convolutional visual feature extractor mapping images to fixed-size identity embeddings.

    Built strictly from PyTorch primitives, producing L2-normalized representations
    suitable for avatar identity consistency tracking and visual similarity matching.
    """

    def __init__(self, config: Optional[VisualEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or VisualEncoderConfig()
        c = self.config.base_dim

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.in_channels, c, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )

        # Multi-scale residual stages
        self.stage1 = ResConvBlock(c, c, stride=1)
        self.stage2 = ResConvBlock(c, 2 * c, stride=2)
        self.stage3 = ResConvBlock(2 * c, 4 * c, stride=2)

        # Global spatial pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding projection head
        self.proj = nn.Linear(4 * c, self.config.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual embeddings from images.

        Args:
            images: Tensor of shape (batch_size, 3, H, W) normalized to [-1.0, 1.0].

        Returns:
            Tensor of shape (batch_size, embedding_dim).
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        embeddings = self.proj(x)

        if self.config.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @staticmethod
    def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two sets of normalized embeddings."""
        return F.cosine_similarity(emb1, emb2, dim=-1)
