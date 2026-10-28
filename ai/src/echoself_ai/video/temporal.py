"""Temporal consistency, motion dynamics, and neural frame interpolation."""
from dataclasses import dataclass
from typing import Optional, Union

import torch

from torch import nn
import torch.nn.functional as F

from echoself_ai.video.frames import FrameSequence


def compute_temporal_consistency_loss(
    frames: torch.Tensor,
    l1_weight: float = 1.0,
    smoothness_weight: float = 0.5,
) -> torch.Tensor:
    """Compute temporal consistency loss penalizing flickering and jerky motion.

    Args:
        frames: Video tensor of shape (B, T, C, H, W) or (T, C, H, W) in range [-1.0, 1.0].
        l1_weight: Weight for first-order temporal difference (frame-to-frame change).
        smoothness_weight: Weight for second-order acceleration (jerk penalty).

    Returns:
        Scalar loss tensor.
    """
    if frames.dim() == 4:
        # (T, C, H, W) -> (1, T, C, H, W)
        frames = frames.unsqueeze(0)

    if frames.dim() != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got {frames.dim()}D tensor: {frames.shape}")

    t = frames.shape[1]
    if t < 2:
        return torch.tensor(0.0, device=frames.device, dtype=frames.dtype)

    # First-order difference: velocity (T - 1)
    diff1 = frames[:, 1:] - frames[:, :-1]
    loss_diff1 = diff1.abs().mean()

    # Second-order difference: acceleration (T - 2)
    if t >= 3 and smoothness_weight > 0.0:
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        loss_diff2 = (diff2**2).mean()
    else:
        loss_diff2 = torch.tensor(0.0, device=frames.device, dtype=frames.dtype)

    return l1_weight * loss_diff1 + smoothness_weight * loss_diff2


def compute_motion_magnitude(frames: torch.Tensor) -> torch.Tensor:
    """Compute spatial motion magnitude map across adjacent frames.

    Args:
        frames: Video tensor (B, T, C, H, W) or (T, C, H, W).

    Returns:
        Tensor of shape (B, T-1, 1, H, W) with motion magnitude values >= 0.
    """
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)

    diff = frames[:, 1:] - frames[:, :-1]  # (B, T-1, C, H, W)
    magnitude = torch.sqrt((diff**2).sum(dim=2, keepdim=True) + 1e-8)
    return magnitude


class TemporalConvBlock(nn.Module):
    """Separable (2D + 1D) Spatiotemporal Convolutional Block.

    Applies spatial Conv2D (1 x K x K) across spatial planes, followed by
    temporal Conv1D (K x 1 x 1) across time frames with residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial 2D convolution
        self.spatial_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(out_channels)

        # Temporal 1D convolution across time dimension
        self.temporal_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm1d(out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through (2D + 1D) spatiotemporal block.

        Args:
            x: Tensor of shape (B, T, C, H, W)

        Returns:
            Tensor of shape (B, T, out_channels, H, W)
        """
        b, t, c, h, w = x.shape

        # Reshape to (B * T, C, H, W) for spatial convolution
        x_flat = x.view(b * t, c, h, w)
        residual = self.shortcut(x_flat)
        out_spatial = F.gelu(self.spatial_bn(self.spatial_conv(x_flat)))
        out_spatial = out_spatial + residual

        # Reshape to (B * H * W, out_channels, T) for temporal 1D convolution
        out_spatial = out_spatial.view(b, t, self.out_channels, h, w)
        # Permute to (B, H, W, out_channels, T) -> (B * H * W, out_channels, T)
        out_temp = out_spatial.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, self.out_channels, t)
        out_temp = F.gelu(self.temporal_bn(self.temporal_conv(out_temp)))

        # Reshape back to (B, H, W, out_channels, T) -> (B, T, out_channels, H, W)
        out = out_temp.view(b, h, w, self.out_channels, t).permute(0, 4, 3, 1, 2).contiguous()
        return out


@dataclass
class FrameInterpolatorConfig:
    """Configuration for neural frame interpolator."""

    in_channels: int = 7  # 3 (frame A) + 3 (frame B) + 1 (alpha coordinate map)
    base_dim: int = 32
    out_channels: int = 3


class FrameInterpolator(nn.Module):
    """Neural frame interpolator predicting intermediate in-between frames.

    Maps (frame_a, frame_b, alpha) to interpolated frame x_alpha in [-1.0, 1.0].
    """

    def __init__(self, config: Optional[FrameInterpolatorConfig] = None) -> None:
        super().__init__()
        self.config = config or FrameInterpolatorConfig()
        c = self.config.base_dim

        # Encoder stages
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.config.in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.GELU(),
        )

        # Decoder stages with skip connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(4 * c, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )

        # Output projection predicting residual delta blended with linear interpolation
        self.to_rgb = nn.Sequential(
            nn.Conv2d(2 * c, self.config.out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        frame_a: torch.Tensor,
        frame_b: torch.Tensor,
        alpha: Union[float, torch.Tensor] = 0.5,
    ) -> torch.Tensor:
        """Interpolate between two frames.

        Args:
            frame_a: Tensor of shape (B, 3, H, W) in [-1.0, 1.0].
            frame_b: Tensor of shape (B, 3, H, W) in [-1.0, 1.0].
            alpha: Temporal blend scalar or tensor of shape (B,) or (B, 1, 1, 1) in [0.0, 1.0].

        Returns:
            Interpolated frame tensor of shape (B, 3, H, W) in [-1.0, 1.0].
        """
        if frame_a.dim() == 3:
            frame_a = frame_a.unsqueeze(0)
        if frame_b.dim() == 3:
            frame_b = frame_b.unsqueeze(0)

        b, _, h, w = frame_a.shape

        if isinstance(alpha, (int, float)):
            alpha_tensor = torch.full((b, 1, h, w), float(alpha), device=frame_a.device)
        elif isinstance(alpha, torch.Tensor):
            if alpha.dim() == 0 or alpha.dim() == 1:
                alpha_tensor = alpha.view(-1, 1, 1, 1).expand(b, 1, h, w)
            else:
                alpha_tensor = alpha

        # Linear blend prior
        linear_blend = (1.0 - alpha_tensor) * frame_a + alpha_tensor * frame_b

        # Concat inputs: (B, 7, H, W)
        net_in = torch.cat([frame_a, frame_b, alpha_tensor], dim=1)

        # Multi-scale U-Net
        e1 = self.enc1(net_in)  # (B, c, H, W)
        e2 = self.enc2(e1)      # (B, 2c, H/2, W/2)
        e3 = self.enc3(e2)      # (B, 4c, H/4, W/4)

        d2 = self.dec2(e3)      # (B, 2c, H/2, W/2)
        d1 = self.dec1(torch.cat([d2, e2], dim=1))  # (B, c, H, W)

        residual_delta = self.to_rgb(torch.cat([d1, e1], dim=1))

        # Output blended frame clamped to valid range
        out = (linear_blend + 0.2 * residual_delta).clamp(-1.0, 1.0)
        return out

    def interpolate_sequence(
        self,
        start_frame: torch.Tensor,
        end_frame: torch.Tensor,
        num_intermediate_frames: int = 6,
        fps: int = 16,
    ) -> FrameSequence:
        """Generate smooth temporal sequence between start_frame and end_frame."""
        if start_frame.dim() == 4 and start_frame.shape[0] == 1:
            start_frame = start_frame.squeeze(0)
        if end_frame.dim() == 4 and end_frame.shape[0] == 1:
            end_frame = end_frame.squeeze(0)

        frames = [start_frame]
        total_steps = num_intermediate_frames + 1

        self.eval()
        with torch.no_grad():
            for i in range(1, total_steps):
                alpha = i / total_steps
                interp = self.forward(start_frame, end_frame, alpha=alpha)
                frames.append(interp.squeeze(0))

        frames.append(end_frame)
        stacked = torch.stack(frames, dim=0)
        return FrameSequence(stacked, fps=fps)
