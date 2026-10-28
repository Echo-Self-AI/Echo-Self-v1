"""Prompt-to-video neural pipeline with identity conditioning and privacy consent gating."""
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from echoself_ai.video.frames import FrameSequence
from echoself_ai.video.temporal import (
    TemporalConvBlock,
    compute_temporal_consistency_loss,
)
from echoself_ai.vision.avatar import (
    AvatarProfile,
    UnauthorizedAvatarError,
    VisualConsentManager,
)
from echoself_ai.vision.encoder import VisualEncoder


@dataclass
class PromptToVideoConfig:
    """Hyperparameters for prompt-to-video generation pipeline."""

    prompt_embed_dim: int = 64
    identity_embed_dim: int = 128
    base_dim: int = 32
    image_size: int = 64
    default_num_frames: int = 16
    default_fps: int = 16


class PromptTextEncoder(nn.Module):
    """Encodes arbitrary text prompt strings into dense semantic condition vectors."""

    def __init__(self, embed_dim: int = 64, vocab_size: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, prompt: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode prompt string into a (1, embed_dim) vector."""
        if not prompt:
            prompt = "idle"

        byte_indices = [b % 256 for b in prompt.encode("utf-8")]
        tokens = torch.tensor(byte_indices, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
        emb = self.embedding(tokens).permute(0, 2, 1)  # (1, embed_dim, L)
        features = self.conv(emb)  # (1, embed_dim, L)
        # Global mean pooling across text sequence
        pooled = features.mean(dim=-1)  # (1, embed_dim)
        return F.normalize(pooled, p=2, dim=-1)


class PromptToVideoGenerator(nn.Module):
    """From-scratch neural prompt-to-video generator.

    Synthesizes spatially coherent, temporally smooth video frame sequences
    conditioned on text prompts and optional personal avatar identity embeddings.
    """

    def __init__(self, config: Optional[PromptToVideoConfig] = None) -> None:
        super().__init__()
        self.config = config or PromptToVideoConfig()
        c = self.config.base_dim
        total_cond_dim = self.config.prompt_embed_dim + self.config.identity_embed_dim

        # Text prompt encoder
        self.text_encoder = PromptTextEncoder(embed_dim=self.config.prompt_embed_dim)

        # Motion and temporal trajectory generator
        # Projects condition + time step embeddings to frame latent representations
        self.temporal_mlp = nn.Sequential(
            nn.Linear(total_cond_dim + 16, 4 * c * 4 * 4),
            nn.GELU(),
        )

        # Spatial upsampling stages: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 4 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )

        # Spatiotemporal smoother across sequence of frames (2D+1D convolution)
        self.temporal_smoother = TemporalConvBlock(c, c, kernel_size=3)

        # Final projection to RGB channels in [-1.0, 1.0]
        self.to_rgb = nn.Sequential(
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _get_time_embeddings(self, num_frames: int, device: torch.device) -> torch.Tensor:
        """Create sinusoidal temporal positional embeddings for video frames."""
        time_steps = torch.arange(num_frames, device=device, dtype=torch.float32)  # (T,)
        # Normalized time phase in [0, 2*pi]
        phases = (time_steps / max(1, num_frames - 1)) * (2.0 * math.pi)

        # 16-dimensional harmonic embedding
        harmonics: list[torch.Tensor] = []
        for freq in [1, 2, 4, 8]:
            harmonics.append(torch.sin(phases * freq))
            harmonics.append(torch.cos(phases * freq))
            harmonics.append(torch.sin(phases * (freq + 0.5)))
            harmonics.append(torch.cos(phases * (freq + 0.5)))

        return torch.stack(harmonics, dim=-1)  # (T, 16)

    def forward(
        self,
        prompt: str,
        identity_embedding: Optional[torch.Tensor] = None,
        num_frames: int = 16,
    ) -> torch.Tensor:
        """Synthesize video frames tensor.

        Args:
            prompt: Text prompt string.
            identity_embedding: Optional tensor (1, identity_embed_dim) or (identity_embed_dim,).
            num_frames: Number of temporal frames to synthesize.

        Returns:
            Video tensor of shape (1, num_frames, 3, image_size, image_size).
        """
        device = next(self.parameters()).device
        c = self.config.base_dim

        # 1. Encode text prompt
        text_latent = self.text_encoder(prompt, device=device)  # (1, prompt_embed_dim)

        # 2. Process identity conditioning
        if identity_embedding is None:
            identity_embedding = torch.zeros(
                1,
                self.config.identity_embed_dim,
                device=device,
                dtype=torch.float32,
            )
        elif identity_embedding.dim() == 1:
            identity_embedding = identity_embedding.unsqueeze(0).to(device)

        # 3. Fuse conditions
        cond = torch.cat([text_latent, identity_embedding], dim=-1)  # (1, total_cond_dim)
        cond_expanded = cond.expand(num_frames, -1)  # (T, total_cond_dim)

        # 4. Temporal positional features
        time_emb = self._get_time_embeddings(num_frames, device)  # (T, 16)
        temporal_input = torch.cat([cond_expanded, time_emb], dim=-1)  # (T, total_cond_dim + 16)

        # 5. Project to spatial feature map: (T, 4 * c, 4, 4)
        x = self.temporal_mlp(temporal_input).view(num_frames, 4 * c, 4, 4)

        # 6. Spatial upsampling
        x = self.up1(x)  # (T, 4 * c, 8, 8)
        x = self.up2(x)  # (T, 2 * c, 16, 16)
        x = self.up3(x)  # (T, c, 32, 32)
        x = self.up4(x)  # (T, c, 64, 64)

        # 7. Spatiotemporal refinement (B=1, T, C, H, W)
        x_video = x.unsqueeze(0)  # (1, T, c, 64, 64)
        x_smoothed = self.temporal_smoother(x_video)  # (1, T, c, 64, 64)

        # 8. Render to RGB [-1.0, 1.0]
        cur_h, cur_w = x_smoothed.shape[3], x_smoothed.shape[4]
        x_flat = x_smoothed.view(num_frames, c, cur_h, cur_w)
        rgb_flat = self.to_rgb(x_flat)
        target_size = (self.config.image_size, self.config.image_size)
        if (cur_h, cur_w) != target_size:
            rgb_flat = F.interpolate(rgb_flat, size=target_size, mode="bilinear", align_corners=False)
        rgb_video = rgb_flat.view(1, num_frames, 3, self.config.image_size, self.config.image_size)

        return rgb_video


    def generate(
        self,
        prompt: str,
        user_id: Optional[str] = None,
        avatar_id: Optional[str] = None,
        identity_embedding: Optional[torch.Tensor] = None,
        avatar_profile: Optional[AvatarProfile] = None,
        consent_manager: Optional[VisualConsentManager] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> FrameSequence:
        """Generate animated FrameSequence from prompt with strict consent gating."""
        target_num_frames = num_frames or self.config.default_num_frames
        target_fps = fps or self.config.default_fps

        # Resolve avatar identity if provided
        if avatar_profile is not None:
            identity_embedding = avatar_profile.embedding
            if user_id is None:
                user_id = avatar_profile.user_id
            if avatar_id is None:
                avatar_id = avatar_profile.avatar_id

        # Enforce consent gating if personalized identity is specified
        if user_id is not None and avatar_id is not None:
            if consent_manager is None:
                raise UnauthorizedAvatarError(
                    f"Consent manager required for personalized video generation "
                    f"(user='{user_id}', avatar='{avatar_id}')"
                )
            # Must have active consent covering generation or animation
            has_consent = consent_manager.verify_consent(
                user_id, avatar_id, scope="generation"
            ) or consent_manager.verify_consent(user_id, avatar_id, scope="animation")
            if not has_consent:
                raise UnauthorizedAvatarError(
                    f"Video generation blocked: user consent not found or revoked "
                    f"for user='{user_id}', avatar='{avatar_id}'."
                )

        self.eval()
        with torch.no_grad():
            video_tensor = self.forward(
                prompt=prompt,
                identity_embedding=identity_embedding,
                num_frames=target_num_frames,
            )

        # (1, T, 3, H, W) -> FrameSequence
        return FrameSequence(video_tensor.squeeze(0), fps=target_fps)


def compute_video_generation_loss(
    video_tensor: torch.Tensor,
    target_identity: Optional[torch.Tensor] = None,
    visual_encoder: Optional[VisualEncoder] = None,
    temporal_weight: float = 0.5,
) -> torch.Tensor:
    """Compute combined video training loss: identity preservation + temporal consistency."""
    loss_temporal = compute_temporal_consistency_loss(video_tensor)

    loss_id = torch.tensor(0.0, device=video_tensor.device)
    if target_identity is not None and visual_encoder is not None:
        # Sample middle frame for identity preservation
        mid_idx = video_tensor.shape[1] // 2
        mid_frame = video_tensor[:, mid_idx]
        pred_emb = visual_encoder(mid_frame)
        cos_sim = F.cosine_similarity(pred_emb, target_identity, dim=-1)
        loss_id = 1.0 - cos_sim.mean()

    return loss_id + temporal_weight * loss_temporal
