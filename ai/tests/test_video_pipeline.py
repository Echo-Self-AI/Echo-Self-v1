"""Unit tests for prompt-to-video neural pipeline and consent gating."""
from pathlib import Path
import pytest
import torch

from echoself_ai.video.frames import FrameSequence
from echoself_ai.video.pipeline import (
    PromptTextEncoder,
    PromptToVideoConfig,
    PromptToVideoGenerator,
    compute_video_generation_loss,
)
from echoself_ai.vision.avatar import (
    AvatarProfile,
    UnauthorizedAvatarError,
    VisualConsentManager,
)
from echoself_ai.vision.encoder import VisualEncoder, VisualEncoderConfig


def test_prompt_text_encoder():
    encoder = PromptTextEncoder(embed_dim=32)

    emb = encoder("avatar talking calmly with a smile")
    assert emb.shape == (1, 32)
    # L2-normalized vector
    assert abs(emb.norm().item() - 1.0) < 1e-4

    # Empty prompt fallback
    emb_empty = encoder("")
    assert emb_empty.shape == (1, 32)


def test_prompt_to_video_forward():
    config = PromptToVideoConfig(
        prompt_embed_dim=16,
        identity_embed_dim=32,
        base_dim=8,
        image_size=64,
        default_num_frames=4,
    )
    generator = PromptToVideoGenerator(config)

    video = generator("a smiling face nodding", num_frames=4)
    # Shape: (1, T, 3, H, W)
    assert video.shape == (1, 4, 3, 64, 64)
    assert video.min() >= -1.0
    assert video.max() <= 1.0

    # Gradient flow
    loss = video.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in generator.parameters())


def test_prompt_to_video_unconditioned_generate():
    config = PromptToVideoConfig(
        prompt_embed_dim=16,
        identity_embed_dim=16,
        base_dim=8,
        default_num_frames=4,
        default_fps=8,
    )
    generator = PromptToVideoGenerator(config)

    seq = generator.generate("cinematic sunrise portrait", num_frames=4, fps=8)
    assert isinstance(seq, FrameSequence)
    assert seq.num_frames == 4
    assert seq.fps == 8
    assert seq.height == 64
    assert seq.width == 64


def test_prompt_to_video_consent_gating(tmp_path: Path):
    manager = VisualConsentManager(storage_dir=tmp_path / "consent")
    config = PromptToVideoConfig(
        prompt_embed_dim=16,
        identity_embed_dim=16,
        base_dim=8,
        default_num_frames=4,
    )
    generator = PromptToVideoGenerator(config)

    emb = torch.randn(16)
    profile = AvatarProfile(
        user_id="user_video_01",
        avatar_id="avatar_sam",
        name="Sam Avatar",
        embedding=emb,
    )

    # 1. Generation without consent manager raises error
    with pytest.raises(UnauthorizedAvatarError):
        generator.generate(
            prompt="avatar waving",
            avatar_profile=profile,
            consent_manager=None,
        )

    # 2. Generation with unconsented profile raises error
    with pytest.raises(UnauthorizedAvatarError):
        generator.generate(
            prompt="avatar waving",
            avatar_profile=profile,
            consent_manager=manager,
        )

    # 3. Register consent -> generation succeeds
    manager.register_consent("user_video_01", "avatar_sam", scope="animation")
    seq = generator.generate(
        prompt="avatar waving",
        avatar_profile=profile,
        consent_manager=manager,
        num_frames=4,
    )
    assert isinstance(seq, FrameSequence)
    assert seq.num_frames == 4

    # 4. Revocation immediately blocks generation
    manager.revoke_consent("user_video_01", "avatar_sam")
    with pytest.raises(UnauthorizedAvatarError):
        generator.generate(
            prompt="avatar waving",
            avatar_profile=profile,
            consent_manager=manager,
        )


def test_compute_video_generation_loss():
    encoder = VisualEncoder(VisualEncoderConfig(embedding_dim=16, base_dim=8))
    video = torch.randn(1, 4, 3, 32, 32, requires_grad=True)
    target_id = torch.randn(1, 16)
    target_id = target_id / target_id.norm()

    loss = compute_video_generation_loss(
        video_tensor=video,
        target_identity=target_id,
        visual_encoder=encoder,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0

    loss.backward()
    assert video.grad is not None
