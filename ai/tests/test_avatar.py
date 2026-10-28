"""Unit tests for Avatar identity profile, neural generator, and consent engine."""
from pathlib import Path
import pytest
import torch

from echoself_ai.vision.avatar import (
    AvatarGenerator,
    AvatarGeneratorConfig,
    AvatarProfile,
    UnauthorizedAvatarError,
    VisualConsentManager,
    compute_identity_consistency_loss,
)
from echoself_ai.vision.encoder import VisualEncoder, VisualEncoderConfig


def test_visual_consent_manager_lifecycle(tmp_path: Path):
    manager = VisualConsentManager(storage_dir=tmp_path / "consent")

    # 1. Initially no consent
    assert not manager.verify_consent("user_123", "avatar_alpha", scope="generation")

    # 2. Register consent
    record = manager.register_consent("user_123", "avatar_alpha", scope="generation")
    assert record.consent_granted is True
    assert record.revoked is False
    assert len(record.verification_hash) == 64
    assert manager.verify_consent("user_123", "avatar_alpha", scope="generation")
    assert not manager.verify_consent("user_123", "avatar_alpha", scope="training")

    # 3. Reload manager from disk
    manager2 = VisualConsentManager(storage_dir=tmp_path / "consent")
    assert manager2.verify_consent("user_123", "avatar_alpha", scope="generation")

    # 4. Revocation
    revoked = manager.revoke_consent("user_123", "avatar_alpha")
    assert revoked is True
    assert not manager.verify_consent("user_123", "avatar_alpha", scope="generation")

    # 5. GDPR Purge
    data_dir = tmp_path / "avatar_data"
    data_dir.mkdir()
    (data_dir / "avatar_alpha_weights.pt").write_text("dummy")
    deleted = manager.purge_avatar_data("user_123", "avatar_alpha", data_dirs=[data_dir])
    assert deleted >= 2
    assert not (data_dir / "avatar_alpha_weights.pt").exists()


def test_avatar_profile_serialization(tmp_path: Path):
    emb = torch.randn(128)
    emb = emb / emb.norm()
    profile = AvatarProfile(
        user_id="user_abc",
        avatar_id="avatar_phoenix",
        name="Phoenix Avatar",
        embedding=emb,
        metadata={"style": "cyberpunk", "resolution": 64},
    )

    # Cosine similarity self-match
    sim = profile.compute_similarity(emb)
    assert abs(sim - 1.0) < 1e-4

    # Save and load
    file_path = tmp_path / "phoenix.json"
    profile.save(file_path)
    loaded = AvatarProfile.load(file_path)

    assert loaded.user_id == "user_abc"
    assert loaded.avatar_id == "avatar_phoenix"
    assert loaded.metadata["style"] == "cyberpunk"
    assert loaded.embedding.shape == (128,)
    assert abs(loaded.compute_similarity(emb) - 1.0) < 1e-4


def test_avatar_generator_forward():
    config = AvatarGeneratorConfig(
        embedding_dim=64,
        condition_dim=16,
        base_dim=16,
        image_size=64,
    )
    generator = AvatarGenerator(config)

    batch_size = 2
    emb = torch.randn(batch_size, 64)
    cond = torch.randn(batch_size, 16)

    out = generator(emb, cond)
    assert out.shape == (batch_size, 3, 64, 64)
    # Output values bounded in [-1.0, 1.0] due to Tanh
    assert out.min() >= -1.0
    assert out.max() <= 1.0

    # Default condition vector handles 1D / None
    out_single = generator(emb[0])
    assert out_single.shape == (1, 3, 64, 64)


def test_avatar_generator_consent_gating(tmp_path: Path):
    manager = VisualConsentManager(storage_dir=tmp_path / "consent")
    config = AvatarGeneratorConfig(embedding_dim=32, condition_dim=8, base_dim=8)
    generator = AvatarGenerator(config)
    emb = torch.randn(1, 32)

    # 1. Unconsented generation raises UnauthorizedAvatarError
    with pytest.raises(UnauthorizedAvatarError):
        generator.generate(
            user_id="user_404",
            avatar_id="avatar_unauthorized",
            identity_embedding=emb,
            consent_manager=manager,
        )

    # 2. Register consent -> generation succeeds
    manager.register_consent("user_404", "avatar_unauthorized", scope="generation")
    out = generator.generate(
        user_id="user_404",
        avatar_id="avatar_unauthorized",
        identity_embedding=emb,
        consent_manager=manager,
    )
    assert out.shape == (1, 3, 64, 64)

    # 3. Revoke consent -> generation fails again
    manager.revoke_consent("user_404", "avatar_unauthorized")
    with pytest.raises(UnauthorizedAvatarError):
        generator.generate(
            user_id="user_404",
            avatar_id="avatar_unauthorized",
            identity_embedding=emb,
            consent_manager=manager,
        )


def test_identity_consistency_loss_and_gradients():
    encoder = VisualEncoder(VisualEncoderConfig(embedding_dim=32, base_dim=8))
    generator = AvatarGenerator(AvatarGeneratorConfig(embedding_dim=32, condition_dim=8, base_dim=8))

    target_emb = torch.randn(2, 32)
    target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

    generated = generator(target_emb)
    loss = compute_identity_consistency_loss(generated, target_emb, encoder)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # Test gradient backprop through generator
    loss.backward()
    grad_found = any(p.grad is not None and p.grad.norm() > 0 for p in generator.parameters())
    assert grad_found
