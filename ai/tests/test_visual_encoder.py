import pytest
import torch

from echoself_ai.vision.encoder import VisualEncoder, VisualEncoderConfig
from echoself_ai.vision.image_ops import generate_synthetic_image


def test_visual_encoder_output_shape():
    cfg = VisualEncoderConfig(base_dim=16, embedding_dim=64)
    encoder = VisualEncoder(cfg)

    images = torch.randn(2, 3, 64, 64)
    embs = encoder(images)

    assert embs.shape == (2, 64)


def test_visual_encoder_l2_normalization():
    cfg = VisualEncoderConfig(base_dim=16, embedding_dim=64, normalize_embeddings=True)
    encoder = VisualEncoder(cfg)

    images = torch.randn(4, 3, 64, 64)
    embs = encoder(images)

    norms = torch.norm(embs, p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)


def test_visual_encoder_similarity():
    cfg = VisualEncoderConfig(base_dim=16, embedding_dim=64)
    encoder = VisualEncoder(cfg)
    encoder.eval()

    img1 = generate_synthetic_image(64, 64, pattern="checkerboard")
    img2 = generate_synthetic_image(64, 64, pattern="gradient")

    with torch.no_grad():
        emb1 = encoder(img1)
        emb2 = encoder(img2)
        sim_self = encoder.compute_similarity(emb1, emb1).item()
        sim_cross = encoder.compute_similarity(emb1, emb2).item()

    # Self-similarity must be 1.0
    assert pytest.approx(sim_self, abs=1e-4) == 1.0
    # Cross pattern similarity should be strictly less than 1.0
    assert sim_cross < 1.0
    assert not torch.allclose(emb1, emb2)



def test_visual_encoder_gradient_flow():
    cfg = VisualEncoderConfig(base_dim=16, embedding_dim=32)
    encoder = VisualEncoder(cfg)

    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    embs = encoder(images)

    loss = embs.sum()
    loss.backward()

    assert images.grad is not None
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any()
