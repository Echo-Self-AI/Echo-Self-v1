from pathlib import Path
import pytest
import torch

from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.consent import VoiceConsentManager


def test_acoustic_model_forward_shape():
    config = AcousticModelConfig(vocab_size=30, n_mels=80, hidden_dim=32, upsample_factor=4)
    model = AcousticModel(config)

    # 2 sequences of 10 tokens
    tokens = torch.randint(0, 30, (2, 10))
    mel, loss = model(tokens)

    # 10 tokens * upsample 4 = 40 frames
    assert mel.shape == (2, 80, 40)
    assert loss is None


def test_acoustic_model_forward_with_loss():
    config = AcousticModelConfig(vocab_size=30, n_mels=80, hidden_dim=32, upsample_factor=4)
    model = AcousticModel(config)

    tokens = torch.randint(0, 30, (2, 10))
    speaker_id = torch.tensor([0, 1])
    target = torch.randn(2, 80, 40)
    mel, loss = model(tokens, speaker_id=speaker_id, target_mel=target)


    assert loss is not None
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"


def test_acoustic_model_speaker_conditioning():
    config = AcousticModelConfig(vocab_size=30, n_mels=80, hidden_dim=32, num_speakers=2)
    model = AcousticModel(config)
    model.eval()

    tokens = torch.tensor([[1, 2, 3, 4]])
    spk0 = torch.tensor([0])
    spk1 = torch.tensor([1])

    with torch.no_grad():
        mel0, _ = model(tokens, speaker_id=spk0)
        mel1, _ = model(tokens, speaker_id=spk1)

    # Predictions under different speaker IDs must differ
    assert not torch.allclose(mel0, mel1)


def test_acoustic_model_consent_gating(tmp_path: Path):
    manager = VoiceConsentManager(storage_dir=tmp_path / "consent")
    config = AcousticModelConfig(vocab_size=30, n_mels=80)
    model = AcousticModel(config)
    tokens = torch.randint(0, 30, (1, 5))

    # Without consent: PermissionError
    with pytest.raises(PermissionError):
        model(
            tokens,
            user_id="user_unauthorized",
            raw_speaker_id="spk_99",
            consent_manager=manager,
        )

    # Grant consent
    manager.register_consent("user_authorized", "spk_1", scope="synthesis")
    # With consent: succeeds
    mel, _ = model(
        tokens,
        user_id="user_authorized",
        raw_speaker_id="spk_1",
        consent_manager=manager,
    )
    assert mel.shape[1] == 80
