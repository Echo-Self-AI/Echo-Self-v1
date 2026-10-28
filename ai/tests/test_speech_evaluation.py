from pathlib import Path
import pytest
import torch

from echoself_ai.security.integrity import verify_checkpoint_integrity
from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import write_wav
from echoself_ai.voice.consent import VoiceConsentManager
from echoself_ai.voice.dataset import SpeechVocab
from echoself_ai.voice.train_acoustic import load_acoustic_checkpoint
from echoself_ai.voice.vocoder import GriffinLimVocoder, VoiceSynthesizer


def test_acoustic_checkpoint_evaluation(tmp_path: Path):
    ckpt_path = Path("checkpoints/acoustic_real/best_model.pt")
    if not ckpt_path.is_file():
        pytest.skip("Acoustic checkpoint not found; run train_acoustic first")

    # Verify sha256 sidecar
    assert verify_checkpoint_integrity(ckpt_path) is True

    vocab = SpeechVocab()
    config = AcousticModelConfig(
        vocab_size=vocab.size,
        n_mels=80,
        hidden_dim=64,
        num_layers=3,
        upsample_factor=8,
    )
    model = AcousticModel(config)
    load_acoustic_checkpoint(ckpt_path, device="cpu", model=model)
    model.eval()

    vocoder = GriffinLimVocoder(sample_rate=16000, n_mels=80, n_iters=16)
    synth = VoiceSynthesizer(acoustic_model=model, vocoder=vocoder)

    # Test utterance
    test_text = "The art of printing was invented in the fifteenth century."
    token_ids = torch.tensor([vocab.encode(test_text)], dtype=torch.long)

    out_wav = tmp_path / "eval_speech.wav"
    waveform = synth.synthesize(token_ids, output_wav_path=out_wav)

    # 1. Output file exists
    assert out_wav.is_file()
    assert out_wav.stat().st_size > 1000

    # 2. Finite numerical values
    assert torch.isfinite(waveform).all()

    # 3. Non-silent audio
    energy = torch.mean(waveform ** 2).item()
    assert energy > 1e-6, "Generated audio is silent"

    # 4. Clipping check
    peak = torch.max(torch.abs(waveform)).item()
    assert peak <= 1.05, f"Audio has severe clipping: peak={peak}"

    # 5. Duration check
    duration = waveform.size(1) / 16000.0
    assert duration > 1.0, f"Expected > 1.0s audio for 58-char sentence, got {duration:.2f}s"


def test_speech_synthesis_consent_enforcement(tmp_path: Path):
    manager = VoiceConsentManager(storage_dir=tmp_path / "consent")
    vocab = SpeechVocab()
    config = AcousticModelConfig(vocab_size=vocab.size, n_mels=80, hidden_dim=32, upsample_factor=4)
    model = AcousticModel(config)
    synth = VoiceSynthesizer(acoustic_model=model, consent_manager=manager)

    tokens = torch.tensor([[1, 2, 3, 4]])
    speaker_tensor = torch.tensor([1])

    # 1. Without consent: must raise PermissionError
    with pytest.raises(PermissionError, match="No active consent"):
        synth.synthesize(
            tokens,
            speaker_id=speaker_tensor,
            user_id="user_alice",
            raw_speaker_id="alice_voice",
        )

    # 2. Grant consent: synthesis must proceed
    manager.register_consent(user_id="user_alice", speaker_id="alice_voice", scope="synthesis")
    wav = synth.synthesize(
        tokens,
        speaker_id=speaker_tensor,
        user_id="user_alice",
        raw_speaker_id="alice_voice",
    )
    assert wav.shape[0] == 1
    assert torch.isfinite(wav).all()

    # 3. Revoke consent: synthesis must be blocked
    manager.revoke_consent(user_id="user_alice", speaker_id="alice_voice")
    with pytest.raises(PermissionError, match="No active consent"):
        synth.synthesize(
            tokens,
            speaker_id=speaker_tensor,
            user_id="user_alice",
            raw_speaker_id="alice_voice",
        )
