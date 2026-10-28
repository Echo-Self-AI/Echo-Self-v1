from pathlib import Path
import pytest
import torch

from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import load_wav
from echoself_ai.voice.consent import VoiceConsentManager
from echoself_ai.voice.vocoder import GriffinLimVocoder, VoiceSynthesizer


def test_vocoder_mel_to_linear():
    vocoder = GriffinLimVocoder(sample_rate=16000, n_fft=512, n_mels=80)
    mel = torch.randn(2, 80, 20)
    linear = vocoder.mel_to_linear(mel)

    assert linear.shape == (2, 257, 20)
    assert (linear > 0).all()


def test_griffin_lim_waveform_reconstruction():
    vocoder = GriffinLimVocoder(
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512,
        n_mels=80,
        n_iters=8,
    )
    mel = torch.randn(1, 80, 25)
    waveform = vocoder(mel)

    assert waveform.dim() == 2
    assert waveform.size(0) == 1
    assert waveform.size(1) > 2000
    assert not torch.isnan(waveform).any()


def test_voice_synthesizer_end_to_end(tmp_path: Path):
    model_cfg = AcousticModelConfig(vocab_size=20, n_mels=80, hidden_dim=16, upsample_factor=2)
    acoustic_model = AcousticModel(model_cfg)
    vocoder = GriffinLimVocoder(
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512,
        n_mels=80,
        n_iters=4,
    )
    synth = VoiceSynthesizer(acoustic_model, vocoder=vocoder)

    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    wav_path = tmp_path / "output.wav"

    wav = synth.synthesize(tokens, output_wav_path=wav_path)

    assert wav.shape[0] == 1
    assert wav_path.exists()

    loaded, sr = load_wav(wav_path)
    assert sr == 16000
    assert len(loaded) > 0


def test_voice_synthesizer_consent_enforcement(tmp_path: Path):
    manager = VoiceConsentManager(storage_dir=tmp_path / "consent")
    model_cfg = AcousticModelConfig(vocab_size=20, n_mels=80, hidden_dim=16)
    acoustic_model = AcousticModel(model_cfg)
    synth = VoiceSynthesizer(acoustic_model, consent_manager=manager)
    tokens = torch.tensor([[1, 2]])

    # Without consent: PermissionError
    with pytest.raises(PermissionError):
        synth.synthesize(
            tokens,
            user_id="user_unauth",
            raw_speaker_id="spk_clone",
        )

    # Grant consent
    manager.register_consent("user_auth", "spk_clone", scope="synthesis")
    # With consent: succeeds
    wav = synth.synthesize(
        tokens,
        user_id="user_auth",
        raw_speaker_id="spk_clone",
    )
    assert wav.size(0) == 1
