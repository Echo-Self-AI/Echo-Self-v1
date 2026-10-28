"""Voice AI subsystem for EchoSelf."""
from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import (
    AudioConfig,
    generate_synthetic_tone,
    load_wav,
    normalize_waveform,
    write_wav,
)
from echoself_ai.voice.consent import VoiceConsentManager, VoiceConsentRecord
from echoself_ai.voice.features import LogMelSpectrogram, create_mel_filterbank

__all__ = [
    "AcousticModel",
    "AcousticModelConfig",
    "AudioConfig",
    "LogMelSpectrogram",
    "VoiceConsentManager",
    "VoiceConsentRecord",
    "create_mel_filterbank",
    "generate_synthetic_tone",
    "load_wav",
    "normalize_waveform",
    "write_wav",
]
