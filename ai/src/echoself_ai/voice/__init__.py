"""Voice AI subsystem for EchoSelf."""
from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import (
    AudioConfig,
    AudioDurationError,
    AudioFileSizeError,
    AudioValidationError,
    CorruptAudioError,
    InvalidAudioFormatError,
    MediaValidationError,
    generate_synthetic_tone,
    ingest_audio,
    load_wav,
    normalize_waveform,
    resample_waveform,
    save_wav,
    write_wav,
)
from echoself_ai.voice.consent import VoiceConsentManager, VoiceConsentRecord
from echoself_ai.voice.dataset import (
    SpeechDataset,
    SpeechItem,
    SpeechVocab,
    collate_speech_batch,
)
from echoself_ai.voice.features import (
    LogMelSpectrogram,
    create_mel_filterbank,
    extract_log_mel_spectrogram,
)
from echoself_ai.voice.vocoder import GriffinLimVocoder, VoiceSynthesizer

__all__ = [
    "AcousticModel",
    "AcousticModelConfig",
    "SpeechDataset",
    "SpeechItem",
    "SpeechVocab",
    "collate_speech_batch",
    "AudioConfig",
    "AudioDurationError",
    "AudioFileSizeError",
    "AudioValidationError",
    "CorruptAudioError",
    "InvalidAudioFormatError",
    "MediaValidationError",
    "GriffinLimVocoder",
    "LogMelSpectrogram",
    "VoiceConsentManager",
    "VoiceConsentRecord",
    "VoiceSynthesizer",
    "create_mel_filterbank",
    "extract_log_mel_spectrogram",
    "generate_synthetic_tone",
    "ingest_audio",
    "load_wav",
    "normalize_waveform",
    "resample_waveform",
    "save_wav",
    "write_wav",
]
