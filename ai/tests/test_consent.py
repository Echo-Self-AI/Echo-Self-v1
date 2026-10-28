from pathlib import Path
import pytest

from echoself_ai.voice.consent import VoiceConsentManager


def test_register_and_check_consent(tmp_path: Path):
    manager = VoiceConsentManager(storage_dir=tmp_path / "consent")
    assert not manager.check_consent("user_123", "spk_1", "synthesis")

    record = manager.register_consent("user_123", "spk_1", scope="synthesis")
    assert record.consent_granted is True
    assert record.revoked is False
    assert len(record.verification_hash) == 64

    assert manager.check_consent("user_123", "spk_1", "synthesis")
    # Scope check: synthesis granted, but cloning not granted
    assert not manager.check_consent("user_123", "spk_1", "cloning")


def test_enforce_gate_success_and_denial(tmp_path: Path):
    manager = VoiceConsentManager(storage_dir=tmp_path / "consent")

    # Gate denied when not registered
    with pytest.raises(PermissionError, match="Voice operation 'synthesis' denied"):
        manager.enforce_gate("user_456", "spk_2", "synthesis")

    # Grant consent
    manager.register_consent("user_456", "spk_2", scope="synthesis")
    # Should not raise
    manager.enforce_gate("user_456", "spk_2", "synthesis")


def test_revocation_blocks_gate(tmp_path: Path):
    manager = VoiceConsentManager(storage_dir=tmp_path / "consent")
    manager.register_consent("user_789", "spk_3", scope="synthesis")
    assert manager.check_consent("user_789", "spk_3", "synthesis")

    manager.revoke_consent("user_789", "spk_3")
    assert not manager.check_consent("user_789", "spk_3", "synthesis")

    with pytest.raises(PermissionError, match="denied"):
        manager.enforce_gate("user_789", "spk_3", "synthesis")


def test_purge_speaker_data(tmp_path: Path):
    storage_dir = tmp_path / "consent"
    audio_dir = tmp_path / "audio_data"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy speaker files
    file1 = audio_dir / "spk_4_01.wav"
    file2 = audio_dir / "spk_4_02.wav"
    other_file = audio_dir / "spk_99_01.wav"
    file1.write_text("audio1")
    file2.write_text("audio2")
    other_file.write_text("keep_this")

    manager = VoiceConsentManager(storage_dir=storage_dir)
    manager.register_consent("user_101", "spk_4", scope="all")

    # Attempting purge before revoking should raise ValueError
    with pytest.raises(ValueError, match="has not been revoked"):
        manager.purge_speaker_data("user_101", "spk_4", audio_dir)

    # Revoke consent and purge
    manager.revoke_consent("user_101", "spk_4")
    removed = manager.purge_speaker_data("user_101", "spk_4", audio_dir)

    assert removed == 2
    assert not file1.exists()
    assert not file2.exists()
    assert other_file.exists()  # Untouched
