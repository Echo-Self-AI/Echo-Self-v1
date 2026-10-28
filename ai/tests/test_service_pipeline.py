"""Unit tests for MultimodalPipeline end-to-end digital twin interactions."""
from pathlib import Path
import pytest

from echoself_ai.service.pipeline import (
    DigitalTwinResponse,
    MultimodalConsentError,
    MultimodalPipeline,
)
from echoself_ai.service.unified import EchoSelfAIService


@pytest.fixture
def service(tmp_path: Path):
    return EchoSelfAIService(
        consent_dir=tmp_path / "consent",
        checkpoints_dir=tmp_path / "checkpoints",
        device="cpu",
    )


def test_generic_digital_twin_interaction(service: EchoSelfAIService):
    pipeline = MultimodalPipeline(service)

    response = pipeline.execute_interaction(
        user_id="user_guest",
        prompt="Hello world",
        max_new_tokens=4,
        num_video_frames=4,
    )

    assert isinstance(response, DigitalTwinResponse)
    assert response.user_id == "user_guest"
    assert len(response.text_reply) > 0
    assert len(response.audio_base64) > 0
    assert len(response.video_gif_base64) > 0
    assert response.video_num_frames == 4
    assert response.execution_time_ms > 0.0

    d = response.to_dict()
    assert d["user_id"] == "user_guest"
    assert "consent_audit" in d


def test_personal_digital_twin_consent_gating(service: EchoSelfAIService):
    pipeline = MultimodalPipeline(service)

    # 1. Unconsented execution fails
    with pytest.raises(MultimodalConsentError) as exc_info:
        pipeline.execute_interaction(
            user_id="user_twin_01",
            prompt="Hello there",
            avatar_id="avatar_self",
            speaker_id="speaker_self",
        )
    assert "consent not found" in str(exc_info.value)

    # 2. Register consent for both voice and visual
    service.voice_consent.register_consent("user_twin_01", "speaker_self", scope="all")
    service.visual_consent.register_consent("user_twin_01", "avatar_self", scope="all")

    # 3. Consented execution succeeds
    response = pipeline.execute_interaction(
        user_id="user_twin_01",
        prompt="Tell me a story",
        avatar_id="avatar_self",
        speaker_id="speaker_self",
        max_new_tokens=4,
        num_video_frames=4,
    )
    assert response.consent_audit["voice_authorized"] is True
    assert response.consent_audit["visual_authorized"] is True

    # 4. Revoke voice consent -> execution immediately fails
    service.voice_consent.revoke_consent("user_twin_01", "speaker_self")
    with pytest.raises(MultimodalConsentError):
        pipeline.execute_interaction(
            user_id="user_twin_01",
            prompt="Tell me a story",
            avatar_id="avatar_self",
            speaker_id="speaker_self",
        )
