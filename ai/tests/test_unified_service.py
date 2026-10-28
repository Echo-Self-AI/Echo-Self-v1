"""Unit tests for Unified EchoSelf AI Service facade."""
from pathlib import Path
import pytest

from echoself_ai.service.unified import (
    EchoSelfAIService,
    ProcessRequest,
    ProcessResult,
)


@pytest.fixture
def service(tmp_path: Path):
    consent_dir = tmp_path / "consent"
    checkpoints_dir = tmp_path / "checkpoints"
    return EchoSelfAIService(
        consent_dir=consent_dir,
        checkpoints_dir=checkpoints_dir,
        device="cpu",
    )


def test_process_request_serialization():
    data = {
        "job_id": "job-1234",
        "job_type": "llm_generate",
        "user_id": "user-5678",
        "media_id": "media-9999",
        "payload": {"prompt": "Hello world", "max_new_tokens": 10},
    }
    req = ProcessRequest.from_dict(data)
    assert req.job_id == "job-1234"
    assert req.job_type == "llm_generate"
    assert req.user_id == "user-5678"
    assert req.payload["max_new_tokens"] == 10

    d = req.to_dict()
    assert d["job_id"] == "job-1234"
    assert d["media_id"] == "media-9999"


def test_supported_job_types(service: EchoSelfAIService):
    types = service.supported_job_types()
    assert "llm_generate" in types
    assert "voice_synthesis" in types
    assert "avatar_generate" in types
    assert "video_generate" in types


def test_unsupported_job_type(service: EchoSelfAIService):
    req = ProcessRequest(
        job_id="job-bad",
        job_type="non_existent_job",
        user_id="user-1",
    )
    res = service.process(req)
    assert res.status == "failed"
    assert "Unsupported" in res.error


def test_llm_generate_job(service: EchoSelfAIService):
    req = ProcessRequest(
        job_id="job-llm",
        job_type="llm_generate",
        user_id="user-1",
        payload={"prompt": "EchoSelf is", "max_new_tokens": 5},
    )
    res = service.process(req)
    assert res.status == "success"
    assert "generated_text" in res.output
    assert res.output["prompt"] == "EchoSelf is"
    assert res.metadata["execution_time_ms"] > 0.0


def test_voice_embedding_job(service: EchoSelfAIService):
    req = ProcessRequest(
        job_id="job-voice-emb",
        job_type="voice_embedding",
        user_id="user-1",
        payload={"speaker_id": "spk-01"},
    )
    res = service.process(req)
    assert res.status == "success"
    assert res.output["embedding_dim"] == 64
    assert len(res.output["embedding"]) == 64


def test_avatar_embedding_job(service: EchoSelfAIService):
    req = ProcessRequest(
        job_id="job-avatar-emb",
        job_type="avatar_embedding",
        user_id="user-1",
        payload={"avatar_id": "av-01"},
    )
    res = service.process(req)
    assert res.status == "success"
    assert res.output["embedding_dim"] == 128
    assert len(res.output["embedding"]) == 128


def test_avatar_generation_consent_lifecycle(service: EchoSelfAIService):
    # 1. Unconsented generation fails
    req_gen = ProcessRequest(
        job_id="job-av-gen",
        job_type="avatar_generate",
        user_id="user-avatar-test",
        payload={"avatar_id": "av-consented"},
    )
    res_fail = service.process(req_gen)
    assert res_fail.status == "failed"
    assert "consent not found or revoked" in res_fail.error

    # 2. Register consent
    req_consent = ProcessRequest(
        job_id="job-consent-reg",
        job_type="consent_register",
        user_id="user-avatar-test",
        payload={"modality": "visual", "avatar_id": "av-consented", "scope": "generation"},
    )
    res_reg = service.process(req_consent)
    assert res_reg.status == "success"
    assert "visual_hash" in res_reg.output["records"]

    # 3. Consented generation succeeds
    res_ok = service.process(req_gen)
    assert res_ok.status == "success"
    assert "image_base64" in res_ok.output
    assert res_ok.output["width"] == 64

    # 4. Revocation
    req_revoke = ProcessRequest(
        job_id="job-consent-rev",
        job_type="consent_revoke",
        user_id="user-avatar-test",
        payload={"modality": "visual", "avatar_id": "av-consented"},
    )
    res_rev = service.process(req_revoke)
    assert res_rev.status == "success"

    # Generation now blocked again
    res_blocked = service.process(req_gen)
    assert res_blocked.status == "failed"


def test_video_generate_job(service: EchoSelfAIService):
    req = ProcessRequest(
        job_id="job-vid",
        job_type="video_generate",
        user_id="user-1",
        payload={"prompt": "gentle nodding", "num_frames": 4, "fps": 10},
    )
    res = service.process(req)
    assert res.status == "success"
    assert "video_gif_base64" in res.output
    assert res.output["num_frames"] == 4
    assert res.output["fps"] == 10
