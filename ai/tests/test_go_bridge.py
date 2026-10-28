"""Integration tests verifying Python AI Service handling of Go ProcessRequest payloads."""
from pathlib import Path
import uuid
import pytest

from echoself_ai.service.unified import EchoSelfAIService, ProcessRequest, ProcessResult


@pytest.fixture
def service(tmp_path: Path):
    return EchoSelfAIService(
        consent_dir=tmp_path / "consent",
        checkpoints_dir=tmp_path / "checkpoints",
        device="cpu",
    )


def test_go_bridge_request_compatibility(service: EchoSelfAIService):
    # Go sends UUID strings and float64 numbers in payload
    go_payload = {
        "job_id": str(uuid.uuid4()),
        "job_type": "llm_generate",
        "user_id": str(uuid.uuid4()),
        "media_id": str(uuid.uuid4()),
        "payload": {
            "prompt": "Hello from Go processor",
            "max_new_tokens": 5.0,  # Go JSON unmarshals ints into float64
            "temperature": 0.7,
            "top_k": 5.0,
        },
    }

    req = ProcessRequest.from_dict(go_payload)
    assert req.job_id == go_payload["job_id"]
    assert req.user_id == go_payload["user_id"]
    assert req.media_id == go_payload["media_id"]

    res = service.process(req)
    assert res.status == "success"
    assert "generated_text" in res.output
    assert "execution_time_ms" in res.metadata
    assert res.metadata["device"] == "cpu"

    # Verify JSON output structure conforms to Go models.JSONMap
    res_dict = res.to_dict()
    assert isinstance(res_dict["output"], dict)
    assert isinstance(res_dict["metadata"], dict)
    assert res_dict["error"] is None
