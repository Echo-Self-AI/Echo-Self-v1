"""Unit tests for EchoSelf AI HTTP server layer."""
import json
from pathlib import Path
import urllib.error
import urllib.request
import pytest

from echoself_ai.service.server import start_server_in_thread
from echoself_ai.service.unified import EchoSelfAIService


@pytest.fixture
def test_server(tmp_path: Path):
    service = EchoSelfAIService(
        consent_dir=tmp_path / "consent",
        checkpoints_dir=tmp_path / "checkpoints",
        device="cpu",
    )
    server, thread, base_url = start_server_in_thread(service, host="127.0.0.1", port=0)
    yield base_url
    server.shutdown()
    server.server_close()


def test_http_health_endpoint(test_server: str):
    url = f"{test_server}/health"
    with urllib.request.urlopen(url) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "healthy"
        assert "text" in data["modalities"]
        assert "video" in data["modalities"]


def test_http_supported_jobs_endpoint(test_server: str):
    url = f"{test_server}/supported_jobs"
    with urllib.request.urlopen(url) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "supported_job_types" in data
        assert "llm_generate" in data["supported_job_types"]
        assert "avatar_generate" in data["supported_job_types"]


def test_http_process_success(test_server: str):
    url = f"{test_server}/process"
    payload = {
        "job_id": "http-job-01",
        "job_type": "llm_generate",
        "user_id": "user-http-01",
        "payload": {"prompt": "AI engine online", "max_new_tokens": 4},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["job_id"] == "http-job-01"
        assert data["status"] == "success"
        assert "generated_text" in data["output"]


def test_http_process_unsupported_job(test_server: str):
    url = f"{test_server}/process"
    payload = {
        "job_id": "http-job-fail",
        "job_type": "unknown_modality",
        "user_id": "user-01",
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    assert exc_info.value.code == 422
    err_body = json.loads(exc_info.value.read().decode("utf-8"))
    assert err_body["status"] == "failed"
    assert "Unsupported" in err_body["error"]


def test_http_process_invalid_json(test_server: str):
    url = f"{test_server}/process"
    req = urllib.request.Request(
        url,
        data=b"invalid json {{{",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    assert exc_info.value.code == 400


def test_http_not_found(test_server: str):
    url = f"{test_server}/non_existent_route"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url)

    assert exc_info.value.code == 404
