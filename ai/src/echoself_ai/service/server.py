"""Lightweight zero-dependency HTTP server exposing EchoSelf AI Service endpoints."""
import argparse
import json
import logging
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
from typing import Optional

from echoself_ai.service.unified import (
    EchoSelfAIService,
    ProcessRequest,
    ProcessResult,
)

logger = logging.getLogger("echoself_ai.server")


class ThreadedEchoSelfServer(socketserver.ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server allowing concurrent AI request processing."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, service: EchoSelfAIService):
        self.service = service
        super().__init__(server_address, RequestHandlerClass)


class EchoSelfRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for EchoSelf AI engine REST API."""

    server: ThreadedEchoSelfServer

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy standard output logs during unit tests
        pass

    def _send_json_response(self, status_code: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json_response(
                200,
                {
                    "status": "healthy",
                    "service": "EchoSelf AI Engine",
                    "version": "0.1.0",
                    "device": str(self.server.service.device),
                    "modalities": ["text", "voice", "vision", "video"],
                },
            )
        elif self.path == "/supported_jobs":
            self._send_json_response(
                200,
                {
                    "supported_job_types": self.server.service.supported_job_types(),
                },
            )
        else:
            self._send_json_response(
                404,
                {
                    "error": "Not Found",
                    "path": self.path,
                },
            )

    def do_POST(self) -> None:
        if self.path != "/process":
            self._send_json_response(
                404,
                {
                    "error": "Not Found",
                    "path": self.path,
                },
            )
            return

        # Read JSON body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._send_json_response(
                400,
                {"error": "Missing or empty request body"},
            )
            return

        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            req_dict = json.loads(raw_body)
        except Exception as exc:
            self._send_json_response(
                400,
                {"error": f"Invalid JSON payload: {exc}"},
            )
            return

        req = ProcessRequest.from_dict(req_dict)
        result: ProcessResult = self.server.service.process(req)

        status_code = 200 if result.status == "success" else 422
        self._send_json_response(status_code, result.to_dict())


def start_server_in_thread(
    service: EchoSelfAIService,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadedEchoSelfServer, threading.Thread, str]:
    """Start HTTP server on a background daemon thread (port 0 chooses an open ephemeral port)."""
    server = ThreadedEchoSelfServer((host, port), EchoSelfRequestHandler, service=service)
    actual_port = server.server_address[1]
    base_url = f"http://{host}:{actual_port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, base_url


def run_cli() -> None:
    """CLI runner for standalone AI service execution."""
    parser = argparse.ArgumentParser(description="EchoSelf AI Microservice Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", default="cpu", help="Execution device (cpu/cuda)")
    parser.add_argument(
        "--consent-dir",
        type=Path,
        default=Path("data/consent"),
        help="Path to persistent consent directory",
    )
    args = parser.parse_args()

    print(f"Initializing EchoSelf AI Service on device={args.device}...")
    service = EchoSelfAIService(
        consent_dir=args.consent_dir,
        device=args.device,
    )

    server = ThreadedEchoSelfServer(
        (args.host, args.port),
        EchoSelfRequestHandler,
        service=service,
    )
    print(f"EchoSelf AI Service listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down EchoSelf AI Service...")
        server.shutdown()


if __name__ == "__main__":
    run_cli()
