"""Exercise the real streaming code path against a tiny fake OpenAI-compatible server.

This verifies TTFT measurement, usage parsing, the token-estimate fallback,
parameter adaptation and friendly errors without needing Ollama or API keys.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from slmbench.config import ProviderPreset
from slmbench.providers.openai_compat import OpenAICompatProvider
from slmbench.schemas import GenerationParams

FIRST_TOKEN_DELAY = 0.15


def _chunk(model, content=None, finish=None, usage=None):
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model, "choices": []}
    if content is not None or finish:
        body["choices"] = [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish}]
    if usage:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n".encode()


class FakeHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/ps":
            self._json(200, {"models": [{"name": "good", "size": 4_000_000_000, "size_vram": 3_000_000_000}]})
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeHandler.seen.append(body)
        model = body["model"]
        if model == "missing":
            return self._json(404, {"error": {"message": f"model '{model}' not found", "type": "not_found_error"}})
        if model == "reject-temp" and "temperature" in body:
            return self._json(
                400, {"error": {"message": "Unsupported parameter: 'temperature'", "type": "invalid_request_error"}}
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(FIRST_TOKEN_DELAY)
        for piece in ["Hel", "lo", " world"]:
            self.wfile.write(_chunk(model, piece))
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(_chunk(model, finish="stop"))
        if model != "no-usage":
            self.wfile.write(_chunk(model, usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def provider(base_url, provider_id="ollama"):
    preset = ProviderPreset(provider_id, "Ollama", "openai_compat", f"{base_url}/v1", (), local=True, needs_key=False)
    return OpenAICompatProvider(preset)


PARAMS = GenerationParams(temperature=0.0, max_tokens=64, system_prompt="be brief", timeout_s=10)


def test_streams_and_measures(server):
    gen = provider(server).generate("good", "hi", PARAMS)
    assert gen.ok, gen.error
    assert gen.text == "Hello world"
    assert gen.ttft_s >= FIRST_TOKEN_DELAY
    assert gen.latency_s > gen.ttft_s
    assert (gen.input_tokens, gen.output_tokens, gen.tokens_estimated) == (12, 3, False)
    assert gen.finish_reason == "stop"
    sent = FakeHandler.seen[-1]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["messages"][0] == {"role": "system", "content": "be brief"}


def test_estimates_tokens_when_usage_missing(server):
    gen = provider(server).generate("no-usage", "hi", PARAMS)
    assert gen.ok and gen.tokens_estimated and gen.output_tokens > 0


def test_drops_rejected_temperature_and_retries(server):
    gen = provider(server).generate("reject-temp", "hi", PARAMS)
    assert gen.ok, gen.error
    assert "temperature" not in FakeHandler.seen[-1]


def test_reasoning_models_get_larger_budget_and_no_temperature(server):
    provider(server).generate("good", "hi", PARAMS, reasoning=True)
    sent = FakeHandler.seen[-1]
    assert "temperature" not in sent and sent["max_tokens"] >= 4096


def test_model_not_pulled_gives_actionable_error(server):
    gen = provider(server).generate("missing", "hi", PARAMS)
    assert not gen.ok and "ollama pull missing" in gen.error


def test_ollama_not_running_gives_actionable_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # a port nothing is listening on
    p = provider(f"http://127.0.0.1:{port}")
    p.client = p.client.with_options(max_retries=0)
    gen = p.generate("good", "hi", PARAMS)
    assert not gen.ok and "ollama serve" in gen.error


def test_memory_snapshot_reads_api_ps(server):
    mem = provider(server).memory_snapshot("good")
    assert mem == {"memory_gb": 4.0, "vram_gb": 3.0, "gpu_share": 0.75}
