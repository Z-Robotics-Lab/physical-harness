"""The model.endpoint seam: one OpenAI-compatible chat client, local or API.

Four properties: the card's provider satisfies the ModelEndpoint contract
(mount-time isinstance); presets resolve and per-field params override them;
chat() speaks the OpenAI shape end to end against a real (local, stdlib) HTTP
server -- lazy model resolution from GET /models, Bearer auth from the NAMED
env var, opts passed through, reply text extracted; and a dead endpoint is a
graceful degrade -- available() False, plugin_doctor Tier-B SKIP (never red),
exactly the model_qwen precedent.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from harness import contracts
from plugins.model_endpoint import PRESETS, _credential_ref, provider
from scripts.plugin_doctor import check

#: 127.0.0.1:9 -- discard port, nothing listens: connection refused immediately.
_DEAD = "http://127.0.0.1:9/v1"


class _Server(BaseHTTPRequestHandler):
    """A minimal OpenAI-shaped server: GET /models, POST /chat/completions."""

    seen: dict = {}

    def do_GET(self):
        self._reply({"data": [{"id": "served-model"}]})

    def do_POST(self):
        _Server.seen = {
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
        }
        self._reply({"choices": [{"message": {"content": "pong"},
                                  "finish_reason": "length"}]})

    def _reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture()
def endpoint_url():
    server = HTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    thread.join()


def test_provider_satisfies_the_mounted_contract():
    ep = provider(preset="local_sglang")
    assert isinstance(ep, contracts.ModelEndpoint)


def test_presets_resolve_and_field_params_override():
    assert set(PRESETS) == {"local_sglang", "deepseek"}
    assert provider(preset="deepseek").identity == \
        "openai_compat(model=deepseek-chat,base=https://api.deepseek.com/v1)"
    # a per-field param beats the preset (the manifest params escape hatch)
    ep = provider(preset="local_sglang", base_url="http://127.0.0.1:8123/v1",
                  model="qwen")
    assert ep.identity == "openai_compat(model=qwen,base=http://127.0.0.1:8123/v1)"
    with pytest.raises(ValueError):
        provider()  # no preset, no base_url: nothing to point at


def test_chat_speaks_openai_shape_end_to_end(endpoint_url, monkeypatch):
    monkeypatch.setenv("ME_TEST_KEY", "sekrit")
    ep = provider(base_url=endpoint_url, api_key_env="ME_TEST_KEY")
    assert ep.available()
    out = ep.chat([{"role": "user", "content": "ping"}], temperature=0.5,
                  max_tokens=8)
    assert out == "pong"
    seen = _Server.seen
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer sekrit"  # key read from the NAMED env var
    assert seen["body"]["model"] == "served-model"  # lazily resolved, GET /models
    assert seen["body"]["messages"] == [{"role": "user", "content": "ping"}]
    assert seen["body"]["temperature"] == 0.5  # opts pass through untouched
    assert seen["body"]["max_tokens"] == 8
    # the reply's own verdict on why it stopped: the evolve proposer reads "length" as
    # "the answer was cut off", which a completion-token count only ever approximates
    assert ep.last_finish == "length"


def test_named_credential_falls_back_to_dsh_store_without_entering_identity(
        tmp_path, monkeypatch):
    dsh = tmp_path / "dsh"
    dsh.mkdir()
    (dsh / ".credentials.yaml").write_text(
        "version: 1\nrefs:\n  BRIDGE_TEST_KEY: stored-secret\n")
    monkeypatch.setenv("DSH_HOME", str(dsh))
    monkeypatch.delenv("BRIDGE_TEST_KEY", raising=False)
    assert _credential_ref("BRIDGE_TEST_KEY") == "stored-secret"
    ep = provider(base_url="https://example.invalid/v1",
                  api_key_env="BRIDGE_TEST_KEY", model="m")
    assert ep._headers()["Authorization"] == "Bearer stored-secret"
    assert "stored-secret" not in ep.identity


def test_environment_credential_wins_over_dsh_store(tmp_path, monkeypatch):
    dsh = tmp_path / "dsh"
    dsh.mkdir()
    (dsh / ".credentials.yaml").write_text(
        "refs:\n  BRIDGE_TEST_KEY: stored-secret\n")
    monkeypatch.setenv("DSH_HOME", str(dsh))
    monkeypatch.setenv("BRIDGE_TEST_KEY", "environment-secret")
    assert _credential_ref("BRIDGE_TEST_KEY") == "environment-secret"


def test_dead_endpoint_is_unavailable_not_an_error():
    assert provider(base_url=_DEAD).available() is False


def test_doctor_skips_gracefully_when_endpoint_is_down(tmp_path):
    card = tmp_path / "endpoint_down"
    card.mkdir()
    (card / "manifest.toml").write_text(
        '[mounts."model.endpoint"]\n'
        'ref = "plugins.model_endpoint:provider"\n'
        f'params = {{ base_url = "{_DEAD}" }}\n')
    rep = check(card)
    assert rep.green  # a SKIP is never a red
    b = [r for r in rep.results if r.tier == "B" and r.name == "model.endpoint"]
    assert b and b[0].status == "SKIP" and "unreachable" in b[0].detail
