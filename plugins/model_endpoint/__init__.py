"""Model-endpoint card: the ONE client behind every model call (plan §1b).

There is no "DeepSeek vs local Qwen" choice to wire: both are OpenAI-compatible
``/chat/completions`` servers, so the seam is a single
``harness.contracts.ModelEndpoint`` config ``{base_url, api_key_env, model}``
and a preset is just a filled-in config. The card owns the HTTP client (stdlib
urllib, the ``governor.proposer.qwen38_transport`` stance); every model-driven
seat (a VLM planner, a model proposer, ph-station's agent) consumes the mounted
contract and nothing else imports an HTTP library.

Distinct from ``plugins/model_qwen``: that card fills the ``reasoner.proposer``
seam (evidence brief -> proposal, via LlmProposer's schema-gated parse); this
card is the raw chat transport underneath such seats. The api key is named by
ENV VAR, never by value, so the manifest params -- which enter the plan sha --
carry configuration identity without leaking a secret into the hash chain.

``available()`` probes GET /models (one short request); a consumer and
``plugin_doctor`` degrade to a graceful SKIP when no endpoint is up -- the
model_qwen precedent, verbatim.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: preset name -> the {base_url, api_key_env, model} triple. A preset is config,
#: not code: our machines point at the local sglang serving (Qwen3.8 on the
#: 4090; model=None resolves lazily from GET /models because launch_qwen38.sh
#: publishes the model *path* as the id -- qwen38_transport's lesson), GPU-less
#: users flip the manifest params to "deepseek" and export DEEPSEEK_API_KEY.
PRESETS: dict[str, dict[str, str | None]] = {
    "local_sglang": {"base_url": "http://127.0.0.1:30000/v1",
                     "api_key_env": "QWEN38_API_KEY", "model": None},
    "deepseek": {"base_url": "https://api.deepseek.com/v1",
                 "api_key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
}


def _credential_ref(name: str | None) -> str | None:
    """Resolve a named secret without ever making it configuration identity.

    Environment wins.  The fallback is the credential store already owned by
    the 3080 console, whose deliberately small on-disk shape is::

        refs:
          DEEPSEEK_API_KEY: <value>

    This is a narrow bridge, not a YAML implementation: only a direct scalar
    under ``refs`` is admitted.  It never logs or returns the surrounding file,
    and callers put the result only in an Authorization header.
    """
    if not name:
        return None
    value = os.environ.get(name)
    if value:
        return value
    home = Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))
    path = home / ".credentials.yaml"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    in_refs = False
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent == 0:
            in_refs = stripped == "refs:"
            continue
        if not in_refs or indent < 2 or ":" not in stripped:
            continue
        key, scalar = stripped.split(":", 1)
        if key.strip() != name:
            continue
        scalar = scalar.strip()
        if len(scalar) >= 2 and scalar[0] == scalar[-1] and scalar[0] in "\"'":
            scalar = scalar[1:-1]
        return scalar or None
    return None


class OpenAICompatEndpoint:
    """``harness.contracts.ModelEndpoint`` over any /v1 chat-completions server."""

    def __init__(self, *, preset: str | None = None, base_url: str | None = None,
                 api_key_env: str | None = None, model: str | None = None,
                 timeout: float = 60.0, images: bool | None = None) -> None:
        cfg: dict[str, str | None] = dict(PRESETS[preset]) if preset else {}
        if base_url is not None:
            cfg["base_url"] = base_url
        if api_key_env is not None:
            cfg["api_key_env"] = api_key_env
        if model is not None:
            cfg["model"] = model
        if not cfg.get("base_url"):
            raise ValueError("model endpoint needs a preset or an explicit base_url")
        self._base = str(cfg["base_url"]).rstrip("/")
        self._key_env = cfg.get("api_key_env")
        self._model = cfg.get("model")
        self._timeout = timeout
        # ponytail: images inferred from the model name; set images=true|false in params to override
        self.images = (bool(images) if images is not None
                       else any(t in (self._model or "").lower() for t in ("vision", "vl")))
        self.last_usage: dict | None = None   # {prompt, completion} tokens of the last chat()

    @property
    def identity(self) -> str:
        """The endpoint identity a consumer stamps into content hashes: which
        model, at which base_url (the model_qwen precedent -- identity is
        content, never an env var smuggled past the sha). ``model=None`` means
        not yet lazily resolved; consumers hashing an identity should chat (or
        set ``model``) first."""
        return f"openai_compat(model={self._model},base={self._base})"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = _credential_ref(self._key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _get_json(self, url: str, timeout: float) -> Any:
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    def available(self, timeout: float = 3.0) -> bool:
        """One GET /models decides whether the endpoint answers at all. An auth
        rejection (hosted API, key env unset) is an OSError too -- honestly
        unavailable, since chat() would fail the same way."""
        try:
            self._get_json(f"{self._base}/models", timeout)
            return True
        except OSError:
            return False

    def chat(self, messages: Sequence[Mapping], **opts: Any) -> str:
        """POST /chat/completions, OpenAI shape; ``opts`` pass through to the
        body untouched (temperature, max_tokens, seed, response_format, ...) --
        decode discipline belongs to the consumer, this is a transport. A message
        ``content`` may be a list of OpenAI content parts (text / image_url data
        URLs) when ``images`` is on; ``last_usage`` holds the reply's token counts."""
        if self._model is None:
            self._model = self._get_json(
                f"{self._base}/models", self._timeout)["data"][0]["id"]
        body = json.dumps({"model": self._model, "messages": list(messages),
                           **opts}).encode()
        req = urllib.request.Request(f"{self._base}/chat/completions",
                                     data=body, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            reply = json.load(resp)
        u = reply.get("usage") or {}
        self.last_usage = ({"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens")}
                           if u else None)
        return reply["choices"][0]["message"]["content"]


def provider(**params: Any) -> OpenAICompatEndpoint:
    return OpenAICompatEndpoint(**params)


#: path -> next reply index for a JSON-list fake file. Module-level so every
#: FakeEndpoint instance in the process (one per planner) consumes ONE sequence.
_FAKE_CURSOR: dict[str, int] = {}


class FakeEndpoint:
    """Test-only ``ModelEndpoint``: chat() returns a canned reply read from
    ``path`` (default: env ``PH_MODEL_ENDPOINT_FAKE``). A file holding a JSON
    LIST is a sequence of replies (str or object) consumed in order across every
    instance on that path; anything else is the one reply, every call. Reached
    by the same registry-ref seam (``plugins.model_endpoint:fake_provider``) so a
    GPU-less e2e drives the real planner_vlm prompt path with fixed graphs."""

    def __init__(self, *, path: str | None = None, images: bool = False, **_: Any) -> None:
        self._path = path or os.environ.get("PH_MODEL_ENDPOINT_FAKE")
        self.images, self.last_usage = bool(images), None
        if not self._path:
            raise ValueError("fake endpoint needs path= or PH_MODEL_ENDPOINT_FAKE")

    @property
    def identity(self) -> str:
        return f"fake({self._path})"

    def available(self, timeout: float = 0.0) -> bool:
        return Path(self._path).is_file()

    def chat(self, messages: Sequence[Mapping], **opts: Any) -> str:
        text = Path(self._path).read_text()
        try:
            seq = json.loads(text)
        except ValueError:
            return text
        if not isinstance(seq, list):
            return text
        i = _FAKE_CURSOR.get(self._path, 0)
        if i >= len(seq):
            raise ValueError(f"fake endpoint {self._path}: canned replies exhausted")
        _FAKE_CURSOR[self._path] = i + 1
        return seq[i] if isinstance(seq[i], str) else json.dumps(seq[i])


def fake_provider(**params: Any) -> FakeEndpoint:
    return FakeEndpoint(**params)
