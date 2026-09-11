"""Base/plugin test split (R3, W6 双层测试) + the live-runtime stand-in.

The `robosuite` marker gates every test that drives the embodiment_robosuite
card's mujoco rollout. When robosuite is unimportable (the extra is not
installed), those items auto-skip -- so `pytest -m "not robosuite"` is the base
fast lane on a card-absent machine, and `pytest -m robosuite` self-skips there
instead of erroring. With the card present the marker is inert and every test
runs, so full-suite parity is untouched.
"""
import json
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "robosuite: needs the embodiment_robosuite card (robosuite+mujoco); "
        "auto-skipped when it is unimportable",
    )
    config.addinivalue_line(
        "markers",
        "robocasa: needs the robocasa venv (robocasa+robosuite-master); "
        "auto-skipped when robocasa is unimportable (harness .venv)",
    )
    config.addinivalue_line(
        "markers",
        "vla: starts the real pi0.5 server (openpi venv, GPU, port 8000) for the test",
    )
    config.addinivalue_line(
        "markers",
        "libero: needs the libero venv (LIBERO+robosuite-1.4); "
        "auto-skipped when libero is unimportable (harness .venv)",
    )
    config.addinivalue_line(
        "markers",
        "mshab: needs the mshab venv (mani_skill+sapien, the ManiSkill-HAB "
        "checkout); auto-skipped when mani_skill is unimportable (harness .venv)",
    )


def _auto_skip(items, pkg, marker, reason):
    if find_spec(pkg) is not None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if marker in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def live_runtime(tmp_path):
    """Make a session dir look SERVED by a real resident runtime.

    ``board.store.runtime_liveness`` reads ``/proc/<pid>/cmdline`` rather than
    trusting ``runtime_status.json`` (a file outlives its writer -- the whole
    2026-08-27/28/29 incident family), so a test that wants a live session must
    supply a live process with a runtime's argv: ``harness_runtime.py
    --session-dir <that session>``. Spawning one is cheaper than mocking the
    /proc seam and it exercises the guard for real -- test_submit_advisory also
    reads the interpreter back off argv[0] the same way.

    Yields ``serve(session_dir) -> pid``; every process is reaped at teardown.
    """
    procs = []
    fake = tmp_path / "harness_runtime.py"
    fake.write_text("import time; time.sleep(120)\n")

    def serve(session_dir) -> int:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, str(fake), "--session-dir", str(session_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(proc)
        # Wait for exec: right after spawn /proc/<pid>/cmdline is still the
        # pytest argv, and runtime_liveness reads it back -- bounded poll.
        for _ in range(500):
            try:
                if b"harness_runtime.py" in Path(f"/proc/{proc.pid}/cmdline").read_bytes():
                    break
            except OSError:
                pass
            time.sleep(0.01)
        else:
            raise RuntimeError(f"pid {proc.pid} never exec'd the fake runtime")
        (session_dir / "runtime_status.json").write_text(
            json.dumps({"pid": proc.pid, "mode": "execution"}))
        return proc.pid

    yield serve
    for proc in procs:
        proc.kill()
        proc.wait()


def pytest_collection_modifyitems(config, items):
    _auto_skip(
        items, "robosuite", "robosuite",
        "robosuite unimportable (embodiment_robosuite extra not installed)",
    )
    _auto_skip(
        items, "robocasa", "robocasa",
        "robocasa unimportable (robocasa venv only)",
    )
    _auto_skip(
        items, "libero", "libero",
        "libero unimportable (libero venv only)",
    )
    _auto_skip(
        items, "mani_skill", "mshab",
        "mani_skill unimportable (mshab venv only)",
    )
