# requirements

The exact dependency manifest for **physical-harness**. The [README](README.md)
has the short version; this file is what you follow when it isn't enough.

The base install pulls **only `numpy`** — the harness boots and
passes its base test lane on a machine with no simulator at all. Every simulator
is an optional card, and the two GPU sim stacks (robosuite, robocasa) live in
**separate venvs** because their `numpy`/`mujoco`/`robosuite` pins are mutually
incompatible (numpy 1.x vs 2.x is an ABI-level split).

## System requirements

| Requirement | Value | Why |
|---|---|---|
| OS | Linux (x86-64) | tested on Ubuntu, kernel 7.0.0 |
| Python | `>=3.11,<3.13` | base + robosuite venv; robocasa venv uses 3.12 |
| GPU + EGL | NVIDIA GPU, `libEGL_nvidia` present | headless MuJoCo rendering (`MUJOCO_GL=egl`) |
| Disk | ~30 GB free per sim venv | robocasa assets alone are **23 GB decompressed** |
| Network / API keys | none | all numbers come from local sim rollouts, no external API |

Base install and its test lane need **no GPU, no network, no API key**.

## 1. Base install (harness only, no simulator)

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHONPATH=. .venv/bin/python -m pytest -m "not robosuite"   # base fast lane, sim-absent
```

Dependency groups from [`pyproject.toml`](pyproject.toml):

| Group | Packages | For |
|---|---|---|
| base (always) | `numpy>=1.26` | kernel |
| `[dev]` | `pytest>=8`, `pytest-timeout>=2`, `ruff==0.16.4`, `mcp==2.0.0` | tests + lint; `mcp` because the both-faces tests import `board/mcp_server` |
| `[cockpit]` | `mcp==2.0.0` | console MCP seam (`board/mcp_server.py`) only. Pinned: dsh is an RC, so the wire stays fixed |
| `[policy_remote]` | `websockets>=15`, `msgpack>=1.0` | the `plugins/policy_vla_remote` websocket transport (StarVLA/openpi policy servers); its protocol tests skip when absent. `>=15` is a hard floor: the sync client gained `ping_interval`/`ping_timeout` in 15.0 and the vendored client passes both |

Run the base lane isolated (a fresh process where `robosuite` is genuinely
unimportable) — never as a subset of a full green run. Snapshot format and the
two isolation methods are in
[docs/project-documentation.md](docs/project-documentation.md) §3. The docs carry
the lane command and never the counts — `tests/test_docs_counts_e2e.py` fails on a
hand-synced `NNN passed` in `README.md`, `README.zh.md`,
`docs/project-documentation.md` or `CLAUDE.md`.

## 2. robosuite sim card (same venv, optional extra)

The Lift/Stack robosuite stack. Installs into the harness venv as an extra —
adding it does not change the base lane (the card stays behind the `robosuite`
marker):

```bash
uv pip install -e ".[dev,embodiment_robosuite]"
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m pytest       # full lane
```

| Package | Pin | Why the hard pin |
|---|---|---|
| `mujoco` | `==3.3.7` | mujoco `>=3.4` renamed `qM` → `M`; robosuite 1.5.2 calls `mj_fullM(..., data.qM)` and crashes |
| `robosuite` | `==1.5.2` | release build, paired with the mujoco pin above |

## 3. robocasa sim card (SEPARATE venv)

RoboCasa is the long-horizon kitchen sim. It pins `robosuite` **master** (not
the 1.5.2 release), `mujoco==3.3.1`, and `numpy==2.2.5` — all incompatible with
the harness venv, so it **must** be its own venv. Layout (sibling to the repo):

```
sims/
├── robocasa/          # git clone, main @ a07e365 (robocasa v1.0.1)
├── robosuite/         # git clone, master @ 5ce6643 (calls itself 1.5.2, is master)
└── robocasa-venv/     # python 3.12 venv, isolated from the harness .venv
```

| Package | Pin | Source |
|---|---|---|
| `robocasa` | `1.0.1` @ `a07e365` | editable clone, `pip install -e` |
| `robosuite` | master @ `5ce6643` | editable clone; **needs the compat flag below** |
| `mujoco` | `==3.3.1` | robocasa hard pin |
| `numpy` | `==2.2.5` | robocasa hard pin (2.x — the ABI reason for the split) |
| `torch` | `2.7.1` | pulled by `lerobot==0.3.3`; venv is ~8 GB |

Install into the robocasa venv (`pip install -e $REPO` also installs the
harness kernel so `import harness` works inside this venv):

```bash
cd sims/robocasa && pip install -e .
cd ../robosuite  && pip install -e . --config-settings editable_mode=compat --no-deps
pip install -e $PHYSICAL_HARNESS_REPO      # harness kernel, base deps only
```

### 3a. The PEP-660 editable trap (do not skip)

A default editable robosuite install produces `robosuite.__file__ == None`, and
robosuite's own `pathlib.Path(robosuite.__file__)` then raises `TypeError` on
import. Reinstall robosuite with `--config-settings editable_mode=compat
--no-deps` (shown above). robocasa itself is fine.

### 3b. The namespace-shadow cwd trap

If the current directory can *see* `sims/robocasa/` (a repo root whose name ==
the package name), `import robocasa` binds the empty namespace package and **374
kitchen envs silently fail to register**. Rules:

- **Install/asset scripts** (`setup_macros`, `download_kitchen_assets`): run from
  inside `sims/robocasa/` (the repo root).
- **The robocasa runtime**: run with `cwd = physical-harness repo` (which has no
  `robocasa/` directory), `PYTHONPATH=$REPO`, `MUJOCO_GL=egl`.

If `setup_macros` won't run, `cp robocasa/macros.py robocasa/macros_private.py`
is equivalent.

### 3c. Asset download (23 GB)

Six zips from box.com, ~25 min, "~10 GB" compressed but **23 GB decompressed**
into `sims/robocasa/robocasa/models/assets/`. The script prompts `input()` on
existing files, so pipe `yes` for headless runs:

```bash
cd sims/robocasa
yes y | python -m robocasa.scripts.download_kitchen_assets
```

### 3d. robocasa test lane

Only run `-m robocasa` inside the robocasa venv (its robosuite is master; never
run the robosuite lane here):

```bash
cd $PHYSICAL_HARNESS_REPO      # NOT inside sims/robocasa — see the shadow trap
MUJOCO_GL=egl $ROBOCASA_VENV/bin/python -m pytest -m robocasa
```

## 4. LIBERO sim card (THIRD venv, scaffold)

LIBERO's 2022-era pins (`robosuite==1.4.0`, `mujoco==2.3.2`, `numpy==1.22.4` —
no py3.11+ wheels) can share neither interpreter, so it is `sims/libero-venv`
(**py3.10**) beside a `sims/LIBERO` clone (master @ `8f1084e`; assets/bddl/init
states are bundled in-repo, ~405 MB, no separate download). Full install
recipe, the pin deviations, and the three traps (empty PEP-660 editable →
`.pth` workaround; the machine-global `~/.libero/config.yaml` singleton →
`LIBERO_CONFIG_PATH`; the unpinned-mujoco 3.x resolution) live in
[docs/project-documentation.md](docs/project-documentation.md) §5.6. Test lane: `-m libero`,
only inside that venv.

## 5. UI companion — ph-station

The operator console is a separate repo,
[ph-station](https://github.com/Z-Robotics-Lab/ph-station) (a dsh web fork). It
needs **Node 22** (or ≥24) and **pnpm 11.7.0** via corepack — see its README.
The harness's `scripts/cockpit` builds and serves it; you do not launch it
standalone.
