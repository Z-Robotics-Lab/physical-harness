# Operator handbook — for every agent working in this repository

Read `GOAL.md` first. It is the project's north star; every piece of work you
do should move toward it. **Never edit GOAL.md** — all agents align on the same
fixed direction, and only the operator may change it. Current state lives in
the board (query it), architecture in `ARCHITECTURE.md`.

## Hard rules

- **First principles.** Reason from what the system must guarantee, not from
  what similar projects usually do.
- **Simplest thing that works.** Prefer deletion over addition; no speculative
  abstractions; a task-name `if` branch in a generic path means the design is
  wrong — add a card instead.
- **Docs stay true, and there is ONE of them.** When you change behavior,
  update the doc in the same commit; when a doc goes stale, delete or rewrite
  it. A wrong doc is worse than no doc. The public set is exactly
  `docs/project-documentation.md` -- there is no second file and no exempt
  subdirectory (a `docs/design/` carve-out was used to park 110 KB of prose). A second file in `docs/` fails `tests/test_docs_allowlist.py` — the
  test is the gate, not your judgement. Everything else has a home:
  **development docs** (current state, open campaign goals, round plans) go to
  `docs-dev/` — git-ignored, local, never tracked (the test checks that too);
  **retired design capital** (why we forked, scout reports, acceptance
  write-ups, per-increment ledgers) goes to `local-archive/docs/`. Cite those
  by their `docs-dev/…` / `local-archive/…` path so a reader can tell at a
  glance that the reference does not ship.
  Clean `docs-dev/` during each change: delete completed scratch plans and
  superseded logs, merge duplicate reports, and retain one current design and
  verification record. Keep unfinished research and unique experimental evidence.
- **Evidence over demos.** A claim is worth exactly the sealed evidence behind
  it. An honest null and an honest NO-GO are deliverables, not failures.
- **Never game a gate.** No tuning thresholds, swapping gates, or cherry-picking
  seeds to manufacture a promotion. If the semantics of a trigger change, old
  numbers are dead — re-earn them.
- **Audit oracles before trusting them.** Simulator-provided predicates have
  lied to us before (a near-always-true grasp check). Verify a predicate
  discriminates before using it as a success criterion.
- **Rendering is live state, not evidence.** Frames and screenshots never enter
  the session-log chain.

## Your one execution door

`submit_brief(brief, session=...)` — drop a work order into a resident
runtime's inbox. A brief is a **selector plus budgets**, with one task-only data
field: optional natural-language `instruction`. Providers, skills, scene
grounding and oracles are still chosen server-side from manifests; any other
extra key is rejected. `instruction` selects no authority and cannot name a
provider.

```
{"kind":"task", "task":"kitchen_thaw", "seed":420011, "max_replans":3, "max_actuations":40}
{"kind":"task", "task":"basket_smoke_vlm", "seed":424243, "instruction":"把所有物品放进篮子"}
{"kind":"campaign", "campaign":"stack", "dev":[41000,41999], "heldout":[42000,42199]}
{"kind":"rsi", "task":"kitchen_thaw"}
{"kind":"evolve", "task":"kitchen_thaw", "seeds":[420011,420016], "rounds":3, "arm":"auto"}
{"kind":"mission", "mission":"把肉从冰箱拿到微波炉", "seed":41, "arm":"scripted"}
```

`kind:"rsi"` is the generic self-improvement chain (evolution mode only): the
minimal form needs only a task name; the runtime runs allocate → calibrate →
gates → prereg → dev → held-out → install by itself. See
`docs/project-documentation.md` §4.
`kind:"evolve"` accepts optional `llm_model` and `llm_effort` for its next run.
These select a model and declared effort on the installed endpoint; they do not
select providers or change robot bindings. `board.store.rsi_model_options()`
returns the available model IDs and configured efforts without credentials.

`kind:"evolve"` is online program-policy learning inside the offline simulation
laboratory (evolution mode only). Each sampling decision allows at most two batched evidence calls. The LLM samples a
single development seed with `trial`, and may compose edits on a measured working
parent across bounded cycles before `choose` runs the full paired seed suite.
Working branches and policy/seed measurements survive cycles within the same run,
incumbent, evaluator and seed range; exact repeated probes reuse their receipts.
Process restarts retain compact experimental memory, not executable branches.
Historical branches do not force a new cycle into selection before fresh sampling.
Capabilities include editable module names; node-scoped catalogs expose exact source
symbols. Invalid read arguments still cost model calls, but do not consume the
successful-inspection allowance. Bounded literal source pages survive cycles only
while their policy, binding and content remain valid; other evidence is refreshed.
Use `board.store.rsi_command_summary` for bounded command/refusal counts without
reading full experiment audits.
Probes never promote a
policy. Only a strict frozen verification gain without regression updates the
campaign's program overlay. Finite briefs share call, input-byte and probe budgets. Explicit
`continuous:true` renews bounded learning-cycle budgets until cancellation or
a terminal error; cumulative run costs never reset. `rounds>0` caps completed
cycles in this submission, including abstentions; `rounds:0` removes that cap.
Each completed cycle proceeds directly to the next; full trajectories/source stay outside prompts until requested. This does
not train model weights or install skills. The GOAL verification battery remains
required for installation. Logs, before/after trajectories, videos and keyframes
remain available through the board. See `docs/project-documentation.md` §4.0.
Outside proposals use `submit_proposal`, with kind tunables|executor|card|plan;
the same installed-authority validation applies. Cancel and resubmit use the
brief lifecycle. A changed evaluator/source identity starts a new epoch without
reinterpreting historical scores. Autonomous evolve proposals are LLM-only: omit
`proposer` or use `llm`; `rules` is rejected. Endpoint failures seal an error and
stop, never trigger a substitute candidate. A model abstention or exhausted
candidate-repair budget ends only the learning cycle. The scheduler continues within the finite brief budget or opens a new
bounded cycle in explicit continuous mode. A stop reason for the entire run is
recorded separately from the cycle outcome. No trial means
no after measurement. Outside proposals remain an explicit submission interface
for agents/operators; action candidates must name their node.

Intervention and verification boundaries:

- **In evolve, the model selects the intervention node** from observed installed
  actions; first-death attribution is evidence, not a target-selection policy.
  The older preregistered recovery campaign has its own attribution gate.
- **Verification thresholds stay frozen.** The model may propose a trigger threshold
  inside the declared candidate grammar; it cannot change the statistical gates.
- **If an embodiment has no registered recovery primitive, say "nothing to work
  with"** — never improvise an action to fill the gap.

`session` picks the robot: `session-main` (robosuite) / `session-robocasa`
(kitchen; separate interpreter and dependencies). Unsure? Call `sessions()`.

**Submitting does not block.** `submit_brief`/`run_task` hand back a handle;
`brief_status(brief_id, wait_ms=…)` is the ONE call that says where the brief is
(queued/running/**stalled**/done/failed/cancelled, with queue position and how
long the thing ahead has been running) and what it did. Waiting out `wait_ms` is
not an error — it means "still running", so wait again. **`stalled` means nobody
will ever claim it**: that session has no live runtime, `runtime.reason` says
why, and waiting is pointless — say so and stop polling. Never rebuild a brief's
fate by hand from `runtime_events` + `session` + `session_progress`.
`cancel_brief(brief_id)` stops one; it lands at a node boundary, seals as
`runtime.task_cancelled`, and is never counted as a failure.

**When anything looks wrong, call `health()` FIRST** — one dict covering every
session's runtime liveness (asked of `/proc`, never of its own leftover
`runtime_status.json`), mode, heartbeat age, inbox backlog and crash orphans,
plus the console, the model/policy servers and the last `cockpit --restart`
(`restart.state`). Read its `problems` list. Never report
"the runtime is alive" from `runtime_status()` — that file outlives the process
that wrote it, and doing so is exactly how a brief sat queued for 21 hours.

Read results through `runtime_events` / `session_progress` / `store` /
`heldout` / `vault_node` — do not reassemble conclusions from raw files under
`runs/`; sealed artifacts are chain-verified and the board is how you read them.

## The seed ledger is irreversible

- Every seed block burns **once**. Reusing a burned block as a gate or held-out
  poisons the conclusion — the whole result is void.
- The ledger is **derived**: `board.store.burned_blocks(runs/)` is the union of
  gate/heldout blocks of every sealed prereg under `runs/`, plus the burned rows
  of `STATUS.md` (pre-store history: phase 1/2 blocks, held-out rescores). No
  store at all ⇒ gate/held-out allocation is refused, never read as "nothing
  burned". New burns come from sealed prereg; STATUS.md only carries history.
- Calibration blocks are the exception: never a gate, always re-runnable.
- Scratch seeds (< 542000, outside any declared block; 42xxxx/43xxxx by
  convention) never burn the ledger. Seeds ≳ 542479 overflow and crash.
- Held-out is scored **once**, and only when something actually promoted.

## The two-state law

**Execution mode** mounts frozen SkillRecords and writes nothing; **evolution
mode** is the only place experiments happen. Sealed artifacts (prereg,
calibration, held-out results) are immutable — if one is wrong, run a new
round; never edit the old one.

## Known traps

- `python -m pytest`, never `bin/pytest` (its 59 collection errors are all
  spurious). cwd is always the repo root — anywhere that can see
  `sims/robocasa/`, the import silently resolves to a namespace package and
  374 kitchen envs never register.
- `STATUS.md` and `progress.md` are the operator's local, untracked notes —
  never `git add` them. They are display-only: nothing reads STATUS.md to decide
  what is burned (that is `board.store.burned_blocks`), and `rsi_campaign`
  prints a STATUS.md-shaped paragraph for the operator without appending it.

Framework development uses compact board summaries and structured metrics only.
Full RSI experiment logs, videos, trajectories, failure diagnosis and benchmark
sampling belong to the built-in physical agent. Do not inspect that material or
advance benchmarks on its behalf; use concise reported outcomes to improve the
framework, and validate changes with focused regression tests.
