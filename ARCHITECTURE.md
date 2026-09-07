# Architecture

The authoritative design is [Project documentation](docs/project-documentation.md).
The project objective and installation requirements are in [GOAL.md](GOAL.md).
This file is a navigation index, not a second architecture specification.

| Responsibility | Code |
| --- | --- |
| Episode orchestration, contracts and evidence chain | `harness/` |
| Mission decomposition and task planning | `scripts/harness_runtime.py`, planner and mission plugins |
| Task graph execution and skill dispatch | `plugins/task/`, `harness/skill_executor.py` |
| Robot, simulator, perception and controller bindings | `plugins/` manifests and providers |
| LLM repair loop over a card copy, paired milestone acceptance | `scripts/evolve.py` |
| Preregistered rule-RSI chain (calibrate → gates → dev → held-out → install) | `scripts/rsi_campaign.py`, `plugins/rsi/` |
| Public task, lifecycle and evidence interface | `board/` |
| Operator console | The separate `ph-station` repository, through `board` |

## 简体中文

唯一完整说明是[项目文档](docs/project-documentation.md)，方向和安装要求见
[GOAL.md](GOAL.md)。本页只定位代码，避免维护两套会漂移的架构描述。

内置 physical agent（evolve 循环里的 LLM）负责读证据、改卡副本、跑种子、判断何时交卷。
开发 agent 负责框架、接口、预算、控制台及必要回归测试。
当前开发状态和未完成研究放在不提交的 `docs-dev/`；历史测量仍保留来源。
