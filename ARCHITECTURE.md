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
| Program-policy sampling and independent evaluation | `scripts/evolve.py`, `scripts/evolve_llm.py`, `plugins/rsi/` |
| Public task, lifecycle and evidence interface | `board/` |
| Operator console | The separate `ph-station` repository, through `board` |

## 简体中文

唯一完整说明是[项目文档](docs/project-documentation.md)，方向和安装要求见
[GOAL.md](GOAL.md)。本页只定位代码，避免维护两套会漂移的架构描述。

内置 physical agent 负责仿真采样、日志取证、诊断和程序策略学习。
开发 agent 负责框架、接口、预算、控制台及必要回归测试。
当前开发状态和未完成研究放在不提交的 `docs-dev/`；历史测量仍保留来源。
