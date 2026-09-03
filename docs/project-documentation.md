# Project Documentation

物理测试台（physical-harness）的**唯一**官方文档。它回答：东西在哪、谁说了算、
怎么跑起来、门禁是什么意思、怎么加一个仿真器、怎么接你自己的模型、学习策略挂在
哪条缝。

机制细节见 [ARCHITECTURE.md](../ARCHITECTURE.md)，项目方向见
[GOAL.md](../GOAL.md)，给 agent 的操作手册见 [CLAUDE.md](../CLAUDE.md)，安装与依赖
清单见 [README.md](../README.md) 与 [requirements.md](../requirements.md)。
`docs/` 下只有本文一份。没有第二份，也没有豁免的子目录。

| 章 | 内容 | 谁读 |
|---|---|---|
| §1 | 仓库分工与决策归属 | 任何新来的人 |
| §2 | 起服务、控制台配置、backbone LLM | 要把系统跑起来的人 |
| §3 | 底座测试快照与隔离复现 | 改底座的人 |
| §4 | 通用 RSI 机制与六条门禁 | 投 rsi brief 的人 |
| §5 | 加一个仿真器（venv、卡片、坑） | 加本体的人 |
| §6 | 接入你自己的 VLM planner / VLA policy / 恢复原语 | 集成模型的人 |
| §7 | 学习策略挂在哪条缝（fast/slow brain） | 做 VLA 接缝的人 |
| §8 | 静默失败面与 `health()` | 系统看起来不对的时候 |
| §9 | Skill graph protocol：谓词、技能记录、执行图、轨迹、账本 | 改 planner/validator/技能库的人 |
| §10 | 没在这份文档里的东西 | 找历史的人 |

---

## 1. 仓库分工与决策归属

### 1.1 一句话定位

| 仓库 | 语言 | 是什么 |
|---|---|---|
| physical-harness | Python | 机器本身。所有能力、所有判断、所有证据 |
| ph-station | TypeScript | 控制台。显示和输入，没有任何判断 |

分界红线只有一条：**TypeScript 里不许有业务逻辑、不许有统计、不许有 gate
决策。** 面板上显示的每一个数字，都是 Python 算好后逐字渲染的。

### 1.2 physical-harness 里有什么

```
physical-harness/
├── harness/          kernel —— 不认识任何机器人、仿真器、任务
│     contracts.py      所有插槽的 Protocol（mount 时校验形状）
│     kernel.py         episode loop、privilege budget 计量
│     events.py         SessionLog：哈希链，篡改即断
│     opstream.py       活状态事件流（永不进链）
│     manifest.py       discover()：折叠所有 card 的 manifest
│     config.py         MountPlan.sha —— 配置即实验身份
│
├── plugins/          card —— 所有"具体"的东西
│     embodiment_robosuite/    机械臂 + robosuite
│     embodiment_robocasa/     厨房机器人（独立 venv）
│     embodiment_libero/       LIBERO（脚手架）
│     mission_*/               任务图 + planner（纯数据 manifest）
│     candidates/<name>/       候选卡：不进 base fold，evolve 经 PH_PLUGINS_EXTRA 按提案挂载
│     task/                    通用任务机器：workload、validate
│     rsi/                     进化引擎：gate、配对检验、campaign
│     policies/                策略驱动
│     planner_vlm/             VLM 生成 node graph
│     policy_vla_remote/       VLA 策略走 websocket
│     model_endpoint/          OpenAI 兼容 chat 客户端
│
├── board/            对 runs/ 的唯一 API，三个 face 字节等价
│     store.py          实现
│     storecli.py       CLI face
│     mcp_server.py     MCP face（agent 用的就是这个）
│
├── scripts/          常驻进程和入口
│     harness_runtime.py   常驻 runtime，盯 inbox
│     cockpit             一键启动一切（含拉起 ph-station）
│     rsi_campaign.py      RSI 七步链
│     frame_dump.py        画面、keyframe 和最新 rollout MP4
│     plugin_doctor.py     card 体检
│
├── profiles/dsh/     ← 注意：控制台的配置在这里，不在 ph-station
│     cordis.patch.template.yml  MCP server 注册 + LLM route + 默认 preset
│     deploy_profile.py   把模板渲染到 $DSH_HOME/cordis.patch.yml
│
└── runs/             证据（gitignored）
```

**关键**：控制台的配置在 physical-harness 里，由 harness 仓库拥有——换模型、注册
MCP server 都改 `profiles/dsh/cordis.patch.template.yml`（模板里没有任何绝对家目录，
路径全从仓库根推导）。细节见 §2.2。

### 1.3 ph-station 里有什么

它是 deepseek-harness（dsh）的 fork。

```
ph-station/
├── packages/core/        agent loop、session、system-prompt 注册表
├── packages/llm/         provider 适配器
│     llm-pi-ai/            通用 OpenAI 兼容 route ← 本地/API 都走它
│     llm-deepseek/         DeepSeek 专用
│
├── packages/host/
│     dsh-ph-board/        ← 唯一连接 physical-harness 的包
│                            每个 @Remote 方法转发给 storecli
│                            自动暴露 POST /api/board/<name>
│
├── packages/client/      面板（全是渲染）
│     ui-ph-livegraph/      执行图谱 + 过程流 + 取景窗/视频下载
│     ui-ph-panels/         RSI 总览 + 迭代记录 + 能力卡 + 账本 + Run RSI 按钮
│     ui-ph-ops/            运行体征侧栏：主机资源、本地模型开关
│                            + `skills` 页（`skills(session)` 记录表，展开看 by_executor 证据）
│                            + `evolve` 页（campaign 列表/轮次表/rsiSeries 折线/rsiFrames 路径；Start=submitBrief `{kind:evolve}`，Stop=cancelBrief）
│     ui-ph-vault/          技能库
│     ui-ph-battle/         Held-out 战报
│     ui-ph-dash/           实验台（面板布局）
│     ui-conversation/      对话
│
└── apps/web/             浏览器入口
```

ph-station 从不直接读 `runs/`。它只会调 dsh-ph-board，后者 exec storecli。

### 1.4 两者怎么连（三个接触点）

```
① 启动
   physical-harness/scripts/cockpit
     → 拉起 harness_runtime（Python，一个 session 一个进程）
     → 拉起 node .../ph-station/apps/cli/lib/bin.js web --port 3080

② agent 调工具（写路径）
   ph-station agent loop
     → stdio JSON-RPC
     → physical-harness/board/mcp_server.py
     → 原子写进 runs/<session>/inbox/

③ 面板读数据（读路径）
   面板 → POST /api/board/<name>
        → dsh-ph-board（TS）
        → exec storecli（Python）
        → board/store.py 计算
        → JSON 逐字返回，TS 只负责画
```

读路径的三层每一层都是纯透传：`board/storecli.py` 把 `board.store` 的返回值
`json.dumps` 到 stdout（名字寻址的 fn 先过 `board.store.safe_child` 这道唯一审计过
的穿越守卫）；TS 侧 `execFile`（不走 shell）+ 固定的方法名白名单 = 没有注入面；
gateway 在 `trusted-host` 栅栏后自动暴露 `POST /api/board/<fn>`。三个 face
（MCP tool / CLI / 面板）调的是同一个函数，并有逐字节等价测试钉住。

**面板只有三条写路径**：`submit_brief`（原子落 runtime 校验过的 inbox）、`submit_skill_plan`（规划面板的“执行”按钮：服务端重新核验一张已校验、全部叶子有 binding 的 skill plan，再投一张**普通** task brief，走同一条原子落盘，见 §6.1.2）和 `cancel_brief`（落取消标记，runtime 在下一个轮边界处理）；桥（dsh-ph-board）白名单里除此之外全是只读。规划本身（`plan_skill_task`）是读：不执行、不落盘。**没有认证层**：`trusted-host` 防的是 DNS rebinding，不是身份；服务绑`127.0.0.1`，`/api/board/*` 只读封存的 `runs/`。

**天花板（已标注）**：每个面板请求一个 Python 子进程，冷导入
`board.store → harness.events.SessionLog`。人类节奏的轮询下够用；真测出慢了再把
bridge 升级成常驻读进程，面板和 CLI face 都不用改。

### 1.5 谁拥有什么决策

| 功能 | 归属 | 为什么 |
|---|---|---|
| 任务图怎么拆 | physical-harness（planner） | 是能力，不是显示 |
| 统计检验、gate 判定 | physical-harness（plugins/rsi） | TypeScript 里不许有统计 |
| seed 账本、burn 检查 | physical-harness（runtime） | 决定结论有效性 |
| 什么算成功 | physical-harness（predicate/oracle） | 是证据 |
| brief 校验 | physical-harness（runtime，唯一权威） | MCP 层故意不校验 |
| 用哪个模型 | physical-harness（profiles/dsh/） | 配置属于主板 |
| agent 怎么理解你的话 | ph-station（agent loop + 模型） | 是交互 |
| 画面怎么显示 | ph-station（面板） | 是显示 |
| 面板布局、语言、主题 | ph-station | 是显示 |
| 红/黄/绿阈值着色 | ph-station（纯展示） | 数据仍来自 Python |

判断归属的问法：**这件事错了，会不会让一个科学结论变错？** 会 →
physical-harness。不会 → ph-station。

---

## 2. 起服务、控制台配置、backbone LLM

安装命令（base venv、robosuite extra、robocasa/libero 独立 venv）在
[README.md](../README.md) 与 [requirements.md](../requirements.md)；本章讲装完之后
的事。

### 2.1 一条入口：`scripts/cockpit`

操作员只用 UI，所以 cockpit 负责拉起**一切**：

```bash
scripts/cockpit          # 常驻 runtime ×3 + ph-station UI @ :3080，全部留活
scripts/cockpit --status # 只打印健康，什么都不启动；exit 1 = 有问题（见 §8）
scripts/cockpit --stop   # 只停本次调用启动的进程（按 pidfile 里的精确 PID）
```

- **`--restart [--build]`**（控制台「重启」/「重建并重启」两个按钮 → MCP `restart_services(build)`
  → `board.store.restart_services`）：先 `setsid` 脱离再返回（按按钮的那个控制台马上就要被
  停掉），脱离的副本记住 pi0.5 是否 SERVING → `--stop` →（`--build` 时在 `$PH_STATION` 跑
  `pnpm build`，失败即中止、什么都不起）→ 正常启动；`--with-policy` 只在之前 SERVING 时补回
  （pi0.5 由操作员手动起，绝不默认）。进度写 `runs/restart.log`，`health().restart` 读它
  （`state: idle|running|failed|done` + 最后一行），控制台回来后面板照此显示。

它起三个常驻 runtime：`runs/session-main`（.venv，robosuite，`--frames`）、
`runs/session-robocasa`（robocasa venv，无头 egl）、`runs/session-robocasa-rsi`
（robocasa venv，`--mode evolution --frames`，默认开——进化态 brief 只能投这里）。

- **领养或派生**：先扫 `ps` 找该 session 目录上活着的 runtime，找到就**领养**
  （打印 PID，不重启，不记进 `--stop` 名单），没有才 `nohup` 派生并记 PID。
  一个 session 目录绝不并存两个 runtime——强制的那一半在
  `harness_runtime.py` 自己身上（boot 时对 `runs/<session>/runtime.lock` 取排他
  `flock`，第二个实例拒绝启动并报出持锁 pid；`kill -9` 会释放）。
- `--render` 只在 `$DISPLAY` 存在时才传（runtime 在无头下硬拒绝 `--render`）；
  无头派生带 `MUJOCO_GL=egl`。opt-out：`--no-runtime` / `--no-render`。
- `--stop` 永不按模式 kill（那会打中操作员自己的 shell）；被领养的 runtime 不动。
- **只重启控制台而不打断实验**：`kill` 那个精确的 `web_pid`，再跑一次 cockpit——它会重新
  领养活着的 runtime（打印 `adopting resident runtime … not restarting`）。**归属是粘的**：
  pidfile 里已记为 `runtime_adopted=0` 的 pid 被再次领养后仍算 cockpit 派生，`--stop`
  照样杀；只有 cockpit 从没派生过的 runtime 才留着不动（此前每次只重启控制台都会把
  全部 runtime 变成「领养」，`--stop` 从此永久空转）。

### 2.2 控制台配置：模板是提交的，跑的是渲染出来的那份

`scripts/cockpit` 每次启动都用 `profiles/dsh/deploy_profile.py` 把提交在仓库里的
`profiles/dsh/cordis.patch.template.yml` 渲染成 `$DSH_HOME/cordis.patch.yml`
（幂等、原子、零第三方依赖）。**渲染失败就拒绝服务**：没有那个文件，agent 就没有
`mcp__physical-harness__*` 工具，会静默退回原生 bash——那正是 MCP server 存在要防的
无治理路径。

- 渲染出来的每一条路径都由仓库根推导，所以 clone 到哪都能用。
- 可变项——backbone `base_url` / model id / 显示名 / `apiKeyEnv`、控制台端口、可选
  trusted host——来自仓库根 git-ignored 的 `.env`（`.env.example` 已提交并逐项注释）。
- `.env` 同时是控制台自己的凭据层：dsh 解析 key 的优先级是 进程环境 >
  `$DSH_HOME/.credentials.yaml` > `<cwd>/.env` > `$DSH_HOME/.env`，而 cockpit 在
  `exec node … web` 前会 `cd` 到仓库根，所以 `<cwd>` 是确定的。
- `llm-pi-ai` 被基础 bundle **挂成休眠态**（零 route），正是为了让部署自己声明，所以
  那一行是裸 `- id:` **override**（不是 `insert:`）。`local-qwen` 是手写 route
  （`api` + `baseURL` + 显式 `models` 列表——pi-ai 不带它的目录，也没有任何东西去问
  端点），声明 `input: [text, image]`、`compat.maxTokensField: max_tokens` 和
  `supportsDeveloperRole: false`（pi-ai 认不出私有 baseURL，否则会按 OpenAI 自己来
  寻址）。
- **占位凭据不是可选项。** 本地服务在 loopback 上不鉴权，但手写 route 没有目录，
  `llm-pi-ai` 的 `provider.ts:132` 因此总是给它声明 apiKey 鉴权——"无 key"这件事拼不
  出来。省掉 `apiKeyEnv` 会让每个请求在最前面就失败：
  `PI_AI_ERROR: No API key for provider: local-qwen`。route 里写的是环境变量**名**
  （`PH_MODEL_KEY_ENV`，默认 `LOCAL_QWEN_API_KEY`），值在 `.env` 或
  `$DSH_HOME/.credentials.yaml`，永不进提交的文件。服务端忽略这个 header。

两件宿主状态**优先级高于**这份 patch，所以部署只**报告**、绝不写：

- `$DSH_HOME/settings.yaml` 的 `agent-default-model:` 盖过渲染出来的模型行——已有的
  dsh 安装会保留它自己的默认模型，直到那里也改掉。settings 是热重载的，cordis 行需要
  重启控制台。
- 同一个文件里的 `reasoningEffort: high` 与 `physical` preset 的一步派发相冲。它是
  操作员状态，这里不改。

**对着运行时验证，不是对着文件**：

```
curl -s -X POST http://127.0.0.1:3080/api/llm.models -H 'Content-Type: application/json' \
  -d '{"type":"client-request","rpcId":"...","method":"llm.models","payload":{}}'
```

`local-qwen` 必须出现在 `groups` 里且 `failures: []`。

### 2.3 backbone LLM：一个模型通吃对话、规划、看图

控制台的 agent 跑在**本地**模型服务上，不是 DeepSeek API。当前 backbone 是
llama.cpp 的 `llama-server`，端口 **30001**，由 `~/models/launch_llamacpp.sh` 启动；
`board.store` 的 `model_server` face（UI 侧栏的本地模型开关、`storecli model_server`）
只认这一个 launcher，调用方只能给 `status|start|stop` 三个动作词，永远不给路径或命令
行——这个 face 从浏览器可达，调用方给的脚本就等于在 harness 机器上远程执行代码。
进程身份靠 `/proc/<pid>/exe` 的 basename 是 `llama-server` **且** 命令行里有
`--port 30001` 两半共同证明（只看 argv 的话，任何提到这个二进制的 shell 都会匹配）。

serving `unsloth/Qwen3.8-27B-GGUF` 的 `UD-Q4_K_XL`（17.56 GB）+ `mmproj-BF16.gguf`
（931 MB），Vulkan 构建，`-ngl 999 -c 24576 --jinja`。实测对比（同一张 24 GB 4090D）：

| | sglang AWQ | llama.cpp Q4_K_XL |
|---|---|---|
| 一次 vision 请求后的常驻显存 | 21.8 GB | **19.3 GB** |
| 结构化 `tool_calls` | 有 | 有（**必须 `--jinja`**） |
| robosuite 画面上的视觉 | 准 | 准 |
| 生成 | ~60 tok/s | 24 tok/s |
| prompt | CUDA 快 | 307 tok/s |
| 更小的档位 | 无 | IQ4_XS 13.3 GB · Q3_K_XL 12.2 GB |

第一次请求报 ~7 tok/s 的 prompt eval，那是 Vulkan 在编译 graph，不是稳态——量第二次。
生成速度的差距大半来自 Vulkan 构建，CUDA 构建应能补回大部分，速度真咬人时那是下一步。

**为什么离开 sglang**：它结构性地压不到 ~21 GB 以下——`embed_tokens` 在里面根本没有
量化路径（`compressed_tensors.py` 实现了 `ParallelLMHead`，对
`VocabParallelEmbedding` 什么都没有），所以 2.37 GiB 权重是永久地板；Hub 上每个 4-bit
checkpoint 又都把 `lm_head` 留在 `ignore` 列表里。GGUF 两个都量化，而 sglang 的
`GGUF_HF_NAME_MAP_BUILDERS` 不列 `qwen3_5`，所以 GGUF 只能由 llama.cpp 提供服务——
这在控制台侧零成本，因为 `llama-server` 是 OpenAI 兼容的，换回去只是把 `baseURL`
改回 `:30000` 再起 sglang。

**如果你回到 sglang 部署**（参考脚本 `~/models/launch_qwen38.sh`），两条硬教训：

```
--served-model-name qwen3.8-27b        稳定 id；真名是一个文件系统路径
--tool-call-parser  qwen3_coder        ← 没有它，任何 agent 都不工作
--reasoning-parser  qwen3              把 <think>…</think> 切成 reasoning_content
--context-length 32768 --max-total-tokens 32768 --max-prefill-tokens 32768
--mem-fraction-static 0.92 --disable-piecewise-cuda-graph
```

- **是 `qwen3_coder`，不是 `qwen`/`qwen25`。** checkpoint 自带的
  `chat_template.jinja:68` 让模型用 XML 方言回答
  `<tool_call><function=NAME><parameter=K>v</parameter></function></tool_call>`，
  不是 `qwen` 系 parser 期待的 hermes 风格 JSON。parser 选错（或没选），sglang 会把那段
  XML 当普通 `content` 返回、没有 `tool_calls` 字段，于是每个 agent 都静默地什么都不
  调——读起来像模型笨，其实是配错了。验证方式是发一个带 `tools:` 的请求看
  `finish_reason: "tool_calls"`，**永远不要靠读它的散文判断**。
- **Qwen3.5 是 mamba/线性注意力混合模型，这就是全部的显存故事。** mamba 状态缓存和
  注意力 KV 池共用同一份 `--mem-fraction-static` 预算，所以抬 context 是在**饿死**状态
  缓存：`--max-total-tokens 32768` 在 `0.86` 下会在启动时死于
  `max_mamba_cache_size=4, mamba_ratio=5, resulting max_num_reqs=0`。解法是更大的
  static fraction，买单的是 `--disable-piecewise-cuda-graph`（分段 prefill graph 在这
  种 prompt 长度下花 1.15 GB 换很少收益）。
- 24 GB 4090D 上实测：权重 17.71 GB，注意力 KV ~64 KiB/token，
  `max_total_num_tokens=26089`，空载剩 ~1.0 GB。**sglang 会静默把
  `--max-running-requests` 从 2 夹到 1**（状态缓存只供得起一个并发请求），所以并发调用
  是排队而不是失败。MTP/NEXTN 投机解码在装着的 sglang 里不可用，所以抬 context 没有
  牺牲解码速度——本来就没有可牺牲的。
- **26089 才是真天花板，不是标称的 32768。** 超过池子的请求即使还在标称窗口内也会失败，
  所以告诉 harness 的是 24576。
- 两个看着有希望、实测不行的旋钮：`--mamba-ssm-dtype bfloat16` 省 0.36 GB 但**把模型
  弄坏**（同一个 tool prompt 从 35 个 reasoning token 变成 600 个且没有答案，死循环）；
  `--enable-memory-saver` 确实工作（`/release_memory_occupation` 真的在 0.03 s 里交回
  20 GB，`/resume_memory_occupation` 再拿回来），但它要求 `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments` 没被设置，而 fp32 状态 + 24576 token 下剩余预算会让 mamba 缓存在
  启动时失败。

### 2.4 一张 24 GB 卡放不下 backbone 和一次标定

模型在 `0.92` 下只剩 ~0.6 GB，够**一个** sim 任务（一次 `stack` 在服务开着时跑完了，
23.9/24.5 GB），离一次 RSI 标定差得远——它的十个池子 worker 每个都要一个 EGL context。
下表每一行都是一次真实的启动尝试：

| `--mem-fraction-static` / context | 结果 |
|---|---|
| 0.92 / 32768 | 起得来，22.5 GB，剩 ~0.6 GB |
| 0.88 / 24576 + `--kv-cache-dtype fp8_e5m2` | 起得来，21.2 GB，剩 ~2.8 GB |
| 0.88 / 8192 | 起得来，21.0 GB，但 8 K 装不下 agent 自己的 prompt（带 board 工具约 22 K） |
| 0.855 / 8192，关 cuda graph | **失败**：`max_mamba_cache_size=2, mamba_ratio=5 → max_num_reqs=0` |
| 0.81 / 16384 | **失败**：权重之后只剩 0.22 GB；状态缓存**每个请求**要 146.81 MB |

**没有任何配置能压到 ~21 GB 以下。** 权重常驻 ~19.7 GB（17.71 GB 张量加激活和 graph），
混合状态缓存要五个 146.81 MB 的槽才肯服务一个请求，所以算术地板就在 20 GB 上面一点，
旋钮怎么拧都一样。

两个杠杆在这里不适用：`--cpu-offload-gb`（拿 64 GB 主机内存换显存）**在加载时就让这个
模型崩**（`Expected all tensors to be on the same device, but found at least two devices,
cuda:0 and cpu` —— AWQ 权重加混合层撑不住这种切分）；投机解码（`--speculative-*`、
MTP/NEXTN、DFlash2）买的是延迟不是显存，草稿模型**增加**常驻权重。真正付账的是
`--kv-cache-dtype fp8_e5m2`：它把注意力池砍半，这才是低 fraction 下把 24576 token 买
回来的原因。

**所以一张 24 GB 卡不能同时托住 backbone 和一次 RSI 标定。** 要么标定前停掉模型服务，
要么把 backbone 换成更小的多模态 checkpoint（8-14 B AWQ，~8-10 GB）——route 是配置，
换模型就是 §2.2 那份模板里的一处编辑。
---

## 3. 底座测试快照（base gate）

### 3.1 什么是底座快道，怎么隔离跑

**底座快道**是 `pytest -m "not robosuite"`，**隔离**运行——一个全新进程，机器上
robosuite 真的不可导入（没装 `embodiment_robosuite` extra，或 import 被挡）。
**绝不能**拿一次全绿全量跑的子集来充数：全量跑会导入那张卡，而收集顺序污染只在卡真的
缺席时才保持修复。

合并门禁：任何动底座的改动**前后各取一次**这个快照，任何字段回退都挡合并。评测
battery 是 RSI 打分的门禁，**不是**底座车道。

快照格式：

```
pass       : <N> passed
skips      : <M> skipped, each with its reason
wall time  : <T>s
AST green  : test_boundaries + test_kernel green (harness-imports-nothing +
             profiles-declarative)
```

任何 `importlib.util.find_spec("robosuite") is None` 的进程都会触发
`tests/conftest.py` 的收集钩子自动跳过 `robosuite` 标记项。两条路：

- **不装 extra 的 venv**：`pip install -e .[dev]`（只有底座依赖），然后
  `pytest -m "not robosuite"`。
- **挡掉 import**（在全新进程里）：把一个
  `import sys; sys.modules["robosuite"] = sys.modules["mujoco"] = None`
  的 `sitecustomize.py` 放进 `PYTHONPATH`，再 `pytest -m "not robosuite"`。

日常在 harness `.venv` 里跑的是两个标记都排除的那条：
`PYTHONPATH=. .venv/bin/python -m pytest -m "not robosuite and not robocasa"`。
它比隔离快照多 3 个 pass（那 3 个 camera-env 跳过项在卡在场时变成通过）。

### 3.2 隔离底座车道：命令与跳过清单

这一节**不写数字**。计数每加一个测试就飘，手抄三处必然落后（一天内飘过两次）；
不变量只有一条：**仿真卡片零安装时，底座车道全绿**。取数用命令，不抄结果：

```
PYTHONPATH=. .venv/bin/python -m pytest -o addopts="" -q -m "not robosuite and not robocasa"
```

隔离复现（robosuite 被挡）按 §3.1 的两条路之一。隔离跑里**预期的跳过**（文件与原因，
不是数量）：

```
test_grasp_geometric.py      camera env unavailable
test_grasp_geometry.py       camera env unavailable
test_reducers.py             cloned weights not present
test_plugin_doctor.py        robocasa unimportable (robocasa venv only)
test_robocasa_card.py        robocasa unimportable (robocasa venv only)
test_robocasa_drivers.py     robocasa unimportable (robocasa venv only)
test_robocasa_marker.py      robocasa unimportable (robocasa venv only)
test_robocasa_missions.py    robocasa unimportable (robocasa venv only)
test_runtime_frame.py        robocasa unimportable (robocasa venv only)
test_libero_marker.py        libero unimportable (libero venv only)
test_rsi_workload.py         runs/campaign-pj-scripted not present
AST green                    test_boundaries + test_kernel
deselected                   robosuite-marked items
```

（policy_remote extra 已安装时，其 2 个 live-socket 测试参与运行。）

**全量对照（卡在场）**：同一条命令在 harness `.venv` 里跑，robosuite 在场。区别只在于
robocasa 标记项没有 robocasa 可导入而跳过，只在 `sims/robocasa-venv` 里经
`pytest -m robocasa` 跑；libero 标记项同理只在 `sims/libero-venv` 里跑；camera-env 跳过项
在卡在场时变成通过。跳过清单之外出现新的跳过或任何失败，才是需要看的信号。

### 3.3 fresh clone 的合法差异

上面的快照定义在**带着封存 `runs/` 证据**的 checkout 上（`runs/` 不在 git 里）。全新
clone 合法地显示**更多跳过，绝不是失败**：

- +2 `test_plugin_doctor.py` 的 verify-claim 测试跳过（封存 store 不在）
- +2 其他读封存 rescore/campaign 产物的测试跳过
- +1 `test_runtime_frame.py` 的 JPEG 写测试、+3 `test_keyframes.py` 的抓帧测试在没有
  Pillow 时跳过（Pillow 搭的是 sim extras 的车，不在底座依赖里；`dump()` 本身会降级成
  无帧，keyframe listener 同理）
- README 里那两条上手命令照写就能用：`dev` extra 带齐了收集所需的一切（包括双 face
  测试要的 `mcp`）

全新 clone 上出现 **FAILURE**（不是跳过）就是真回归。

## 4. 通用 RSI 机制

一句话：`{"kind":"rsi","task":"<任务名>"}` 投进进化态 runtime，整条纪律链自己走完。
不再为每个任务手写一个 campaign 脚本。

实现：`scripts/rsi_campaign.py`（链本体）+ `scripts/harness_runtime.py`（brief 面）。
通用路径里**没有任何任务名 if 分支**——任务是参数，加任务仍然是装一张卡。

### 4.0 轻量 evolve 循环

`kind:"rsi"` 是重链（下面 4.1–4.5）。旁边还有一条**轻量**循环，不做候选分类、不做晋级状态机、不加门禁，harness 不训练：

```
{"kind":"evolve","task":"kitchen_thaw","seeds":[lo,hi],"rounds":N,"arm":"auto"}
```

只在进化态被接受（与 campaign/rsi 同一道拒绝）；runtime `_run_evolve` 起 `scripts/evolve.py` 子进程（走 `_run_watched`，取消/killpg 复用）。`rounds` 是**总目标**，`seeds`/`arm` 续投时可省（取 campaign.json 里的）。

一轮：
1. **看**：同种子 suite（与 task brief 同一条 `_mount_plan → workload.run` 路径）→ 每种子首死节点、fault `{kind,node,msg}`、每节点 executor。
2. **试**（内置 proposer，按序取第一个可行的）：① 首死节点换 executor —— 绑定 policy 的 record `evidence.by_executor` 成功率高于实测者优先，否则 record 上任一本轮之前没试过的其它绑定 executor（无证据也诚实试一次）；② 该节点驱动 mount 参数一维 ±20% 扰动（`[tunables]` 表或顶层数值；键按轮次轮转、符号交替，经 `PH_MOUNT_PARAMS_OVERRIDE` 并进 `manifest.mount_params`）；③ 都没有 → `kind:"none"` 并写明原因，`detail.needs` 列出能解锁的项（`tunables on <ref>` / `evidence for another executor` / `proposal`）。
3. **同种子再跑**，试验已应用（executor 经 `scripts.evolve:planner_provider` 盖进 `node.executor`）。
4. **成功种子数变好才发布**：带实测 `by_executor` 行（tunables 还有新值）的 record 走 `InMemorySkillGraph.publish` —— 与 `publish_plans` 同一道进化态专用门；否则不发布。试验抛异常 → 记 `tried.detail.error`，after=before，不崩。

每轮封存一行 `rsi_step {brief,task,round,tried,before,after,best,published,suite_sha,per_seed,needs}`（`per_seed` = 保留 suite 的 `[{seed,success,first_death,failure_mode}]`，`needs` 只在 `tried.kind:none` 时非空）（按 (task,round) 幂等，2 s 轮询时实时封存，退出时兜底），并 tmp+rename 写 `runs/<session>/campaigns/evolve-<task>/campaign.json`：
`{task,session,seeds,arm,rounds:[{round,tried:{kind,node,detail},before,after,best,suite_sha,published,per_seed,needs,media:[路径],media_dropped:{"种子/节点":原因},ts}],best,cursor,status:running|cancelled|done,applied:{executors,tunables}}`。
`applied` 是已接受的状态，后续每轮重新应用；下一轮的 before 直接沿用上一轮保留的结果，不重测。

**进度（live）**：campaign.json 还带一个 `live` 块，每到阶段/种子/节点边界就随整个文件 tmp+rename 重写（单写者，无竞争）：`{phase: idle|baseline|propose|retest|publish|done|cancelled, round, seeds_total, seed_index, seed, node, started_at, round_started_at, phase_started_at, last_round_s, per_seed_partial:[本次 suite 已跑完的 {seed,success,first_death,failure_mode}], tried, message:"第 1 轮 基线评测：种子 4244 运行中 (nav-can1)，1/2"}`。`node` 是推断：`task.plan` 落地取首节点，每个成功的 `task.verify` 落地推进到计划序的下一节点（没有 node-start 行）。`rsi_run` 原样返回为 `live`（旧文件为 null）；它是活状态，`rsi_step` 只封轮行，永不封 live。
**取景窗**：runtime `--frames`（egl 自动开）时给 evolve 子进程传 `PH_RSI_FRAMES=<session>/frame.jpg`，与 rsi 链同一把 `_maybe_arm_frames` 锁，suite 每集镜像到 `read_runtime_frame` 读的那份文件；帧永不进链。

**停/续**：`cancel_brief` 落标记 → evolve.py 在轮边界退出（状态 `cancelled`，exit 3；轮中则 killpg）→ `runtime.task_cancelled`，brief 进 `cancelled/`。同 task 再投 evolve → 从 `cursor` 继续；已 `done` 且 rounds 不变 → 空操作。

**媒体规则**（`harness/media.py`）：段级节点每 4 步录一帧 128px 到内存，来源按序取第一个存在的：`embodiment.frame(obs)`（robocasa：obs 里已有的 `robot0_agentview_left_image`，不需要渲染器，所有驱动免费得到）→ `driver.frame()` → `env.frame()`；verify 成功才落 `media/<task>/<seed>/<node>.mp4`（无 imageio 则 .gif），失败即丢；>1 MB 降 fps/抽帧重编；同节点重跑覆盖。录制失败不影响任务但绝不静默：该节点 `diagnostics.media` 封 `{kept:true,file}` 或 `{kept:false,reason:verify_failed|no_frame_source|no_frames|encode_failed[,error]}`，同一原因写进 `index.json["dropped"]`（evolve 轮行 `media_dropped`）。帧永不进链，链和 campaign.json 只存路径。

**三面** `skills(session)`（逐字节等价）：records 概览，每技能一行 `{name, kind, bindings: {emb: [executor 键]}, evidence: {emb: {n, k, by_executor}}, limits, failure_modes, source}`——库记录被会话 `skills/` 下发布的同名副本覆盖（`source: session`）。ph-station 桥（dsh-ph-board）白名单同步加 `skills` / `rsiRun` / `rsiSeries` / `rsiFrames`，写路径只有 `submitBrief` / `cancelBrief`。

**提案与候选卡**：`runs/<session>/proposals/<id>.json` 是 evolve 的收件箱，条目 `{task, kind: tunables|executor|card, payload, note}`（三面 `submit_proposal` 校验形状后原子落盘（store/CLI 收 JSON 字符串，MCP 面收 `proposal: dict`）、`proposals(session)` 列出，`applied` 为 null 表示待处理）。evolve 每轮开头取该 task 最旧的待处理条目，就地盖 `applied:{round,ts}`，封存 `rsi_proposal_applied {brief,task,round,id,kind,note}`，并把它当作本轮的「试」——取代内置 proposer，发布规则不变（同种子成功数变好才写回 record）。payload：
- `tunables`：`{ref, path:[...], to, node?}`（与内置 ② 同一条 `PH_MOUNT_PARAMS_OVERRIDE` 路径）；
- `executor`：`{to, node?}`（record 里已绑定的 executor 键）；
- `card`：`{path: plugins/candidates/<name>, to: <executor 键>, ref: "module:attr", params?, node?}`——该轮 suite 把候选目录追加进 `PH_PLUGINS_EXTRA`（`discover` 接受单卡目录），绑定只注入内存里的 records/segment_specs；变好才把 `bindings[emb].policies[<键>] = {ref, params, transport}` 写进发布的 record。
`node` 缺省为最常见首死节点；缺字段或节点未跑过 → `tried.kind:"none"` 并写明原因（提案照样盖 `applied`）。campaign.json 每轮多一项 `proposal: {id,kind,note}|null`，`applied` 多一项 `cards`。
候选卡 `plugins/candidates/<name>/` 与普通卡同一 manifest 形状，不被 base fold 扫到；`[executors.<键>] skill=, embodiment=, ref=, transport?` 由 `discover` 折进 `Registry.executors`，`skill_library.bind_executors` 在加载时把它盖到 `bindings.<本体>.policies.<键>`（只在挂载时可见，record 文件不动）。首张代码候选 `grasp_geometric_robocasa`：executor 键 `geometric`，code-as-policy `hover→descend→close→lift`，自带 `[tunables]`，provider 参数 `{tunables:{...}}` 覆盖（`mount_params` 只扫 `plugins/*/`，不扫 PH_PLUGINS_EXTRA）；`KitchenThawDriver` 对有 `bind(env, target=)` 的 executor 走原生路径（raw obs 进，12 维 env action 出）。`scripts/plugin_doctor.py plugins/candidates/<name>` 可直接体检（今天报 claim-only SKIP）。
提案人：ph-station 的 `skill-author` preset 只读 `rsi_run/rsi_series/rsi_frames` 与链，唯一写口是 `submit_proposal`。

**三面**（store / storecli / mcp 逐字节等价，只读 campaign.json）：`rsi_run(task, session)` = campaign.json + `latest`；`rsi_series(task, session)` = 每轮 `{round,before,after,best,per_seed,needs}`；`rsi_frames(task, round, session)` = 那一轮的 `media` 路径列表。没有 campaign → `None` / `[]`。

### 4.1 brief 形状

```
{"kind":"rsi","task":"kitchen_thaw"}
```

最小形态就这一行。可选键**只用来覆盖本来由测量决定的东西**：

| 键 | 默认 | 覆盖了什么 |
|---|---|---|
| `node` | 由归因选 | 目标治理节点（覆盖会记进裁决书） |
| `cal` / `dev` / `heldout` | 自动领 | `[lo,hi]`，钉住某个块而不是领新的 |
| `workers` | 10 | 池子宽度 |
| `floor` | 0 | 领块的起始下限 |

其他键一律被 `_BRIEF_KEYS` 拒掉（和 task/campaign 同一道闸）。
`kind:"rsi"` 和 `kind:"campaign"` 一样**只在进化态被接受**。

### 4.2 链上七步

| 步 | 做什么 | 在哪 |
|---|---|---|
| a. 领种子 | 从**派生账本**（`board.store.burned_blocks(runs/)`：`runs/` 下所有已封存 prereg 的 gate/heldout 区间，并上 STATUS.md 已烧行的历史）之外领**一整块 650**，切成标定 150 / dev 300 / held-out 200；钉住的 dev/heldout 撞上已烧区间同样拒绝。没有任何 store ⇒ 拒绝领 gate/heldout，绝不当作「没烧过」；标定块不过闸（标定永不设门、永远可复测），可用 `cal` 钉住旧块复测 | `rsi_campaign.allocate` + `harness_runtime._rsi_blocks` / `_assert_unburned` |
| b. 标定 | **通用探针**：把 `{"kind":"task"}` 那条路在池子里跑 N 次，skills root 指向空目录 → 臂天然是 baseline。产出链基率、**逐节点 × 机制**首死、每集耗时。任务的节点图/kind/after 边由 planner 现问，不是硬编码表 | `rsi_campaign.calibrate` / `_probe_one`，brief 装配复用 `harness_runtime.task_brief`（和活跑逐字节同一张 brief） |
| c. 门禁 | 六条机械判据逐条打分。**没过就停在这里**，裁决书写清缺哪条能力 + 触发它的那个数，一粒 dev 种子不烧 | `rsi_campaign.gate` |
| d. prereg | content-hash 封存，**在任何 dev 种子跑之前** | `rsi_campaign.build_prereg` + `plugins.rsi.workload.run` 盖 provider 三元组 |
| e. dev campaign | 调既有 `run_campaign`，FROM-SCRATCH（`parent_store=None`）。门 = 配对同种子 McNemar（对父）+ blind twin + `min_fixed`，功效缩放取前缀 | `plugins/rsi/campaign.py` |
| f. held-out | 仅当有晋级，**只评一次** | 同上 |
| g. 折入 | 发布记录复制进该 session 的 skills root；两态铁律照旧（执行态 skills-root 变更触审计 → 归档旧 log + 全新 boot 封 row0） | `harness_runtime._run_rsi` → `_copy_skills` |
| h. 账本 | 生成一段 STATUS.md 形状的条目**打印给操作员**，并进 `runtime.rsi_scheduled` 链行。**从不自动 append**——STATUS.md 只是操作员的展示用笔记，真正的账本由 d 步封存的 prereg 派生（a 步），没有第二份可写的真相 | `rsi_campaign.ledger_entry` |

每集还有一道墙钟帽（`scripts/rsi_campaign.py:EPISODE_WALL_S`，池子 worker 里用
SIGALRM）：被砍掉的一集返回诚实的 `first_death="wall_timeout"` 行，`attribute()` 把它
算作未治理——不 charge 给任何人，永远不会成为目标。它存在是因为 8 个 worker 曾在病态
标定场景上各挂 >1h，把链饿死在 138/153。

**每任务一个 campaign 脚本的做法已退役**（`scripts/probe_*.py`、`stack_campaign.py`、
`grasp_cube_campaign.py` 都被通用路径覆盖，保留只为复现历史证据）。仍然保留自己脚本的
是三个形状不同的：`clear_build_campaign.py` 和 `place_campaign.py` 是从父 bundle 续跑
（`parent_store` 种起，通用路径今天只做 FROM-SCRATCH），`acceptance_campaign.py` 要
`--claim <卡目录>` 读该卡的 `[claim]` 表，不是"给任务名就够"的形状。

### 4.3 六条门禁判据

| id | 触发条件 | 结论 |
|---|---|---|
| `c1_base_degenerate` | 基率 0% 或 100% | 无残余可学，停 |
| `c2_base_ceiling` | 基率 ≥ 0.90 | 诚实 null，不烧 dev/held-out |
| `c3_budget_exhaust_dominant` | 多数失败死于 `max_actuations`/horizon 耗尽 | 调配置，不是 RSI |
| `c4_attribution` | 未治理节点死数 ≥ 可治理节点死数 | 归因 pivot，先要那个节点的能力 |
| `c5_recovery_primitive` | 目标节点所属本体没注册恢复原语 | RSI 无从下手 |
| `c6_wall_clock` | 估计标定 + 一代 dev > 2h | 今夜只标定 |

### 4.4 诚实边界

**1. 恢复原语不存在就明说。** 恢复原语由**本体卡**在自己的 manifest.toml 里声明
（`[recoveries.<name>] ref = "module:attr"`，见 §6.3），由
`harness.manifest.discover()` 像 mounts/campaigns 一样折叠（重名 loud fail）；
`plugins/rsi/repertoire.py` 只读折叠结果，并按 `harness.contracts.RecoveryStrategy`
Protocol 做 isinstance 校验。理由：`RecoveryActor` 把 phase 名翻成动作靠的是
`harness/spec_tabletop.py` 的 `PHASE_HEIGHT`——tabletop 手臂的 above/descend/close/lift
词汇，换本体即无意义，所以修复形状写在说这套词汇的那张卡里。没有声明的卡，链就原样报
「该本体（卡 X）无注册恢复原语，RSI 无从下手」，并把 robosuite 侧的
`servo_descend`/`servo_probe` 指出来当**模板**，**不现编一条**。

> 这条注册是必需的，因为更便宜的那个检查是**错的**：
> `plugins/embodiment_robocasa/kitchen_driver.py` 把 `retarget` / `on_handback`
> 定义成有文档的 **no-op**，所以 `hasattr` 探针会把厨房驱动报成「可治理」，
> 而真触发时那条规则悄无声息地什么都没做。**方法在场 ≠ 原语在场。**
> 驱动协议检查保留为第二道必要条件，不再是唯一条件。

`blockers` 是列表，不是第一个就返回——只被告知其中一条的操作员会去修错的东西。

**2. 目标节点不由 agent 挑。** 由 `attribute()` 从首死数据选：verify 节点的死沿
`after` 边回 charge 给它验的那个执行节点（它本来就没有自己的治理面），perceive/decide
的死谁也不 charge（那是 c4 的 pivot 信号），然后在可治理节点里取 argmax。
`node` 键能覆盖，但覆盖会写进裁决书。

**3. 阈值不由 agent 挑。** `critic_budget=0` 让 `plugins/rsi/stats/search.py` 结构性地
够不到特权特征——「优先非特权」是预算，不是偏好。特权规则只能靠调高预算进来，而
`run_campaign` 在每次晋级都跑转移消融，所以特权收益必定带着它的塌陷曲线一起出现。
恢复形状同理：由目标节点**实测**的主导失败 stage 决定（stage 名落在 place 词汇里 →
place 形修复），且只能在该本体已注册的 repertoire 里取。

**4. 诚实 NO-GO / 诚实 null 是合格产出。** 裁决书带 `proceed` + 逐条判据 + 触发它的
那个数；账本条目把「未烧」写明。链停在门禁时 store 里有裁决、没有 skills，这是**完成
态**，不是失败态。

### 4.5 已验证

`--task stack` 与 `--task inventory_build` 均在 scratch 种子（43xxxx，<542k、不在任何
STATUS 声明块内）上跑通：

* `inventory_build` 标定 20 席 → 链基率 45.0%，首死 `grasp-cube 7 / build-stack 4 / none 9`
  → 归因自动选中 **`grasp-cube`**，即操作员当初手工挑的那个节点。
* `stack` 全链 a–h：标定 30 席（基率 53.3%）→ 门禁 GO → prereg 封存
  `dbe21b8b4081` → dev gen-1 功效前缀 100 席 → **0 晋级 = 诚实 null** → held-out
  431300-431329 **未烧**。
---

## 5. 加一个仿真器

原则：**不为任何一个 sim hardcode harness**——每个 sim 是一张 embodiment 卡 + 一个
解释器，底座零 sim 知识。落地形态是 venv-per-sim + 卡片 per-sim + 常驻 runtime per-sim。

### 5.1 为什么每个 sim 一个 venv

* 三套引擎不可共 venv：robocasa 钉 robosuite master（自称 1.5.2，含 pip 1.5.2 没有的
  `load_model_on_init`）+ mujoco 3.3.1 + numpy 2.2.5；harness `.venv` 钉 robosuite
  1.5.2 release + mujoco 3.3.7 + numpy 1.26.4。**numpy 1.x/2.x ABI 是隔离的最大理由。**
* 底座依赖是松的（`numpy>=1.26` + zstandard），base lane 在 robosuite 不可导入的机器
  上全绿——**底座本来就是 sim-agnostic 的**，这是本设计的全部凭据。
* RoboCasa venv：`sims/robocasa-venv`（py3.12，robocasa 1.0.1@a07e365 +
  robosuite master@5ce6643 editable-compat + 23G 资产），EGL 无头冒烟通过，同 seed
  双 rollout 逐元素一致（确定性成立），`get_ep_meta()/set_ep_meta()` 可做场景指纹。
* **sys.path 遮蔽陷阱**：cwd 能看见 `sims/robocasa/`（repo 根目录名 == 包名）时
  `import robocasa` 命中 namespace package，374 个 kitchen env 静默不注册。
  规矩：robocasa runtime 一律 cwd=physical-harness repo（那里没有 robocasa 目录）。
* 隔 websocket 的那一类（RoboTwin(SAPIEN) / RoboDojo(Isaac 5.1)）走 XPolicyLab 契约 =
  每 policy 一个目录，`model.py` 四方法（`__init__/update_obs/get_action/reset`）+
  `deploy.yml` server 配置。policy 与仿真器异环境运行是官方设计——两个 benchmark 只需
  一次契约实现。RoboDojo 前置风险：驱动 570/580 + CUDA 12.8，license 非商业限定。

### 5.2 架构 —— 三条既有轴各自延长，零新概念

```
sims/robocasa-venv  ──解释器──▶  常驻 runtime #2 (runs/session-robocasa, MUJOCO_GL=egl,
                                  cwd=$REPO, PYTHONPATH=$REPO)
plugins/embodiment_robocasa/  ── 卡片：env provider + percept provider + PREDICATES 原语
board submit_brief(session=…) ── 路由：写哪个 session 的 inbox（默认 session-main 不变）
```

* **runtime 就是"看一个 inbox 的进程"**（`harness_runtime.py`：claim = 原子 rename）。
  多 sim = 多常驻 runtime，各自 session 目录、各自解释器，互不知晓。cockpit 的
  adopt-or-spawn 逐 session 应用；`--stop` 精确 pidfile 不变。
* **board 面有一个 `session` 参数**（默认 `session-main`）：storecli / board fn /
  mcp tool 三脸同改（双脸铁律）。
* **路由错了照样投得进去**：任务名在 manifest 联合表里是一张表，所以 kitchen_thaw
  投给 session-main 会被接受，几秒后在**另一个进程**的日志里 mount 失败。MCP 的
  `submit_brief`/`run_task` 因此在返回值里附一个**只读** `warning`：绑定里的 `env`
  ref 指向哪张 embodiment 卡（无 `env` = 跟着 session 自己的 base，不出声），那张卡
  的 `third_party` 拿去问目标 session 活着的解释器（`runtime_status.json` 的 pid →
  `/proc/<pid>/cmdline` 的 argv[0]）能否 import。它**不拦**——`submit_brief` 的零校验
  是防权限洗白的设计，runtime 仍是唯一权威；任何读不出来的情形（绑定无 `env`、
  runtime 已死、pid 被回收）一律**不给** `warning` 键，错的警告和错的文档一样坏。
* **测试 marker 镜像 robosuite 模式**：新 marker + conftest 同款 `find_spec` 自动跳过。
  base lane 会多出跳过项——§3.2 的跳过清单**同 commit 更新**。sim 自己的
  venv 里只跑 `pytest -m <marker>`（那里的 robosuite 是 master，不许跑 robosuite lane）。

### 5.3 卡片要提供什么（以 `plugins/embodiment_robocasa/` 为例）

manifest：`actuation="sim"`、`needs_sim=true`、
`third_party=["robocasa","robosuite","mujoco"]`（边界测试读它；任何其他卡 import
robocasa = embodiment 泄漏）。

* **env provider**：包 `robocasa create_env`；TASKS 注册表把 mission 任务名映射到
  robocasa env 名 + kwargs（seed 透传给场景生成 rng）。每次 make 归档
  `get_ep_meta()`（layout/style/object_cfgs 场景指纹）进 episode 封存。
* **percept provider**：privileged obs（`{name}_pos/_quat` 全物体 + fixture 状态）
  加噪走现行 oracle/percept 双轨——robocasa 不自带噪声模型，wrapper 里加，与
  robosuite 卡同构。
* **PREDICATES 原语层**：`robocasa.utils.object_utils`（~30 个谓词：
  `obj_inside_of/check_obj_grasped/gripper_obj_far/…`）+ fixture 状态 API
  （`microwave.is_closed/get_state()['turned_on']`、`fridge.is_open`、
  `sink.get_handle_state` 等）就是免费 oracle。卡片把它们包成 mission 可引用的
  谓词表；时序 flag（"水开着时菜必须在槽里"）抄 MultistepSteaming 的累积模式，
  每步采样、wrapper 持有，不动 robocasa 源码。**按仓库纪律，把任何 sim 自带谓词当
  gate 用之前，先审计它的区分度**——这里有过一个近乎恒真的抓取检查。
  **那条抓取检查现在审完了**（`scripts/probe_grasp_predicate.py`，合成对照 +
  100 条人类 demo 重放，与 place 谓词同一套方法）：`check_obj_grasped` = 夹爪-物体
  接触 AND 两个手指关节 < 0.035，**没有升起项**，所以"握住"和"碰到"它分不开。把闭合
  的夹爪摆到肉的静止位姿上（肉仍搁在托盘上），7/7 条可构造的对照全读 True；100 条
  demo 的 25261 帧里，它读 True 的 9583 帧中 20.3% 肉根本没升起（<20 mm），2.0%
  既没升起又仍被支撑。所以 grasped 类 verify 一律用 `obj_grasped_secure` = latch
  AND 肉的 live z 比 survey 封存的静止 z 高出 `GraspDriver.SECURE_DZ`——0.08 m 不是
  这里新挑的阈值，是 grasp segment 自己 `done()` 的那一把尺，直接 import 复用，两边
  不会漂开；裸 latch 只作组件用。同一次重放里修好的谓词在 **94/100** 条 demo 上仍
  读 True，漏掉的 5 条全是高层架（rest_z 1.37-1.48）上人类横向抽出、整程没升够
  80 mm 的抓取——诚实的假阴性面，不是"严到永远说 False"。
* **驱动**：PandaOmron 12 维（arm OSC 6 + gripper 1 + torso 1 + base vx/vy/wyaw 3 +
  base_mode 1）。两类脚本化 driver，均为冻结策略、可被治理：
  - `navigate`：privileged fixture 位姿做目标的速度闭环（base_mode=+1），无路径
    规划（robocasa 没有），直线 + fixture 停靠位；撞不动就是诚实失败面。
  - `arm 阶段 driver`：robosuite 卡的分阶段脚本模式平移（grasp/place/开关门/
    按钮各一个 stage 表）。底座前进轴 master 改过向——**以本 venv 实测为准**。

### 5.4 首发 mission —— `kitchen_thaw`（MicrowaveThawingFridge）

一个持久 episode（`EpisodeContext`，`episodic: true`），≥14 节点：
survey(perceive) → plan(decide) → nav-fridge(segment) → verify-at →
grasp-item(segment) → verify-grasped → nav-microwave(segment) → verify-at →
place-in(segment) → verify-inside(`obj_inside_of`) → close-door(segment) →
verify-closed → press-start(segment) → verify-on(`turned_on`) → report(decide)。
每个 verify 读活状态；失败 → 回合内 replan（同一世界重试）。EPISODE horizon 必须
装得下各段 cap 之和：六段 cap 和一度是 2350 而 horizon 是 2000，于是 110/150 标定集
死在时钟上——那正是门禁 `c3_budget_exhaust_dominant` 的真身。`tests/test_kitchen_
thaw_horizon.py` 把 mission 卡里的 `_NOMINAL_STEPS` 与 kitchen_driver 的 `_STAGES`
逐位钉住（卡不能 import 兄弟卡，那个常数是抄的，必须有钉子）。

证据纪律照旧：scratch 种子（<542k）冒烟 → 标定块另 alloc → prereg 先于烧块 →
held-out 一次。首发里程碑是**架构 E2E**（一个 episode 里图执行 + 逐节点活状态
verify + 回合内 replan 在 UI 图谱实时可见），不是成功率——成功率是 RSI 后续的活。

### 5.5 落地顺序（每步有机械验收）

1. **接线**：sim venv 里 `pip install -e $REPO`（底座）+ 新 marker + conftest 钩子 +
   §3.2 跳过清单更新。验收：venv 里 kernel 可导入，base lane 绿。
2. **卡片**：env/percept provider + doctor 绿（sim venv 里 `-m <marker>`）。
3. **驱动**：navigate + 各 arm 阶段 driver，逐阶段独立冒烟（从 reset 驱动到该阶段
   谓词为真）。
4. **mission 卡**：图 + PREDICATES + `episodic:true`，scratch 种子走 runtime 正路
   E2E，`runtime_events` 全程可见。
5. **路由 + UI**：cockpit 加这个 runtime、board `session` 参数三脸、浏览器亲验图谱
   面板实时执行。

<!-- ponytail: 一个 session 参数 + 一张卡 + 一个 marker，没有插件间 RPC、没有
     跨 venv 序列化层；等第三个 sim 真出现共性再抽象。 -->

### 5.6 LIBERO（第三个 venv，脚手架）

骨架：`sims/libero-venv` + `plugins/embodiment_libero/`（`enabled=false`，仅
embodiment.env 座）+ `libero` marker（conftest `find_spec` 自动跳过，照 robocasa 模式）。

装法（**已验证的偏离项，不是照抄 upstream requirements.txt**）：

```bash
cd $REPO/sims
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO   # @8f1084e，含 405M 资产/bddl/init_files，无需另下大文件
uv venv -p 3.10 libero-venv        # 3.12 不行：numpy==1.22.4 无 3.11+ 轮子
VIRTUAL_ENV=$PWD/libero-venv uv pip install \
  "numpy==1.22.4" "robosuite==1.4.0" "mujoco==2.3.2" "bddl==1.0.1" \
  "hydra-core==1.2.0" easydict future "matplotlib==3.5.3" \
  "cloudpickle==2.1.0" "gym==0.25.2" opencv-python \
  "torch==2.4.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
  termcolor pynput "pytest-timeout>=2"
VIRTUAL_ENV=$PWD/libero-venv uv pip install -e ./LIBERO   # 只登记 dist-info，见坑 1
echo "$PWD/LIBERO" > libero-venv/lib/python3.10/site-packages/libero_repo.pth
```

跳过未装（纯训练用，env 创建/评测不需要）：wandb、transformers、robomimic、
einops、thop。torch 必须有（`libero.libero.benchmark` 用 `torch.load` 读
init states），cpu 轮子即可。

坑（全部实测踩过）：

1. **upstream editable 安装映射为空。** setup.py 用 `find_packages()`，但仓库顶层
   `libero/` 没有 `__init__.py`，PEP-660 finder 的 MAPPING 是空的，
   `import libero` 直接 `ModuleNotFoundError`。解法：一行 `.pth` 指向 checkout，
   `libero` 走隐式 namespace package。推论：任何 cwd 里有 `libero/` 目录都会
   遮蔽它——robocasa namespace 坑的同族。
2. **`~/.libero/config.yaml` 是机器级全局单例。** 首次 import 写入绝对路径并永久
   复用，且无 config 时首次 import 会 `input()` 交互提问，无头环境直接挂死。解法：
   预写 `sims/libero-venv/.libero/config.yaml`（五个键指向本 checkout），运行时
   `LIBERO_CONFIG_PATH` 指过去——卡片 `make_env` 里
   `setdefault(sys.prefix + "/.libero")`，全局文件永不碰（别的项目还要用）。
3. **mujoco 不钉会解析到 3.x。** robosuite 1.4.0 元数据只写 `mujoco>=2.3`，uv
   解析出 3.12.0（API 早不兼容）。钉 `mujoco==2.3.2`。
4. robosuite 1.4.0 的轮子少声明依赖：termcolor、pynput 要手补；LIBERO 侧
   matplotlib/cloudpickle/gym 同理（envs 包顶层 import 它们）。
5. venv 里跑本仓库 pytest 需要 pytest-timeout（pyproject addopts 带 `--timeout=300`）。

冒烟（EGL 无头，已通过）：

```bash
cd $REPO   # 或任一 worktree
MUJOCO_GL=egl PYTHONPATH=. sims/libero-venv/bin/python -c "
from harness.spec import EpisodeSpec
import plugins.embodiment_libero as card
import numpy as np
p = card.provider(); spec = EpisodeSpec(task='libero_pick_bowl', seed=420001)
env = p.make_env(spec); obs = env.reset()
for _ in range(10): obs, r, done, info = env.step(np.random.uniform(-1, 1, 7))
print('ok', obs['agentview_image'].shape)"
sims/libero-venv/bin/python -m pytest tests/test_libero_marker.py -m libero   # 1 passed
```

obs 是 dict：逐物体 `{name}_pos/_quat/_to_robot0_eef_*`、robot0 proprio、
`agentview_image`/`robot0_eye_in_hand_image`（128×128×3）、`object-state`；
动作 7 维（OSC 6 + gripper 1）。谓词面：bddl goal（如
`(On akita_black_bowl_1 plate_1)`）是 terminal oracle 候选——先审计其区分度再当
gate 用。

---

## 6. 接入你自己的模型

三条缝，都已建好并有测试。每条缝 = 一次 manifest 编辑加一条验证命令，**永远不用改
kernel**。插入点的封闭清单在 ARCHITECTURE.md §3.1。

| 你有 | 缝 | recipe |
|---|---|---|
| 会写任务图的 VLM | `task.planner`，经 `[task_bindings.*]` 的 planner ref | §6.1 |
| 会出动作的 VLA | `policy.driver`，走 websocket policy server | §6.2 |
| 本体特有的修复动作 | 你自己本体卡里的 `[recoveries.*]` | §6.3 |
| 不改 kernel 的一次实验（换 executor / 调 tunable / 挂候选卡） | `submit_proposal`（三面同名；MCP 面收 dict）→ evolve 下一轮消费 | §4.0 |

### 6.1 把 planner 换成你的 VLM

一张 brief 的 task 字符串解析到某张卡 manifest 里的一行 `[task_bindings.<task>]`，
那里的 `planner` ref 在这一次运行里被挂成 `task.planner`。活样例是
`plugins/planner_vlm/manifest.toml`：

```toml
[task_bindings.stack_vlm]
policy    = "plugins.policies:stack_scripted_provider"   # SAME policy as stack
planner   = "plugins.planner_vlm:provider"               # only the planner swapped
catalogue = "plugins.planner_vlm:CATALOGUE"              # card-authored vocabulary
oracles   = "plugins.planner_vlm:ORACLES"                # declared verify predicates
```

这就是整条 A/B 通道：`{"task": "stack"}` 跑确定性 planner，`{"task": "stack_vlm"}`
跑 VLM，policy 相同，其余一切相同。两条都用 `submit_brief` 投（scratch 种子 42xxxx
永不烧账本）：

```json
{"kind": "task", "task": "stack_vlm", "seed": 424242}
```

**指向你的模型。** planner 通过唯一的 `ModelEndpoint` 缝跟模型说话——任何 OpenAI 兼容
的 `/v1/chat/completions` 服务。两个 preset 在
`plugins/model_endpoint/__init__.py:PRESETS`：

- `local_sglang` —— `http://127.0.0.1:30000/v1`，key env `QWEN38_API_KEY`（可选），
  `model = None` 从 `GET /models` 懒解析。**任何**在那个端口上的 OpenAI 兼容服务
  （sglang、vLLM、llama.cpp）零编辑即可用。
- `deepseek` —— `https://api.deepseek.com/v1`，`model = deepseek-chat`；
  `export DEEPSEEK_API_KEY=...`。key 走的是环境变量**名**、绝不是值——秘密不进哈希链。

一个已在代码里核实的坑：runtime 的 task 路径挂 binding 的 planner ref 时**不带
params**，planner 用它**自己**的默认值按 ref 解析 endpoint。当前与 3080 的操作员设置
对齐到 `deepseek-official / deepseek-v4-pro`；`model_endpoint` 先读
`DEEPSEEK_API_KEY` 环境变量，缺失时只按同名 ref 从 `$DSH_HOME/.credentials.yaml` 读取，
key 不进入 manifest、brief、prompt、日志或 endpoint identity。
改 `plugins/model_endpoint/manifest.toml` 的 params **不会**给 planner 改道——那些
params 只在有东西 kernel-mount `model.endpoint` 时才起作用，而今天没有。要用托管 API
或别的端口，改 planner 的 `endpoint_params` 默认值（一行；支持逐字段覆盖，如
`{"preset": "local_sglang", "base_url": "http://host:8001/v1"}`）。
**endpoint 身份进 plan sha：换端点就是换实验。**

**你的 VLM 可以做什么、不可以做什么。** provider 用（goal、catalogue、oracles、scene、
budget、last fault）提示，并要求一张严格 JSON 图 `{goal, nodes[], verify[]}`。它**按
构造是不可信的**——`plugins/task/validate.py:validate_plan` 在每张图派发前运行，拒绝：
未知 skill/参数/参数类型（planner 只能从卡片作者写的 catalogue 里选，绝不发明）、
非更早的 `after` 边、空 `nodes`、空或未知谓词的 `verify`、任何没被 verify 覆盖的
manipulate/segment 节点、任何丢掉或改写已完成节点的 replan（Typed/Grounded/Supported/
Covered 与 replan 单调性都委托给 `harness/protocol.py`，见 §9）。被拒的图花掉一次 replan
（`invalid_plan` 折回），绝不崩；解析不了的回复给且只给一次带上解析错误的重问，然后
返回一张空 nodes 图——validator 保证会拒——**卡片绝不悄悄编一张图出来**。

provider 上的 `deterministic = False` 是对 plugin_doctor 豁免的**响亮 opt-in**：
LLM planner 只做形状校验，绝不双跑比对。它仍然承诺"生成一次然后冻结"——
(endpoint, task, seed, fault) 键上的第一张图在进程生命期内被缓存，所以同进程重放挂的是
逐字节相同的图。

**注册一条平行 binding 做 A/B**：把 `planner_vlm` 卡的形状抄进你自己的卡目录（卡之间
永不互相 import，只有 ref 字符串）：`plugins/planner_yourvlm/{manifest.toml,__init__.py}`
带一行 `[task_bindings.stack_yourvlm]` 指向你的 provider。binding 名在所有卡之间取并集，
重名 loud fail。**catalogue 要限制在被绑的 policy 真能驱动的 skill 上**——给活模型提供
一个它的通道执行不了的 skill 是**实测过的**失败模式，不是假想。

验证：

```bash
PYTHONPATH=. .venv/bin/python scripts/plugin_doctor.py plugins/planner_vlm
```

（Tier A 把每条 binding ref 都经真契约闸加载；planner 冒烟探 `available()`，没有端点
应答时**响亮地 SKIP**。离线逻辑由 `python -m pytest tests/test_planner_vlm.py
tests/test_model_endpoint.py` 覆盖，不需要端点。）

#### 6.1.1 静态 skill library 与 `pack_all_robocasa`

共享技能库在 `skill-library/records/`，一技能一份 `SkillRecordV0` JSON（模式在
`harness/protocol.py`，§9）：

```text
skill-library/records/<skill>.json   # args / requires / ensures / clobbers（与本体无关）
                                     # + bindings[<本体>]（task_template+backend 或 episode kwargs）
harness/skill_library.py             # load_records / select / catalogue_of / planner_docs
                                     # / segment_specs / skill_specs
```

上层只看 `navigate / grasp / carry / place`——**和 §7.7 capability 记录同一套词**，不是
第二套命名；例如 `grasp(object="hot0")` 在 RoboCasa 绑定成 `grasp_hot0`。LIBERO 复用
`grasp/place` 的**同一语义契约**，但当前卡只有 env 骨架，没有合格的 policy 和 terminal
oracle，所以绑定明确写着 `implemented=false`，不会被暴露给 planner——不能执行的技能绝不
假装存在。support（摞放）是和 containment（装入）不同的后置条件，所以 `place` 之外另留一个
`place_on`——它绑在 robosuite Stack 的脚本 driver 上（`skills.place_on`），是真有东西执行才
留的名，不是凭空的第六个契约。

**一个事实，一个家。** record 的 symbolic 半边（`requires` / `ensures` / `clobbers` 谓词引用）
与本体无关；测过的数字不在这里，在 capability 记录里（§7.7，
`runs/pi05-campaign/round99_skills/`）。名字共享时二者不冲突：record 说"这个技能是什么"，
capability 记录说"这个 executor 测出来能做到几成"。record 的 `evidence[<本体>]` 目前为空，
等有 store 再填。

`plugins/mission_pack_all/` 是第一条闭环：manifest 把共享 catalogue、技能说明、场景物体清单
和 `target_by_object` 交给 `planner_vlm`；VLM 为四件食物逐件生成
`navigate -> grasp -> carry -> place` DAG；`plugins/task/validate.py` 查图的技能、必填参数、
类型、拓扑和 verify 覆盖；派发前 `_segment_spec` 再查物体/容器 grounding，然后才把抽象节点
翻译为 `lunch_driver.py` 已实现的 RoboCasa stage。调用示例：

```json
{"kind":"task","task":"pack_all_robocasa","seed":424242,
 "instruction":"把所有食物按冷热分别装进正确的保鲜盒"}
```

新增 benchmark 时不需要复制整套 skill：保留能共享的语义契约，只在各 record 的
`bindings[<benchmark>]` 写适配；该 benchmark 特有的动作另加一份 record。
也就是说，**共享的是抽象和图语言，控制器、动作空间、成功谓词仍由 benchmark 自己实现。**

`basket_smoke_vlm` 是更小的端到端冒烟任务：场景固定提供 `item0/item1/item2` 和
`basket`，VLM 只需生成三组 `grasp -> place`，不包含 navigation 或 carry。每个
对象都必须恰好出现一次，`place` 必须在对应 `grasp` 的依赖后且目标必须是 `basket`；
这些约束由 validator 读取 task-authored `planning_context` 执行，不依赖 prompt 自觉。每个
segment 失败后先在**同一张已验证图、同一世界状态**上原地重试一次；仍失败才消耗 VLM
replan，因此偶发的边缘放置不会立刻让长任务重新规划。RoboCasa 的 `place` 上限为
450 step，任务总 horizon 为 4000 step；失败记录同时携带 driver phase、是否进入容器和
是否已经释放，供下一次诊断使用，这些字段不参与成功判定。

启用 `--frames` 时，runtime 会把当前任务渲染帧汇成
`runs/<session>/rollout.mp4`。它是可丢弃的实时产物，不进入证据链，也不会影响任务判定；
任务结束后可直接在执行图的取景窗点击“下载视频”。新任务开始时会替换上一条视频。

#### 6.1.2 自然语言 → skill graph → 可执行组合（`plan_skill_task`）

`sims/robocasa/skill_annotation_analysis/taxonomy/unified_skill_graph.json` 是 RoboCasa365 标注
生成的**只读**统一技能图（生成器 `robocasa/scripts/build_skill_taxonomy.py`；本仓库不复制、不重扫、
不改它）。五种边各有各的语义，绝不混用：`IS_A`（taxonomy 分类）、`HAS_STAGE`（observed skill 的
有序阶段）、`REALIZES`（阶段对应的 canonical 接口）、`DECOMPOSES_TO`（组合技能的有序配方）、
`OBSERVED_TRANSITION`（数据里观察到的相邻转移，**不是**因果）。图里节点的 `executable: true` 是
标注者的本体断言，**不是** binding。

`skill_library` 把这棵 IS_A 树与当前安装的 task catalogue 做只读 union，供 ph-station 的技能库标签页展示。图节点内嵌 HAS_STAGE / REALIZES、DECOMPOSES_TO、annotation label、数据集与计数证据；运行时技能另列 task、policy 与参数 schema。只有同名 catalogue 项算直接 binding；映射到同一 canonical 的 `pick` 等实现只列为候选，绝不冒充图节点已经可调度。

一条闭环，全部复用既有部件：

```
instruction ──► harness/unified_skill_graph.py   检索相关 IS_A 子树（渐进披露：紧凑目录，不是整张图）
            ──► plugins/task/skill_planning.py    路由到一个词表（channel）：
                                                    · robocasa_skill_graph（图词表，plugins/skill_graph_robocasa 卡授权参数/符号 oracle）
                                                    · 或某条由自然语言驱动的 task binding（pack_all_robocasa / basket_smoke_vlm / …）
            ──► plugins/planner_vlm                DeepSeek 严格 JSON {goal,nodes[],verify[]}；坏 JSON 重问一次，再坏就是可拒绝的空图
            ──► plugins/task/validate.validate_plan  runtime 派发前用的同一个闸，不是拷贝
            ──► 服务端展开                          HAS_STAGE / DECOMPOSES_TO 递归，模型自己展开的阶段名当未知 skill 拒绝
            ──► binding 检查                        叶子只有在某张卡的 task_bindings catalogue 里**同名**声明时才算 bound
```

四种 `status`：`executable`（每个叶子都 bound）、`planning_only`（符号链可以展示，`missing_bindings`
逐条列出缺口——标注不是控制器）、`rejected`（`validation.message` 给出校验器原话；模型两次都给不出
合法 JSON 也落这里）、`no_match`（没有词表命中，根本不调模型）。

三张脸调的是同一个函数（`board/planning.py`）：MCP 工具 `skill_library` / `plan_skill_task` / `submit_skill_plan`、
`storecli skill_library` / `plan_skill_task --instruction=… [--channel X]` / `submit_skill_plan --plan=<record>`、ph-station
的 `POST /api/board/skillLibrary` / `planSkillTask` / `submitSkillPlan`（技能库与规划面板）。`skill_library` 与 `plan_skill_task` 都是读，不落盘。
`submit_skill_plan` 是唯一的执行入口：从零重新核验 `composite_plan` 记录（channel 必须是当前装着的
task binding、`validate_plan` 用该任务当下的 catalogue/planning_context 再过一遍、每个叶子 bound），
不合格就拒绝、不落任何文件；合格就投一张**普通** task brief `{"kind":"task","task":…,"instruction":…,
"seed":…}`——走 `submit_brief` 同一条原子落盘，返回 `brief_status` 句柄。常驻 runtime 从 brief 重新
规划、重新校验，是唯一权威；预览链是参考，runtime 封存的计划才是证据。

配置：图路径 `PH_UNIFIED_SKILL_GRAPH`（默认工作区相对路径 `../sims/robocasa/.../unified_skill_graph.json`）；
planner 端点默认走 planner_vlm 自己的 DeepSeek 默认值（key 只经 `DEEPSEEK_API_KEY` 或控制台凭据库），
`PH_PLANNER_BASE_URL` / `PH_PLANNER_MODEL` / `PH_PLANNER_API_KEY_ENV` 可把三张脸一起指到别的
OpenAI 兼容服务（测试就是这样指到假服务器的）。

验收两例（`tests/test_skill_planning.py`、`tests/test_planning_faces.py`、ph-station 的
`ui-ph-panels/tests/plan-*.client.spec.tsx`，模型一律是假的）：

- **Prepare a cup of coffee.** → 图词表；紧凑目录 9/56；链 `CoffeeSetupMug.pick → CoffeeSetupMug.place →
  StartCoffeeMachine.execute → done`；规范展开 `CoffeeSetupMug → Pick → Place`、`StartCoffeeMachine →
  PressButton`；`planning_only`，三个叶子全在 `missing_bindings`，提交被拒。
- **Pack every food item into its assigned tupperware.** → `pack_all_robocasa` 词表；16 个叶子全部 bound
  （`lunch_driver` 的 `nav_/grasp_/carry_/pack_{object}`）；`executable`；提交落 brief，`brief_status`
  在有活 runtime 时 `queued`、没有时 `stalled`。真机/仿真里的实际执行仍是 runtime 的事：这条测试证明的是
  到 inbox 为止的控制流，不是 policy 跑通。

**仍没有真实 binding 的 RoboCasa 技能**：图里 40 个 observed skill 与 16 个 canonical skill 全部
（CoffeeSetupMug、StartCoffeeMachine、Open*/Close*、PickPlace*、TurnOn*、NavigateKitchen、Pick、
Place、PressButton……）。今天有 binding 的是静态技能库那套折叠后的族名（`navigate / grasp / carry /
place`，以及 robosuite 侧仍在用的 `pick` / `place_on`），且各自只在绑定它的任务场景与物体清单内。
要让一个图技能变成可执行，得有一张卡在 `[task_bindings.<task>]` 的 catalogue 里以**同名**声明它并给出
policy——展示别名（`grasp`→`Pick`）不算。

### 6.2 把你的 VLA 放在 socket 后面

`plugins/policy_vla_remote/` 是一张 `policy.driver` 卡，说 StarVLA/openpi 的 websocket
policy-server 协议：连上后的第一帧是服务端的 metadata dict，之后是 msgpack 打包的 dict，
其中 ndarray 走 `__ndarray__` 扩展（**pickle 按构造拒绝**）。VLA 栈——torch、flash-attn、
钉住的 transformers——留在**它自己的 venv/进程**里，在 socket 后面。这不是可选的卫生
习惯：harness 底座是 `numpy>=1.26` 且无 torch，而 sim venv 之间本来就不能共享 ABI
（venv-per-sim）；socket 就是同一个隔离动作用在 policy 缝上。harness 侧的传输层是三个
vendored 的 MIT 文件加上：

```bash
uv pip install -e ".[policy_remote]"    # websockets + msgpack only
```

**openpi 开箱即用**：它的 `serve_policy.py` 说的就是这个协议（默认端口 8000，也正是
provider 的默认值 —— `host="127.0.0.1", port=8000`）。

**握手就是契约检查。** 卡的 manifest params 声明**训练时的观测契约**；连接时
`reconcile()` 拿它跟服务端第一帧 metadata 对：

| manifest param | 服务端回显的握手键 | 没回显时 |
|---|---|---|
| `image_size` | `training_obs_image_size` | 落进 `unverified` |
| `views` | `camera_views`（StarVLA 从不回显） | 落进 `unverified` |
| `chunk` | `action_chunk_size` | 落进 `unverified` |
| `unnorm_key` | `default_unnorm_key`（列在 `available_unnorm_keys` 里也算） | 落进 `unverified` |
| `checkpoint_sha`（可选） | `checkpoint_sha` | **抛** |

任何被回显的键不匹配就**在 mount 时抛**——train/test 漂移响亮地失败，绝不表现为悄悄
变低的成功率。前四个键服务端没回显就落进 `handshake["unverified"]`（openpi 服务端常发空
metadata —— 合法，但那样这道闸什么也验不了），整份握手记录随驱动进入 episode 证据。
提交在 manifest 里的值是 openpi LIBERO π0.5 约定的模板——按你的 checkpoint 设。

**第五个键 `checkpoint_sha` 是另一类东西，规则也不一样**（§7.4）。前四个说的是"在什么
观测契约下训练的"，同一个任务的两次 π0.5 训练按构造共享这四个值——它们分不出**是哪份权重
在应答**。`checkpoint_sha` 分得出，所以它**失败朝闭**：manifest 声明了它而服务端不回显，
就跟服务端回显了一个不同的摘要一样**抛**，不进 `unverified`。理由是"没人回答"和"错的权重
回答了"在证据上无法区分。它同时是**可选的**：manifest 不写，就完全不验这一项（见 §7.4
末尾为什么这个 opt-in 是诚实的）。

**包你自己的模型**：在你的模型 venv 里复用 vendored 的 server（它只 import
websockets + msgpack + numpy）：

```python
from plugins.policy_vla_remote.websocket_policy_server import WebsocketPolicyServer

WebsocketPolicyServer(
    policy=my_policy,          # any object with predict_action(**obs) -> {"actions": ...}
    host="0.0.0.0", port=8000,
    metadata={"training_obs_image_size": [224, 224], "action_chunk_size": 10,
              "default_unnorm_key": "my_dataset"},   # echo the table above
).serve_forever()
```

`actions` 必须是 `[T, D]`（或 `[B, T, D]`，取第一个 batch 元素）且**已经反归一化**——
norm stats 永不跨边界，它们跟 checkpoint 一起留在服务端。驱动每个 chunk 推理一次，
每步弹出一个动作。

**chunk 怎么执行，是三个 opt-in 的 serving 参数**（跟 `host`/`port` 并列，不进被
reconcile 的契约——服务端对它们没有意见可验；它们封在 `handshake["execution"]` 里，
所以一份记录仍然说得出自己是在哪种执行策略下跑出来的）。三个都不写时行为跟以前**逐字
节相同**：一次推理喂满 10 步，其中 9 步开环。

| param | 作用 | 不写时 |
|---|---|---|
| `replan_every = k` | 只执行 chunk 的前 k 个动作就重新推理（`k=1` 是每步闭环） | 抽干整个 chunk |
| `ensemble = m` | k < chunk 时多个 chunk 预测同一个 timestep，按 `exp(-m*age)` 加权平均（越新权重越大） | 不做 ensembling |
| `discrete_dims = [i, ...]` | ensembling **不许**平均的维度，取最新 chunk 的原值 | 空——所有维度都平均 |

`ensemble` 不带 `replan_every` 会**抛**：没有重叠就没有东西可平均，一个读起来像开着
实际什么都不做的旋钮比没有这个旋钮更糟。

`discrete_dims` 是给 `control_mode`、`gripper` 这类**两值决策**用的：+1 和 -1 的平均是 0，
controller 会把它读成第三件事——既不是新 chunk 的意思，也不是旧 chunk 的意思。把平均值
按符号 snap 回去也不行，那是拿过时的预测做多数投票，而它恰好会审查掉少数派决定（本仓库
实测的 π0.5 checkpoint 只在 8.1% 的步上命令 base mode）。所以这些维度直接取最新 chunk 的
值，其余维度才平均。

代价是线性的：实测单次推理 **155 ms**（π0.5 LoRA，RTX 4090，warm），控制环 20 fps
= 50 ms/步，所以 `k=1` 跑不动实时（3.1×超时），`k>=4` 才摊得进预算。ensembling 本身几乎
免费（10 个 chunk 的加权平均 < 0.1 ms）。`scripts/probe_pi05_rollout.py --replan-every /
--ensemble` 每个 episode 记 `inference_calls` 和 `base_mode_share`，先测再声明。

**测了，是个 null。** 这三个旋钮是为一个具体怀疑造出来的：π0.5 LoRA（100 条 RoboCasa
place demo）单步模仿接近天花板、闭环接近零，而 serving 路径把 10 个动作的 chunk 抽干，
每 10 步有 9 步开环——闭环失败有多少是这个？配对实验（同一 checkpoint `ea09cb15…`、同
一批 scratch 种子、`--split train`，只有执行策略变），`scripts/compare_serving_arms.py`
折出的表在 `runs/pi05-campaign/round98_serving_ablation/`：

| arm | n | place `obj_in_microwave` | base mode 占比 | inference/episode | s/episode |
|---|---|---|---|---|---|
| sealed baseline（抽干 chunk） | 10 | 1/10 | 8.1%* | 240 | 111 |
| control 重跑（抽干 chunk） | 20 | 0/20 | 5.2% | 240 | 106 |
| `replan_every=1` | 20 | 2/20 | 7.3% | 2400 | 359 |
| `replan_every=1, ensemble=0.25` | 10 | 0/10 | 3.2% | 2400 | 273 |

（*baseline 早于 per-step 计数器，只有 `action_trace` 的子采样估计。demo 是 20.09%。）

`k=1` 对 control 是 2/20 vs 0/20，Fisher p=0.49、配对 McNemar p=0.50；对 sealed
baseline 是 2/10 vs 1/10，p=1.0。**2/10 不是 1/10 的改进**——先看噪声地板：baseline 和
control 执行策略逐字节相同、种子相同，place 仍然 1/10 vs 0/10，grasp 有 3/10 的种子翻面
（openpi 每次请求抽新的 noise key，这个策略是随机的）。所有臂之间的差都在这条地板以内。

机制指标说得更直接：每步重新推理**没有**把 base mode 占比拉回 demo 的 20.09%——四个臂
落在 3.2–7.3%，而同一个执行策略跑两次就能从 5.2% 走到 8.1%。所以 chunk 被开环抽干不是
闭环失败的主因，剩下的要到策略自己身上找。`k=1` 是**诊断臂，不是可上线配置**：2400 次
推理 × 155 ms 已经把 episode 变成推理绑定（359 s），仍然 3.1× 超 20 fps 预算。

（这张表的 grasp 列被刻意省掉了：`obj_grasped` 的 latch 在这轮跑到一半时被 845b57a 换
成了 `obj_grasped_secure`，同一条 k=1 臂上 latch 读 9/12、secure 读 2/8——两把尺子的
数不能并排放，`compare_serving_arms.py` 遇到混合尺子直接拒绝比较而不是悄悄平均。）

这张卡 ships `enabled = false`，因为 `plugins/policies` 拥有 `policy.driver`
（一条缝一张卡）；某条车道要上线时把它打开、把在位的那张关掉。

验证（你的 server 在跑）：

```bash
PYTHONPATH=. .venv/bin/python -c "from plugins.policy_vla_remote import provider; \
print(provider(image_size=[224, 224], chunk=10).connect())"
```

（打印封存的握手记录，或者在契约不匹配时带着服务端完整 metadata 抛出。离线：
`python -m pytest tests/test_policy_vla_remote.py` 跑 codec、闸、驱动和一次进程内 socket
往返；`scripts/plugin_doctor.py plugins/policy_vla_remote` 在没人监听时响亮 SKIP。）

### 6.3 注册恢复原语

RSI 的修复是**本体词汇**，所以由说这套词汇的本体卡声明——你卡的 manifest.toml 里
`[recoveries.<name>] ref = "module:attr"`，由 `harness.manifest.discover()` 像 mounts
一样折叠（重名 loud fail）。`plugins/embodiment_robosuite/manifest.toml` 和它的
`recoveries.py` 是模板：

```toml
[recoveries.regrasp]
ref = "plugins.embodiment_robosuite.recoveries:REGRASP"
```

每个 ref 必须满足 `harness.contracts.RecoveryStrategy` —— 一个 frozen dataclass 就够：

```python
name: str                                          # MUST equal the [recoveries.<name>] key
steps: tuple[tuple[str, int, float, float], ...]   # (phase, duration, dx, dy)
rationale: str
length: int          # property: step-duration upper bound
uses_feedback: bool  # property: any servo_* phase present
```

robocasa 卡（`plugins/embodiment_robocasa/recoveries.py`，12 维 PandaOmron 词汇，`RobocasaRecoveryActor` 执行）除 `regrasp_kitchen`/`redock_retry` 外声明三条 reach 修复，
供 §9.3 的 recovery 节点用：`reapproach`（升回 hover，按实时目标位姿再下降）、`base_nudge`（底盘向目标 xy 微移 ≤0.15 m 后重 hover）、`release_reset`（张开、抬升、回 hover）。
目标取活跃 stage：place/drop 段是其 drop point，其余是其 obj_name 的实时位姿。

`plugins/rsi/repertoire.py` 在加载时解析每个 ref 并对 Protocol 做 isinstance 校验——
形状不对或名字与键不一致会在那里失败，绝不会在修复中途。`steps` 的 phase 是你自己卡的
词汇（robosuite 的 above/descend/close/lift 来自 `harness/spec_tabletop.py`）；一条策略
**永不跨本体借用**。

**没有声明 = 一次诚实的拒绝，不是回退。** 没有 `[recoveries.*]` 的卡回答
`strategies_for(card) == []`，RSI 链的 `c5_recovery_primitive` 门禁逐字报出这个本体没有
注册恢复原语——并指向 robosuite 的 servo 原语当模板，**绝不现编一个动作填空**。
（`hasattr` 探针已被证据否掉，见 §4.4。）

验证：

```bash
PYTHONPATH=. .venv/bin/python -c "from plugins.rsi import repertoire; \
print(repertoire.strategies_for('embodiment_yourcard'))"
```

（跑折叠加 isinstance 和名字闸，覆盖每条声明的 recovery；你卡的名字会打印出来，
`[]` 表示什么都没注册。）
---

## 7. 学习策略挂在哪条缝（fast / slow brain）

慢脑：控制台的 VLM 已经在生成节点图（§6.1）。快脑：一个按 segment 微调的 VLA
（openpi π0.5 是当前的选择），作为一张卡挂在脚本驱动**旁边**，绝不取代它们。证据纪律
（§4）不因此改动一个字。

### 7.1 为什么共存是整个设计

今天一个 segment 由一个手写驱动执行。那个驱动是 `code as policy`：一台四相位状态机，
高度、时长、夹爪时序都是常数，腕部旋转硬置零，物体估计在 t=0 读一次之后永不刷新
（`plugins/policies/drivers.py:FrozenPolicy.act`、`harness/spec_tabletop.py`）。

这个形状精确解释了实测的墙：`nav` 成，因为开环开到某个位姿能活下来；`grasp` 到 49%，
因为恢复原语在打补丁；`place` 是 0/22，因为"把手里的东西坐到另一个东西上"需要三样脚本
结构上就没有的东西：活的相对位姿、腕部控制、接触感知。

所以 VLA 不是一个更好的脚本，而是**同一条缝的另一个 provider**。两者必须能同时挂载：

1. **配对门禁需要两边。** RSI 的比较是同种子上的挑战者 vs 在位者。VLA 若取代了脚本，
   就没有在位者可配对。
2. **只治一个 segment。** 归因挑一个节点，其余五段保留脚本。整任务换掉是改六个变量去
   测一个。
3. **冻结的 SkillRecord 必须仍可复现。** 一条安装记录写明它是用哪个 provider 测出来的。
   provider 消失了，记录就不可重放，`scripts/parity_check.py` 也没有东西可逐字节比。

### 7.2 缝在哪：一行 manifest

派发路径本来就是逐 segment 路由的，所以**不需要改 kernel**：

```
plan node {kind: "segment", skill: "place"}
  → SEGMENT_SPECS["place"]              # mission card, pure data
  → EpisodeSpec(task="place_meat", …)
  → workload._governed_segment
      → driver.enter_segment(env, seg_spec)   # heterogeneous episodic driver
      → gov.governed_segment(...)              # bundle/recovery governance
```

`SEGMENT_SPECS` 是 mission 卡里的**纯数据表**。所以给某一段选一个不同的执行器是一次
manifest 编辑，不是代码编辑——和 task binding 可换是同一个性质。

**改动就是：一条 segment spec 可以指名自己的执行器。**

```toml
# plugins/mission_kitchen_thaw/manifest.toml
[segment_executors.place]
ref = "plugins.policy_vla_remote:provider"
```

没有这一行，该段保留 mission 自己的驱动——所以每个现存 mission 逐字节不变，而 VLA 卡上
的 `enabled = false` 让底座折叠和它的 sha 永不移动。

### 7.3 三层各归谁

| 层 | 谁 | 这里会变吗 |
|---|---|---|
| 任务 → 节点图 | VLM（控制台的 backbone） | 不变 —— `planner_vlm` 已经在 |
| 节点 → segment spec | mission 卡的 `SEGMENT_SPECS`（纯数据） | +1 条可选 executor ref |
| **segment → 动作** | **脚本驱动 *或* 学习策略** | **就是这一层** |
| 有没有做到 | 卡片声明的谓词读活状态 | 不变 |
| 失败怎么修 | 本体卡折出来的 `[recoveries.*]` | 不变 |

慢脑与快脑之间的契约**就是那条 segment spec**——一个子目标加预算。它不发自由文本，
也永远不发动作。这条边界是两个脑打不起来的原因：一个决定**做哪一段、什么顺序**，另一个
决定**在这一段里怎么动**。

### 7.4 冻结机制就是握手

`plugins/policy_vla_remote/` 的握手校验（§6.2）**同时也是冻结机制**：SkillRecord 存
checkpoint 的**摘要**，绝不存权重（GB 级）。

```
harness (base venv)  ──websocket+msgpack──▶  policy venv (JAX/torch)
   policy_vla_remote card                    scripts/serve_vla_openpi.py
   handshake gate ────── checkpoint_sha ─────── 第一帧 metadata
```

**摘要是什么。** `checkpoint_sha` 是 64 位小写 sha256 hexdigest，算的是**权重字节本身**：
遍历 checkpoint 下的 `params/` 和 `assets/`，按 POSIX 相对路径排序，把每个文件的
`relpath.encode() + b"\0" + file_bytes` 依次喂进同一个 sha256。不哈希路径、不哈希 run
名——那些能改名，而 SkillRecord 的身份主张必须扛得住改名。这是
`plugins/policies/bc.py` 里 `MLPPolicy.sha()` 的同一招（那边哈希的是 numpy 权重数组的
`tobytes()`）。闸只做字符串比对，不会去验证对面是不是真按这个规约算的——所以这条规约是
两边必须共同遵守的契约，而一条两边各写一遍的契约会漂。所以它只有**一份实现**：
`plugins/policy_vla_remote/__init__.py` 的 `checkpoint_sha()`（stdlib-only，1 MiB 分块读，
9 GB 的 checkpoint 约 6 秒），闸这边和回显那边调的是同一个函数，填 manifest 的算式也是它。

**为什么只哈希 `params/` 和 `assets/`，不哈希整棵树。** 摘要覆盖的是**决定应答的东西**。
orbax 的 `train_state/` 是优化器状态——它决定下一个训练步，永远不决定一次应答——而且它占
9 GB 里的 3.1 GB，是磁盘紧张时第一个被删的东西。把它算进去，等于删一次没人动过的权重就要
换一次身份，逼着每份声明过它的 manifest 重新声明。`assets/` **要**算：里面是 norm stats，
同样的权重配不同的统计量，反归一化出来的动作就不一样——按这里唯一算数的定义，那是**另一个
策略**。

**服务端怎么回显。** `scripts/serve_vla_openpi.py` 是给 openpi checkpoint 的包装器：它
**不 fork 也不 vendor** openpi 的 server，只是把 `create_trained_policy` 建出来的 policy
和一份带 `checkpoint_sha` 的 metadata 交给 openpi 自己的 `WebsocketPolicyServer`。它跑在
**openpi 的解释器**下（`PYTHONPATH=<harness> <openpi>/.venv/bin/python
<harness>/scripts/serve_vla_openpi.py --checkpoint-dir … --config …`），这正是 §6.2 那条缝：
模型栈留在自己的 venv 里，harness 这边只要 websockets+msgpack+numpy。它在 `scripts/` 而
不在卡里，因为 `plugins/` 只能 import 自己 manifest 声明的东西（`tests/test_boundaries.py`
是闸），而 openpi 恰恰**不是**这张卡的依赖——卡的全部意义就是模型栈在 socket 那头。
`training_obs_image_size` / `action_chunk_size` / `default_unnorm_key` 一律从解析出来的
`TrainConfig` 读，不在包装器里重写一遍；`camera_views` **不回显**——slot 顺序在 config 的
input transform 里（RoboCasa 是 `RoboCasaInputs`），没有可问的接口，手抄一份就是那种会
悄悄过期的第二份拷贝，所以它照旧留在 `handshake["unverified"]`。
`--print-sha` 只算摘要不碰 GPU。

**这条路已经端到端验过**（`scripts/probe_vla_handshake.py`，对着真 server 真 socket，不是
对着 `reconcile()` 的 dict）：摘要相符 → MOUNTED；摘要差一个字符 → REFUSED（`handshake
mismatch`）；manifest 声明了摘要而**原样的** openpi `serve_policy.py` 发 `{}` → REFUSED
（`handshake gap`）——注意最后这一条里两个 server 加载的是**同一份权重**，被拒的理由不是
权重错了，而是**证不出来**。

**闸怎么判。** manifest 声明了 `checkpoint_sha` 时：服务端回显同一个摘要才 mount；回显
了不同的摘要**抛**；**一个字都不回显也抛**（错误信息是 `handshake gap`，不是
`handshake mismatch`）。最后这条是刻意跟前四个键分开的——观测契约那四个键容忍部分回显，
因为 `views` 上游根本不回显、openpi 服务端默认发 `{}`，严格化会让这张卡压根挂不上；而身份
键是"声明出来就是为了被回答"的，"没人回答"和"错的权重回答了"在证据上无法区分，所以它必须
**失败朝闭**。

**它是 opt-in，这是有意的。** manifest 不写 `checkpoint_sha`，就没有这道身份闸——原样的
openpi `serve_policy.py` 发的是 `{}`，强制要求摘要会让任何没包过的服务端挂不上，进而逼人
往 manifest 里填一个没人真算过的摘要：**缺席的身份主张是诚实的沉默，编造的是 SkillRecord
里的谎**。封存的握手记录里 `contract` 有没有这个键是看得见的，所以"这次 mount 关于权重
什么也没证明"在下游读得出来；某个成对比较**要不要求**已验身份，是封 SkillRecord 的人的判断，
不是一个 websocket 客户端的判断。反过来说：**执行模式下要把某个 delta 归因给这个 executor
的成对比较，manifest 里就必须有这个摘要**，否则那条归因是空的。

一个已知残留：`connect()` 对 `self._client is None` 幂等，而协议只在连接后的**第一帧**发
metadata——所以服务端在一条活连接上热重载了不同权重，这边不会重验。协议层面重验就等于重连；
现在的规约是**一次 mount 一次身份证明**。

### 7.5 数据：RoboCasa 自带示范

已在 `sims/robocasa/robocasa/utils/dataset_registry.py` 核实：每个 atomic task 都同时带
`human_path`（遥操）和 `mg_path`（MimicGen 生成），由
`robocasa/scripts/download_datasets.py` 拉取。

这解掉了"过滤式行为克隆"会撞上的堵点：克隆我们自己的成功 rollout 对 `place` 行不通——
脚本在那里从没成功过，没有可模仿的东西。RoboCasa 的示范不依赖我们的脚本好不好。

**未定的数、且它卡住下面一切**：place 形状的 atomic task 到底有多少条示范，以及它们的
动作空间是否匹配我们的 `PandaOmron` 12 维挂载。**训练前先量。**

### 7.6 做的顺序，和硬约束

每一步以一个事实结束；产出否定事实的那一步**诚实地停掉整条链**，而不是抱着希望往下走。

1. **先审计谓词。** 这个仓库有过一个近乎恒真的抓取检查留下的疤。一条不能区分的谓词会
   毒化训练过滤器并把谎言烤进权重里，那比在规则里找难得多。先在重放的示范上证明
   `place` 的谓词能把成功和失败分开，再谈收数据。
2. **数数据。** registry → place 形状任务有多少条 human + MimicGen 轨迹；确认动作空间
   匹配。太少是一个有效的停止条件。
3. **轨迹采集**（只在需要我们自己的数据时）：`plugins/task/workload.py` 目前丢弃
   obs/action。存进内容寻址的 `datasets/` 根，由训练 prereg 按摘要引用，**在封存链
   之外**——它是数据，不是证据，而且很大。
4. **训练卡**放在它自己的 venv 里（sim 卡隔离那套模式）。产出：checkpoint + 摘要。
5. **接执行器**：`[segment_executors.<node>]`，打开 VLA 卡，服务那个 checkpoint，
   并证明握手闸**会拒绝错的 checkpoint**。
6. **过门禁**：脚本驱动当在位者，微调策略当挑战者，**同种子**，配对 → blind twin →
   held-out 只评一次。晋级写一条 SkillRecord，写明 checkpoint 摘要。
   **NO-GO 是正常结果；阈值不会为了造一个晋级而移动。**

硬约束：

- **一张 24 GB 卡。** π0.5 LoRA 约 22.5 GB，所以训练期间必须停掉 backbone LLM（操作员
  侧栏有开关，§2.3），而且训练不能和一次标定的十个 EGL worker 共用这张卡。
- **checkpoint 是 GB 级的。** 摘要进 SkillRecord 和链，权重按摘要寻址放在旁边。
- **时钟换了单位。** 分钟变成小时，所以 dev 的代数会缩水——先把**一代**端到端跑通，
  再谈多代。

### 7.7 技能库怎么说话：capability 记录

到 §7.6 为止，技能库只会说一句话：**"这条 RSI 恢复规则晋级了"**（`plugins/rsi/workload.py`
写的那种记录）。VLM 规划器要把一个 mission 拆成若干 segment、再给每一段挑一个执行器，
需要的是另一句话：**"这个执行器能做这个 skill，在这些前提下，是这么测出来的。"**
这就是 **capability 记录**——同一个 store、同一扇 `publish()` 门，靠 `kind` 字段区分。

```
{"kind": "capability",
 "skill": "place",                     # 规划器可以选的 CATALOGUE 名字
 "task":  "kitchen_thaw",              # mission 上下文
 "binding": {"ref": "plugins.policy_vla_remote:provider",
             "checkpoint_sha": "<64 位小写 hex>"},        # 谁来执行
 "preconditions": ["plugins.embodiment_robocasa.predicates:obj_grasped"],
 "effects":       ["plugins.embodiment_robocasa.predicates:obj_in_microwave"],
 "measured": {"predicate": "plugins.embodiment_robocasa.predicates:obj_in_microwave",
              "successes": 12, "n": 20}}
```

**前提和验收是同一种东西。** `preconditions` / `effects` 用的是 mission 卡的
verify 表（`plugins/embodiment_robocasa/predicates.py` 的 `PREDICATES`）**一模一样**的
`"module:factory"` 引用形式，由 `harness.registry.load_provider` 解析成
`pred(env) -> bool`：一个查入口，一个查出口。**散文不收**——整个设计的要点就是派发器能
把它们**对着活状态求值**，前提不成立就跳过这个 skill。所以组合是谓词级的、现场的：
B 的前提在 A 真正留下的状态上成立，才把 A 接到 B。

**已知边界，故意不加字段。** 谓词名字对上并不保证 B 的实测率能迁移过来——A 交接过来的
状态可能落在 B 被测量的分布之外，而两边谓词都读 True。**有一次交接测量正在跑**，用来
回答这件事在实践中咬不咬人；要加字段等有证据说需要，不是提前加。

**还有一条边界，是谓词形状本身。** 上面例子里 `preconditions` 写的 `obj_grasped` 就是
那条裸 latch。前提/效果这一面只认 `pred(env) -> bool`，而"真握住"需要第二个参数——它
静止时的 z（`obj_grasped_secure(env, z0)`），episode 才知道的东西。所以**能挡住假阳性
的那一版谓词，写不进 capability 记录的前提栏**；今天由 mission 卡的 verify 承担这件事
（`plugins/mission_kitchen_thaw/planner.py:_secure_grasp_verify`，参考 survey 封存的
`meat_pos`）。派发器真开始按前提跳过 skill 的那一天，这条要先补上，否则前提会重演一遍
同一个谎。

**三种执行器共用一个 `binding`。** 脚本驱动是进程内的一个 ref，π0.5 是一张走 socket 的
卡，外部包是和 π0.5 一样的形状——卡片边界本身就是那层抽象，不需要再发明一层传输抽象。
`checkpoint_sha` **有权重的时候才有**：脚本驱动没有权重可以摘要，逼它交一个只会造出这套
schema 存在的目的所要挡的那种假话。身份闸在**权重那一侧**（§7.4 的
`policy_vla_remote.reconcile`：声明了摘要而服务端不回显，就拒绝挂载）。

**六条校验，每条挡一种真实的失败**（`harness/skill_record.py`，在 `publish()` 里执行，
不合格**直接抛**，不是警告后照写）：

| 规则 | 挡住的失败 |
|---|---|
| 每个谓词引用形如 `"module:attr"` | 散文条件派发器没法求值，只能靠人读 |
| `measured.predicate ∈ effects` | 声称一件事、测的是另一件——这套 schema 就是为它存在的 |
| `0 <= successes <= n`，`n > 0` | 无分母的率 |
| `split` 只能是 `train` / `test`（RoboCasa：layout 11-60 / 1-10） | 分不清是能力还是泛化 |
| `checkpoint_sha` 出现时必须是 64 位小写 hex | 记不住是哪份权重拿到的这个数 |
| **未知顶层键一律拒绝** | 打错的字段名把证据静默丢掉，记录 claim 得就比测的多 |

`preconditions` / `effects` **不许为空**：空不是"没有入口条件"，而是"永远适用"这个最宽的
主张，并且和"这一格没人填"完全不可区分。真的无条件适用，就写一条这么说的谓词——那是
可以 grep 的。没有 `kind` 的记录（和 `kind` 是别的值的记录）走的还是原来的路，
**逐字节不变，摘要不动**——恢复记录不带 `heldout_judgement_established` 以外的东西进
`assemble_bundle`，capability 记录因此对治理装配天然是惰性的。

**规划器一次读完：`skill_index`。** `skills()` 交出来的是 N 条按摘要寻址的记录；VLM 要
的是**一份**能塞进 context 的文档。`harness.skill_record.skill_index(records)` 就是那份
文档，**每次现算，永不存第二份真相**（存下来的索引会跟记录漂，而且没人会发现）：

```
skills: skill 名 -> [{digest, binding, preconditions, effects, measured{successes,n}}]
edges:  [{from: A, to: B, via: [共享的谓词引用]}]        # B 的前提 ⊆ A 的效果
```

`edges` 是纯集合包含，没有推断、没有模型调用、没有启发式。挂点在 `scripts/harness_runtime.py`
的 boot：和封 `skills_manifest` 用的是**同一次读**，落成 `<session>/skill_index.json`——
和 `runtime_status.json` 同一类的 live state，每次 boot 覆写，永远不进封存链。

**脚本执行器的记录已经发布**（`scripts/publish_pi05_capabilities.py`）：五条记录——
脚本的 navigate / grasp / carry / place 加 π0.5 的 place binding（checkpoint
`ea09cb15…`）——数字**只从封存的 episode 文件里读**（secure 抓取尺子），不接受手填。
π0.5 因此是表里普通的一行，不是特例。这张索引照出的组合图（同一批 10 个 seeds 上）：

```
navigate 10/10 → grasp 6/10 → carry 3/10 → place {scripted 0/10, π0.5 0/10}
```

这正是这套 schema 存在的意义：mission 的失败不再是一个数，而是一条能指认哪一格在
耗散的链——这一轮指认的是 grasp 和 carry（脚本侧），不是 place。记录本身封存在
`runs/pi05-campaign/round99_skills/`（证据，不进 git）；发布器进 git，随时可以对着
新一批封存证据重发。

---

## 8. 静默失败面与 `health()`

三起事故同一个病：**流水线的某一段死了，而每一个 face 读起来都正常。** 共同的根：
`runtime_status.json` **是一个文件，而文件比写它的进程活得久**——读路径里没有任何东西
去问过内核那个 pid 还在不在。

### 8.1 第一条命令

```
scripts/cockpit --status        # 什么都不启动，不需要 node；exit 1 = 有问题
```

同一张脸三种叫法，一个实现（`board.store.health`）：

| face | 调用 |
|---|---|
| 操作员终端 | `scripts/cockpit --status` |
| 控制台 / agent | `mcp__physical-harness__health` |
| UI bridge / 脚本 | `python -m board.storecli health [PORT] --runs runs/` |

它在一次调用里覆盖：每个进件 session 的 runtime 活性（问 `/proc`，不是问状态文件）+
模式 + 心跳年龄 + inbox 积压 + `processing/` 孤儿，然后是控制台和模型服务。
**`problems` 是要读的那个列表**，其余都是它背后的证据。

**停着的 runtime 加空 inbox 被刻意判定为"不是问题"。** 真机器上退役的 session 目录比
活的多，而一个永远红着的健康面没人会读——三起事故正是这样一直隐形的。警报关心的是
**卡住的工作**，不是缺席的进程。

### 8.2 活性怎么判

`board.store.runtime_liveness`：从 `runtime_status.json` 里读出 pid，然后问
`/proc/<pid>/cmdline` 那个 pid 是不是一个 `harness_runtime`、**且它的 `--session-dir`
就是这个 session**。两半都重要——死 pid 是事故本身，而**被回收的 pid** 在一台同时跑三个
runtime 的机器上会替错的 session 作保。这与 `store._model_identity` 对模型服务用的是同
一道身份闸（§2.3）。心跳年龄在这一层**只报告、不裁决**：poll 循环在一个 brief 跑着的
时候不心跳，所以只有能同时看见 `processing/` 的 `health()` 才把年龄变成结论。

### 8.3 失败模式表（投递 → 认领）

`✅ loud` = 操作员或 agent 被指名告知，不需要问第二个问题。

| # | 失败 | 在哪 | 现在会怎样 |
|---|---|---|---|
| 1 | **runtime 死了**，brief 排着队 | `harness_runtime` 没了，`inbox/` 没人动 | ✅ `brief_status` → `state: stalled`，`runtime.alive: false` 带原因；`health()` 指名那个 session 和条数 |
| 2 | **runtime 死时手里攥着 brief**（`processing/` 孤儿） | `_process` 被打断 | ✅ `state: stalled`，`stalled_from: running` |
| 3 | **同一张 brief 反复把 runtime 打死**（segfault / OOM-kill —— `_process` 的 try/except 接不住） | `boot()` 重排队 | ✅ `_MAX_REQUEUES = 2`（共 3 次尝试）之后进 `failed/` 并写一行点明次数的 `runtime.task_error` → 由 `brief_status.outcome` 浮出 |
| 4 | **runtime 活着但 poll 循环卡死** | `main()` 循环 | ✅ `health()` 用 `processing/` 打破平局：**空闲时**心跳陈旧 = 卡死，并直说 |
| 5 | **模式不对**：`campaign`/`rsi` brief 投进执行态 session | `_process` | ✅ `ValueError` → `failed/` + `runtime.task_error` → `brief_status.outcome`；投之前可以先看 `health().sessions[].mode` |
| 6 | **session 目录不存在**，非默认 session | `store.session_inbox` → `safe_child(is_session)` | ✅ 投递前就 `{"error": "unknown session"}` |
| 7 | **session 目录不存在**，默认 session | `store.brief_inbox` 故意绕过 `is_session` 闸（首投可能早于首次 boot），然后 `submit_brief` 做 `inbox.mkdir(parents=True)` | 🟠 部分：拼错的 `--session` 会**造出** `runs/<typo>/inbox/`，brief 在一个没有 runtime 看守的目录里烂掉。`health()` 会把它列成 `DOWN` 且 `queued≥1`，但没有东西拒绝它，见 §8.4 |
| 8 | **inbox 不可写** | `brief_drop.drop` | ✅ `OSError` 直接从工具调用里抛出 |
| 9 | **brief 是坏 JSON** | `_process` | 🟠 直接进 `failed/` 且**没有链行**（刻意：没有可归属的对象）。`brief_status` → `failed`，无 `outcome`——`failed` 是一个终局答案 |
| 10 | **注入的/未知的 brief 键** | `_process` 里的 `_BRIEF_KEYS` | ✅ `failed/` + `runtime.task_error` |
| 11 | **未知任务名** | `_run_task` | ✅ `ValueError` 并列出每一个已知任务 |
| 12 | **任务已知但 session 不对**（robocasa mission → `session-main`） | mount 时，在 runtime 内部 | 🟠 只是建议：`submit_brief` 在能证明目标解释器 import 不了该卡 `third_party` 时返回 `warning`，否则沉默。真正的拒绝是 mount 时的 `ImportError`，在一份没人读的日志里 |
| 13 | **控制台死了，runtime 活着** | node web server | ✅ `health().console.serving` |
| 14 | **模型端点挂了** | llama.cpp :30001 | ✅ 折进 `health().model` |
| 15 | **一个 session 上两个 runtime** | `_claim_session` flock | ✅ 第二个拒绝启动并报出持锁 pid |
| 16 | **最后那次改名到 `done/` 失败** | `_process` 末尾 | ✅ `runtime.task_error` 行；brief 留在 `processing/` 下次 boot 重排队，由第 3 行封顶 |
| 17 | **session 休眠**：runtime 没跑、inbox 也空 | `health()` | ✅ `state: dormant`，不算 problem（`--status` 折成一行 `dormant`）；有 brief 排队才升级为 `stalled` |
| 18 | **模型服务停了** | llama.cpp :30001 | ✅ `health().model.running: false`，`--status` 的 STOPPED 行带启动命令；只有 `PH_WITH_MODEL=1`（`scripts/cockpit --with-model` 会导出）时才计为 problem，免得为省显存停模型的机器常年红 |
| 19 | **campaign 子进程在 stop / boot 时残留** | `_run_watched` / `_requeue` | ✅ 任何退出（cancel、SIGTERM、Ctrl-C、崩溃）都按 `processing/<brief>.pgid` 杀整个进程组（TERM，宽限后 KILL）并落 `CANCELLED`；`cockpit --stop` 中途的 brief 进 `cancelled/`（stage `runtime_stopped`），boot 时发现活着的孤儿组先杀再重排队，写 `runtime.orphan_killed {brief,pgid}` |
| 20 | **evolve 中途停/死** | `_run_evolve` / `evolve.py` | ✅ cancel 标记在轮边界生效（`campaign.json.status: cancelled`，exit 3；轮中走第 19 行的 killpg）→ `runtime.task_cancelled`；已封存轮次的 `rsi_step` 行不重复；同 task 再投 `evolve` 从 `cursor` 续跑。执行态投 evolve → 同第 5 行 |

### 8.4 仍然开着的口子

- **第 7 行**——默认 session 的 inbox 按需创建是**故意的**（首投可能早于首次 boot），
  于是拼错会制造一个幽灵 session。`health()` 让它**可见**（`DOWN` + 积压），但没有东西
  拒绝它。要真正堵上，得由控制台声明它期望的 session 集合，那个清单还不存在。
- **第 12 行**——仍然只是建议。合适的形状是投递前的
  `plugin_doctor --session <name>`：把 `mcp_server._compat_warning` 那个探针抬进
  `board/store.py`（`runtime_liveness` 已经能把 session 解析到活 pid），对 manifest 并集
  里的**每一个**任务跑一遍，打印 doctor 已有的 PASS/SKIP 报告卡。大约 30 行、零新机制。
  **今天没建，因为还没有东西消费"投递前的兼容性答案"**；mount 时的拒绝无论如何都是唯一
  权威——"不合格在 mount 报错"不会因为有个 doctor 提前同意而变软。
- **第 3 行的天花板**——`_MAX_REQUEUES` 是逐 brief 且终身的，所以一次长 brief 期间三次
  **操作员**重启也会把它花掉。那是响亮的（`failed/` + 一行点明次数的链行）且可重投，
  好过一个看不见的启动死循环。真有运行踩到再改成衰减窗口。
- `health()` 用 **TCP connect** 探控制台，不是 HTTP GET：一个卡死的 node 进程后面还
  listening 的 socket 会读成 SERVING。
- **控制台面板还没接上这些。** `stalled` 是 `brief_status.state` 上的新值、`health` 是
  新的 `storecli` fn——两个都已服务，都还没被渲染。在 ph-station 那边接上之前，操作员从
  `scripts/cockpit --status` 读它们，控制台 agent 从 `health` 工具读。

---

## 9. Skill graph protocol

planner 产出的图、runtime 的验收事件、技能库的记录和种子账本共用一套对象，模式全部在
`harness/protocol.py`（stdlib + `sha_json`，不依赖任何卡）。对象都是内容寻址的
（`content_id` = sha256 of canonical JSON），链行是唯一真相，下面每一样都是链行的投影。

### 9.1 五个对象

| 对象 | 是什么 | 代码 |
|---|---|---|
| 状态字典 σ | `key -> 值`；每个键的来源是 `sensed` 或 `privileged`。谓词只在 σ 上求值 | 卡的 `env.py` / `predicates.py` |
| Predicate | `id, name, args, reads(读哪些键), bindings{本体: "module:attr"}, audit{本体: {n,tp,fp,tn,fn,seed_block,store}}`。**三值**：读键缺失 → `None`（未知），绝不伪造 False。审计门 `sens>=th_s ∧ spec>=th_p ∧ eps<=base_rate<=1-eps`，阈值是参数不是常量 | `PredicateRecord` / `Audit.passes`；本体卡 manifest `[[provides]] kind="predicate"` 声明，`harness/predicates.py` 的 `records()` / `evaluate()` / `audit_gate()` |
| SkillRecord | `id, name, kind, class, lineage{parent,round}, args(模式), requires / ensures / clobbers（谓词引用，`clobbers` 是 STRIPS 删除表）, limits, failure_modes, bindings{本体}, evidence{本体}`。symbolic 半边与本体无关，绑定与证据按本体分开。入库规则：`ensures` 非空，且引用的每个谓词对目标本体都有审计记录 | `SkillRecordV0`；`skill-library/records/<name>.json`，`harness/skill_library.py` 加载 |
| ExecutionGraph G | `{mission, seed, tasks[{id, goal[谓词]}], nodes[{id, task, skill, args, after[], on_fail{policy: replan\|recovery\|abort, budget?, rule?}}], rationale, planner{}}` | `ExecutionGraph.from_dict`；`plugins/task/validate.py:plan_to_graph` 把 planner 的 `{goal,nodes,verify}` 形状抬成它 |
| Trajectory τ | `(x, y, o)`：x = {mission, σ₀ 的 sensed 投影, 可见技能 id, show_evidence, done, fault}，y = {graph id, rationale}，o = {legal, 每节点 verify, L, success, replans, seed, block, role∈{dev,heldout}}。`id = hash(x, y)`。**纯投影**，从链行算出来，不另存 | `board.store.trajectories(session)`；storecli / MCP 同名 |
| 种子账本 B | 所有**已封存** prereg 的 gate/heldout 区间之并，再并上 STATUS.md 的已烧行（store 格式之前的历史：phase 1/2 区块、held-out 复评；标定块永不烧）。`alloc(block, role)` 合法 ⇔ block ∩ B = ∅。没有任何 store ⇒ **拒绝**分配 gate/heldout，不是「没烧过」 | `board.store.burned_blocks(runs/)`；`rsi_campaign.allocate` / `harness_runtime._assert_unburned` 消费 |

`class` 是技能的粗分组（小写 token，`[a-z][a-z0-9]*`），也是唯一的可选新字段。推导规则（`protocol.skill_class`）：显式声明优先；否则 `kind` ∈ {verify, decide, perceive} 就取 `kind`；再否则取 `name` 的第一段（`name.split("_", 1)[0]`）。库里 99 条记录都已显式写上。

谓词引用的规范串是 `name(a,b)`（零元 `name()`）；record 里的 `holding(object)` 是模板，
派发时用节点 args 实例化成 `holding(apple)`（`protocol.instantiate`）。

### 9.2 Legal(G)：四条规则，缺一张图就不派发

`validate_graph(G, records, σ₀.facts, σ₀.objects) -> (ok, problems[])` 一次收齐全部问题：

1. **Typed**：每个节点的 args 与 record 的 args 模式逐键匹配（`TYPES`：entity/str/int/float/bool）。
2. **Grounded**：`entity` 类型的实参必须在 `σ₀.objects` 里，或由某个前驱节点产出。
3. **Supported**：节点 n 的每个 `requires` 谓词 p，要么在 σ₀.facts 里，要么在某个祖先 m 的
   `ensures` 里，且不存在威胁 c（p ∈ clobbers(c)，c 既不在 m 之前也不在 n 之后）——**与 n
   不可比的节点算作可能的威胁**。
4. **Covered**：每个 task 的 goal ⊆ 该 task 各节点 `ensures` 之并，且在 task 结束时未被威胁。

不合法的图**从不派发**，但会作为负样本封进链（`task.plan` 行照封，`legal=false`）。
今天 mission 卡的 CATALOGUE 只有 args，所以 Supported/Covered 对它们是空真；卡通过
`brief["records"]` / `brief["facts"]` / `brief["objects"]` 发布真 record 后四条全部生效。

### 9.3 运行期：verify → fault → replan 单调

每个节点跑完，在 σ 上求 `ensures` → 链行 `task.verify {node, results{谓词: true|false|null}}`；
任一非 true → `task.fault {node, failed[], signature?}`。replan 输入 `(G, D=已验收节点, fault, σ)`，
输出 G′ 必须满足 **D ⊆ nodes(G′) 且每个 done 节点的 (skill, args) 逐字节相同**
（`replan_monotone`），再以当前 facts 为 σ₀ 重新过 `Legal(G′)`。违者封为
`task.replan_rejected {replan, problems}`，折回成一次 `invalid_plan` fault，计入 `max_replans`，
不派发。超过 R 次 → 不可恢复；**L = 第一次不可恢复失败前已验收的节点数**，进 τ.o。

进展规则（`protocol.replan_progress`）：对同一 (node, fault 签名)，同一 `graph_sha`（同 args、同 executor）的图只允许**一次原样重跑**；
再次给出同图 → `task.replan_rejected {reason:"no_progress"}`，折回成 `no_progress` fault（携带原签名、graph_sha 与「同图已试过，改 args/executor 或加 recovery 节点」提示）交回 planner；
planner 仍给同图 → 任务以该 fault 诚实终止。`{"kind":"task"}` 缺省 `max_actuations` 取 task binding 的声明值（mission 卡按节点数声明），无声明才是 3。
「加 recovery 节点」= `protocol.insert_recovery(plan, node, strategy)`：在失败节点前插一个 `kind:"recovery"` 节点（`skill` 是本体卡 `[recoveries.*]` 的策略名，无 args，不进 catalogue）；
workload 经 episode driver 的 `make_recovery` 缝在持久世界上跑完 actor，封 `task.verify{node:"recover-<id>"}` 与节点 `diagnostics.strategy`，失败段随后在修复后的世界重跑。mission 表 planner（recycle_cans）对 `no_progress` fault 就这样答。

### 9.4 卡怎么接入

- 本体卡 manifest 加 `[[provides]]`（`kind ∈ {embodiment, predicate, recovery, skill, planner}`，
  `ref = "module:attr"`；谓词必须带 `reads`）。`discover()` 折进 `Registry.provides`，
  形状错在 discover 就 `ValueError`。
- 技能加一份 `skill-library/records/<name>.json`；不能执行的本体写 `implemented: false`，
  planner 看不见它。
- 不要在卡里再写一份 catalogue/segment 表：`select(RECORDS, 本体, names)` +
  `catalogue_of` / `planner_docs` / `segment_specs` / `skill_specs` 就是那张表。

### 9.5 σ₀、VLM 投影、轨迹导出

- σ₀ 怎么到 validate：卡 manifest 的 task binding 带 `records = ...:SKILL_RECORDS` 与
  `initial_facts = ...:INITIAL_FACTS`；`task_brief` 穿到 brief，`workload.run()` 用
  `_records` / `_sigma0`（episode obs + 声明 facts + 已注册谓词求值）算出 facts/objects，
  一并传给 `validate_graph`（Supported/Covered 非空判），封在 `task.plan {facts, objects, visible}`。
- VLM：`harness.protocol.vlm_projection(records, facts, objects, done, fault)` 生成 prompt 卡片
  + `VLM_OUTPUT_SCHEMA`；`planner_vlm` 只从这份投影建 prompt，`task.plan.planner.prompt_sha =
  content_id(messages)`。端点经 registry ref 到达，无 GPU 用 `plugins.model_endpoint:fake_provider`
  （`PH_MODEL_ENDPOINT_FAKE=<graph.json>`）。
- 导出：`python -m board.storecli trajectories <session> --out DIR` 写 `dev.jsonl` / `heldout.jsonl`，
  按 seed 落在 `burned_blocks` 的 role 分（无 store → 全 dev，`o.role_source` 记来源）。
  e2e：`tests/test_mission_e2e.py`（G1–G4，真 runtime + 真 CLI）、`tests/test_mission_sim_e2e.py`（G5，robosuite）。

测试：`tests/test_protocol.py`（Legal 四条 + 单调 + 三值 + 稳定 id）、`tests/test_predicates.py`、
`tests/test_static_skill_library.py`、`tests/test_task_protocol_events.py`、
`tests/test_trajectories.py`、`tests/test_store.py::*burned_blocks*`。

### 9.6 Benchmark suites and arms

- brief：`{"kind":"suite","suite":"robocasa_v0","arm":"scripted"|"pi05","seeds":[lo,hi],"max_replans"?,"max_actuations"?}`。
  runtime 把卡上每个 (task, seed) 当普通 task brief 走 `_run_task`（无第二个执行器），
  封 `<session>/suites/<sha>.json`：`{suite, arm, seeds, per_task:{n,k,L_mean,first_death}, prereg_sha, checkpoint_sha?}` + 链行 `suite.sealed`。
- 卡：`plugins/benchmark_robocasa/manifest.toml` `[benchmarks.robocasa_v0]` 纯数据（tasks / arms / max_replans / max_actuations），`Registry.benchmarks` 折叠。
- 烧块：第一集之前先封 prereg（`blocks.heldout=[lo,hi]`，链行 `runtime.suite_preregistered`），
  `burned_blocks` 立刻看见；与已烧块重叠的 suite 在任何一集之前就被 `_assert_unburned` 拒绝。
- 读：`board.store.suite_result(session, sha=None)` = `storecli suite_result <session> [--sha]` = MCP `suite_result`，三面字节相同。
- arm：记录 binding 的 `policies.{scripted,pi05}`；`segment_specs(..., arm)` 按 arm 解析 provider。
  handover 规则：某 skill 没有 pi05 binding ⇒ pi05 arm 回落到 scripted 阶段驱动；走 provider 的段在
  `task.verify` 带 `driver:{ref:"plugins.policy_vla_remote:provider", handshake{checkpoint_sha}}`，suite 工件顺带 `checkpoint_sha`。
- pi0.5 服务：`scripts/cockpit --with-policy`（或 `.env` `PH_WITH_POLICY=1`，`PH_POLICY_CHECKPOINT` 换 checkpoint）经
  `board.store.policy_server` 起 :8000，`health().policy` 行只在该 flag 下计 problem。
  e2e：`tests/test_suite_e2e.py`（S1/S2）、`tests/test_suite_robocasa_e2e.py`（S3，robocasa）、`tests/test_suite_pi05_e2e.py`（S4，robocasa+vla，测试自起自停服务）。
- tunables：robocasa 段驱动的常量（`hover_dz`、`reach_tol`、`standoff`、`segment_cap`、`stall_k`）是卡 manifest `[tunables]` 的数据，
  `drivers.tunables()` 读默认值并按进程用 `PH_TUNABLES='{"stall_k":20}'` 覆盖（未知键拒绝）；`drivers.tunables_sha()` 随每个 robocasa 段的
  `diagnostics.tunables_sha` 封入节点/`task.plan_complete.nodes`/`actuation_end`，同处 `diagnostics.failure_mode`（`"reach_stall"` = eef 到目标距离 K=stall_k 步不降，段提前失败）。
  `[tunables]` 经 `manifest.mount_params` 到达该卡每个 driver provider（`params["tunables"]`，evolve 的 `PH_MOUNT_PARAMS_OVERRIDE` 同路并入，`drivers.mount_tunables` 一处共享读）；任一 nav 段（载物/空载，含 NavToObjectDriver）底盘到 dock 距离 K 步不降且离 dock > 到达带（载物 CARRY_NEAR，空载 NAV_POS_TOL）→ `failure_mode "nav_stall"`。pack_lunch / kitchen_thaw planner 同 recycle_cans：对 `no_progress` 按首死节点的阶段词插入本体声明的修复（nav/carry → `redock_retry`，grasp → `regrasp_kitchen`，pack/place → `reapproach`；载物段修复保持夹持）。

### 9.7 Plan library and mission briefs

- PlanRecord（与 SkillRecord 同一 publish 门进 skills_root，文件名 = digest）：
  `{"kind":"plan","id":graph_sha,"task","goal":[pred refs],"graph":<task.plan 的 graph 字典>,"embodiment","arm",
   "evidence":{n,k,L_mean,seed_blocks,sessions},"rule":{theta,n_min,lower},"published_from":[chain refs]}`。
  发布规则：Jeffreys 95% 下界 `lower(k/n) >= theta` 且 `n >= n_min`；theta/n_min 是显式参数（默认 0.8 / 10）并写进 `rule`，不是暗常数。
  `validate_capability` 对 kind=plan 走 `validate_plan`（id == graph_sha、rule.lower 复算、带 records 时跑 validate_graph）。
- 门：`scripts/publish_plans.py --mode evolution --runs ... --skills-root ...`（execution 模式 rc 3 拒绝）；
  候选来自 `board.store.plan_index(session)`（= storecli `plan_index` = MCP `plan_index`，三面字节相同）。执行模式永不写 skills_root。
- Library planner：`plugins/planner_library` 包一个内层 planner ref；(task, embodiment, arm) 命中挂载的 PlanRecord 就取最高 `rule.lower` 的 graph，
  `planner={"provider":"library","plan_id"}`；未命中原样交给内层。`_mount_plan` 对 task brief 也挂它。
- Mission brief：`{"kind":"mission","mission":<自然语言>,"seed","arm"?,"max_replans"?,"max_actuations"?}`。两层：
  (1) 分解：model endpoint 吃 `mission_projection`（已知 task / 谓词目录 / objects / MISSION_DECOMPOSE_SCHEMA）→ `{tasks:[{id,task?,goal}],rationale}`，
  谓词必须在目录里、命名 task 必须已知，封 `mission.decomposed{tasks,prompt_sha}`；拒绝封 `mission.refused`，不派发。
  (2) 每个 task 先 library planner，未命中走 VLM planner；合成一张 ExecutionGraph（goal 非空，Covered 生效）→ validate_graph → 普通 workload 路径。
- 链：每条 `task.plan` 带 `graph_sha`（去掉 planner/rationale 的 content_id）和 `planner:{provider:"library"|<ref>,plan_id?,prompt_sha?}`；合成图只封一条 task.plan，逐任务来源在 `planner.tasks`。
- e2e：`tests/test_plan_records.py`、`tests/test_mission_plan.py`（单元）、`tests/test_plan_library_e2e.py`、`tests/test_mission_decompose_e2e.py`（FakeEndpoint JSON 列表按序应答）、`tests/test_mission_robocasa_e2e.py`（robocasa）。

### 9.8 Executor choice

- 节点可带 `executor: <record.bindings.<本体>.policies 的键>`（今天 scripted|pi05）；省略则由 brief `arm` 定默认。arm `auto` = planner 逐节点自选，record 没有该 policy 时只在**未显式**写 executor 时回落 scripted。
- Legal 第五条 **Bound**：显式 `executor` 必须是 record 该本体 policies 的键（plain binding 只认 scripted），问题前缀 `bound:`，派发前拒绝，绝不在集中途。
- 投影：`vlm_projection` 卡片带 `executors:{key:{evidence:[lo,hi]|null, checkpoint_sha?}}`，`VLM_OUTPUT_SCHEMA` 收 `node.executor`；区间只来自 `record.evidence[本体].by_executor[key]{n,k,seed_blocks,store}`（`evidence_interval`），缺就 `null`，不编造。
- 链：每条 `task.verify`（及 segment 结果）在 `driver.ref` 旁封 `executor: <实际用的键>`；`board.store.skill_evidence(session)` 投影成 per (skill, embodiment, executor) `{n,k}` = storecli `skill_evidence` = MCP `skill_evidence`，三面字节相同。
- e2e：`tests/test_arm_routing.py`、`tests/test_protocol.py::test_bound`、`tests/test_skill_evidence.py`、`tests/test_executor_choice_e2e.py`（真 runtime，无 GPU）、`tests/test_executor_pi05_e2e.py`（robocasa+vla，自起自停 :8000）。

### 9.9 Executor 契约

- `harness/skill_executor.py`：`StepExecutor`（handshake/reset/act/done/diagnostics，harness 逐步 act）与 `SegmentExecutor`（handshake + `run(spec, deadline_s)->{ok, diagnostics}`，执行器自己跑完整个子目标）；`is_segment(x)` = 有 run 无 act。
- `normalize_handshake(transport, ref, meta)` 是 `task.verify.driver.handshake` 唯一封存形状：`{transport: inproc|ssp|mcp, ref, checkpoint_sha|None, unverified:[...], ok, meta}`；未知 transport 挂载即报错。scripted / skill_geometric_grasp 走 inproc（`InprocExecutor` 基类），pi05 走 ssp（reconcile 结果经此归一化）。
- record `bindings[emb].policies[key]` 显式带 `transport`（缺省 inproc）；`skill_library.rearm(spec, arm, executor)` 返回 `{key, transport, ref, params, checkpoint_sha, spec}`，workload 按 ref 挂 provider、每节点 `make_driver`。
- 段执行器接入点（`plugins/task/workload.py::_segment`）：rearm 后若 `is_segment(executor)`，不走 driver.act 循环，调 `run({skill,args,sigma}, SEGMENT_DEADLINE_S)`；stage driver 仍 `enter_segment`，其 `segment_success` 与图上的 verify 谓词照常判定——执行器的 ok 只是主张，验证永不外包；ok=false → 该节点 fault → replan。
- 首个 MCP 段执行器卡 `plugins/executor_mcp_segment/`（provides executor `mcp_segment`, transport mcp）：`provider(command=[...])` 起子进程，stdio 上按行 JSON-RPC 2.0（MCP stdio）：`initialize` → `notifications/initialized` → `tools/call run_segment {skill,args,sigma,deadline_s}`；handshake = initialize 的 serverInfo。零新依赖（~40 行客户端）。
- e2e：`tests/test_skill_executor.py`；`tests/test_executor_mcp_e2e.py`（真 runtime，假服务 `tests/fakes/mcp_segment_service.py` 记录调用到 `$PH_FAKE_MCP_LOG`、`args.fail` → ok=false）：handshake.transport=="mcp"、服务收到节点 spec、ok=false 产生 task.fault 与 replan。

### 9.10 技能图：class、依赖、benchmark

三条派生关系全部在 `harness/protocol.py` 上算，只读记录，不依赖任何卡：

- `skill_class(rec)` → 上面 9.1 的推导规则，一条技能一个 class。
- `skill_dependencies(records)` → `(src, dst, rule)`：`causal` = `src.requires` 里某个**带参**谓词与 `dst.ensures` 逐位相等（ground 实例 `at(can1)`←`at(can1)`，或泛型同名变量 `at(obj)`←`at(obj)`；自环排除）；零元谓词（`gripper_free()`、`water_on()`）是资源，不产生依赖；`uses` = plan 记录图上每个节点的 `(plan.id, node.skill)`，去重保序。
- `skill_instances(records)` → `[(instance, generic)]`：同 class 且 `instance.name == generic.name + "_" + 后缀`，最长的 generic 名胜出。
- `skill_benchmarks(records, cards, mission_cards)` → `{技能: [benchmark]}`：卡的 `embodiment` 在记录 `bindings` 里，且（卡没写 `tasks`，或某条 plan 记录的 task 在 `tasks` 里且用了该技能，或 benchmark 覆盖的某张 mission 卡在 `skills` 里列了它）。benchmark 卡的 `embodiment` / `tasks` 从 `plugins/benchmark_*/manifest.toml` 的 `[benchmarks.<name>]` 读，当数据看。
- `benchmark_coverage(cards, mission_cards)` → `[(benchmark, 卡目录, task)]`：benchmark 的 task 是某张 mission 卡 `[task_bindings.<task>]` 的键，且同一 embodiment（binding 的 `env` ref 指向的 `embodiment_<x>` 卡）；`mission_uses(mission_cards)` → `[(卡目录, 技能)]`，读 binding 的 `skills` 行。
- mission 卡的 `[task_bindings.<task>]` 多一行 `skills = [...]`（排序后的记录名，与规划器 `SKILL_RECORDS` 的键一一相等，`tests/test_mission_card_skills.py` 钉死）——board/ 只读这一行，永不 import `SKILL_RECORDS`。

`board/vault.py:build_graph` 把静态技能库和这三条关系折进**同一张图**——控制台看到的技能图就是这一张，没有第二处：

- 节点：`skill:<name>`（库记录，status `library`）、`class:<c>`、`benchmark:<name>`，与既有的 `package` / `capability` 并列（5 类）。
- 边：`IN_CLASS`、`DEPENDS_ON`（rule `requires∩ensures` / `plan uses`）、`INSTANCE_OF`（rule `name prefix within class`，via 实例记录路径；generic 节点带 `instances: n`）、`BOUND_TO`（bindings 的 `plugins.<card>:attr` → 卡目录）、`EVIDENCED_ON`（有证据时额外带 `n` / `k`）、`COVERS`（benchmark → mission 卡，rule `benchmark tasks ∩ task_bindings`，via 两份 manifest；benchmark 节点带 `missions: [卡目录]`）、`USES`（mission 卡 → `skill:<name>`，rule `manifest task_bindings.skills`，via 该卡 manifest；mission 卡的 package 节点带 `tasks: [...]` / `skills: n`），与既有血缘边并列（16 种关系）。
- 记录当数据读（json → `SkillRecordV0.from_dict`，读不动的文件跳过），`board/` 依旧零插件导入。

**能力、卡片、技能的边界**（图里三层，各答一个问题）：

| 层 | 是什么 | 答什么 | 来源 |
|---|---|---|---|
| 能力 `capability` | 内核认识的接口插槽，固定 10 个（`embodiment.env`、`policy.driver`、`task.planner`、`model.endpoint`…） | 内核需要什么 | `harness.definitions.CAPABILITIES` |
| 卡片 `package` | 一个插件目录 + manifest，安装单位；声明它提供哪些能力，承载执行器、谓词、恢复原语、benchmark | 装了什么、谁来实现 | `plugins/*/manifest.toml` |
| 技能 `skill` | 一条 SkillRecord：符号契约 + 对某张卡上执行器的绑定 + 证据；规划器拼图的单位 | 会做什么、靠谁做、做得多好 | `skill-library/records` |

跨层只有三种边：卡片 **PROVIDES** 能力、卡片 **MOUNTED_IN** 本体卡、技能 **BOUND_TO** 卡片；技能之间是 `DEPENDS_ON` / `INSTANCE_OF` / `IN_CLASS`。旧重链路晋级出的 `runs/*/skills` 记录（candidate / promoted / retired）及其血缘边是历史，控制台默认隐藏。

---

## 10. 没在这份文档里的东西

**开发史与设计资本不随仓库发布。** 它们在 git 历史里，和操作员机器上的
`local-archive/docs/`（git-ignored）里。两份被本文取代、只保留在本地归档
`local-archive/docs/retired-from-public/` 的：

- `ph-station-design.md` —— 当初为什么 fork dsh、rebrand 要改哪 7 处上游源码和行号、
  数据面三个候选方案里为什么否掉另外两个、面板保留/隐藏清单、npx→fork 迁移的 parity
  检查表与退役计划、v1 之后的 roadmap。**当前仍然为真的部分已进入本文 §1.4 / §2 / §8。**
- `base-gate.md` —— 逐次增量的流水账（"The +N over the previous snapshot is…"：每一次
  测试数字变化对应哪个测试文件、钉住了什么）。**当前快照、隔离复现方法和纪律已进入
  本文 §3。**

开发用的活页文档（当前进度、卡在哪、下一步）在 `docs-dev/`，也是 git-ignored，不随仓库
发布。
