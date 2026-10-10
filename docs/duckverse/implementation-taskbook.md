> Planning document supplied by the project owner. Proposed capabilities are not delivery claims. Current implementation: [stage board](03-implementation-board.md).

# ROSClaw × Microduck｜Duckverse 第一季工程实施任务书

**项目代号**：`DUCK-GAME-01`
**正式作品**：**Last Duck Standing｜最后一块地板**
**执行对象**：本地 Codex（开发、实验、验证、视频制作）
**日期**：2026-10-08
**性质**：明确研发目标与验收规则；本文列出的新功能**均为待实施**，不是已有成果。

> **总任务**：在已有 `ros-claw/microduck` 仓库中，建设一个**四只真实 MuJoCo 双足机器人同场竞技、地板逐块物理塌陷、角色自主移动、发生真实接触与淘汰、能以自然语言开启比赛、能真实记录/评测/回放并生成精彩视频**的第一部 Duckverse 作品。优先完成“可玩的真实物理游戏”，再做 AI、镜头和包装。禁止用直接改写机器人根节点、临时隐藏支撑地板、预设赢家、动画假动作来获得视觉结果。

---

## 0. 执行纪律——先读这段

1. 本项目**继续使用既有仓库** `https://github.com/ros-claw/microduck`，不要另起一个影子游戏工程。保留现有跳绳与 Neon Escape 基准和视频，新增独立 `arena/` 模块，复用已有 `sim/composer.py` 的多鸭装配方法与 `sim/runtime.py` 的 PolicyBank/DuckRuntime。**不直接把 Neon Escape 的单鸭连续地板 `game/world.py` 改成竞技场**。
2. 参考并锁定：
   - `https://github.com/ros-claw/microduck`（当前开发仓）
   - `https://github.com/ros-claw/rosclaw`（Agent、SIMULATION、Practice、Darwin、Team、Skill 的事实来源）
   - `https://github.com/ros-claw/e-urdf-zoo`（Body 描述；确认是否已有 Microduck 条目，没有就做最小合法链接，不重做 Schema）
   - `https://github.com/pollen-robotics/microduck_rl`（MJCF/官方步行、恢复、翻滚、轮滑训练基线）
   - 当前 `upstream.lock.yaml`（必须锁定 commit、policy digest、观测/动作契约；注意上游新版 `robot_allcollisions.xml` 与旧锁定版本可能不是同一碰撞范围）。
3. **不先做一个庞大游戏平台**。这次只做一款完整作品，输出可复用模块；新模块是否泛化由第二部游戏再检验。不要先开发通用地图编辑器、账号系统、全量 Web Studio、通用多 Agent 群聊、强化学习训练平台。
4. 每个 PR 必须提交：`目的与影响范围`、`新增文件`、`实测证据`、`失败记录`、`命令`、`测试结果`、`样例视频或截图`、`限制与下一步`。测试不得使用假响应伪装为真实 Jev/ROSClaw Live 调用。
5. 遇到障碍时先做独立复现实验，保持失败记录，不许把目标悄悄降级成动画。高风险目标失败时标记 BLOCKED，给出最小复现、物理原因和保守替代方案；继续交付已通过的里程碑。
6. 所有物理执行目标限定 **SIMULATION**。不能把 `SIMULATED` 证据描述为真机部署。运动控制中不得让大模型/Jev直接写 14 个关节位置。
7. **作品与测试要分开**：游戏娱乐模式允许掉落/碰撞/失败，但物理评估仍保持严谨；用于视频的成功片段必须可由原始状态与真实决策重放。导演可以选择优秀运行，不能修改其物理结果。

---

## 1. 成品长什么样——先把观众看到的内容定下来

**一句话剧情**：四只颜色不同的 Microduck 被困在悬空竞技场，地板不断闪红、断裂、下坠，它们必须自己决定跑向哪里、什么时候跳、什么时候停，最后只剩一只鸭子。

**开场 3 秒**：高空竞技场俯视镜头，Lavender、Cream、Sky、Graphite 四只鸭子相对而立；地板灯光由蓝变黄、变红；巨大倒计时 `3 → 2 → 1`；脚下第一片地板轰然坠落。

**中段 15–35 秒**：四鸭分头逃向安全地板，几次相遇产生拥挤、让路、碰撞或抢位；切换跟拍与顶视镜头，观众始终看得清“下一片红色地板为什么危险”。动作必须来自 ONNX 策略、关节动力学和真实接触。

**尾段**：场上剩两只；仅余一条狭窄生路；一只失败跌落下方可见接收平台，另一只真的站稳并获胜；获胜者触发已有/新训练的庆祝动作。**禁止预先指定哪只鸭子赢**。若出现双双落下，准确判 `DRAW`，不得补拍假冠军。

**首版最终交付**：
- 游戏：`Microduck Last Duck Standing` 可重复运行，含至少四种关卡布局和两种塌陷节奏；同 seed、同模型与控制记录可复现。
- 成片：16:9 主片 45–65 秒；9:16 短片 15–25 秒；2–4 分钟技术幕后；一张 B站封面；中英文标题和发布说明。
- 技术：Demo 可以独立 CLI 运行；**已接通且实测的** ROSClaw chat SIMULATION 开赛与回执；按结果逐项标注 Jev、Training/Darwin 是否实际参与。
- 源码：场景、脚本、测试、基线、产物 manifest、README/README.zh-CN 持续维护。

> **产品原则**：观众第一眼就明白目标，观众看得见危险发生，技术人员查得到动力学证据。不要让大段 HUD、思维链、API 延迟数字占掉游戏主体。

---

## 2. 现状审计与升级边界（DG-00 必做）

### 2.1 已有可复用基础
- `src/microduck_lab/sim/composer.py`：多鸭命名前缀、共享物理世界、`DuckSpec`、多机器人模型装配。
- `src/microduck_lab/sim/runtime.py`：`PolicyBank`、`DuckRuntime`、官方 61D observation/14D action、50Hz policy、力矩限制接口。
- `src/microduck_lab/game/`：Neon Escape 的场景、种子、世界状态、动作选择、分数和 Jev 异步接口；可以**借鉴代码范式**，不强行复用单鸭假设。
- `docs/NEON_ESCAPE.md`：真实 Jev 请求/重放的历史证据和延迟实测；不是新比赛的可靠性数据。
- `src/microduck_lab/video/` 与 `scripts/render_neon_escape.py`：离线重放和摄像机、HUD 基础。
- `policies/`、`upstream.lock.yaml`：已有策略与锁定资产。

### 2.2 先探测再声明的事情
- 四鸭同场时各自 `joint_qpos_idx`、actuator IDs、传感器、camera/site 的命名前缀是否全部独立。
- 每只 Duck 初始化是否真站稳；默认站立、行走、转向、恢复、Jump、Roulade 是否可调用且适合多地板世界。不要只依赖旧文档声称已支持。
- Microduck Body 在 e-URDF-Zoo 的实际注册状态与 `rosclaw body` 的有效上下文；有无成熟 SIMULATION capability 接线。**没有则列出缺口，不能在本项目中伪造成功。**
- 当前 MuJoCo 版本、`MjSpec` 约束编译/运行时能力、动力学表现和 CPU/显卡实时倍率。
- 若新版官方模型的碰撞模型与锁定版本差异大，创建独立对照，不悄悄替换旧版。

### 2.3 DG-00 交付
新增：`docs/duckverse/00-audit.md`、`docs/duckverse/01-architecture.md`、`docs/duckverse/02-physics-contract.md`、`docs/duckverse/03-implementation-board.md`。
包含 upstream SHA/asset hash、当前架构连接图、接口缺口、物理运行倍率、命令、风险矩阵。此 PR 不改已有物理行为。

---

## 3. 游戏规则冻结：不做“漂亮但无规则”的电影

### 3.1 游戏模式

先实现 `SURVIVAL`：每回合 2–4 只鸭子，全部从各自出生地起步，地板提前预警、按规则失去支撑。最后保持有效支撑的鸭子获胜；如果同时消失则 DRAW；超过时间上限按“剩余有效存活状态→已约定的 tie-breaker”判定或 DRAW，不能临时裁定赢家。

第一版**不需要**其他鸭主动推搡；自然发生的身体碰撞可以保留并记录。后续可加 `CHAOS`（允许更高风险/互相影响）和 `TEAM`（合作救援），但不得挡住 DG-01 交付。

### 3.2 地板生命周期

```text
LOCKED (有物理约束、可支撑)
  → WARNING (仅警示灯/边缘变色，物理约束仍在)
  → RELEASED (真正切断约束，格子成为自由落体/可运动刚体)
  → FALLING (由实际 z、vz、姿态确认已开始下坠)
  → LOST (不再构成比赛有效支撑)
```

不要通过视觉隐藏、关闭渲染、改 `geom.rgba`、把地板瞬移到地下、强制清除鸭子 contact 来实现塌陷。地板先有真实的锁定结构，释放后让重力决定下落与碰撞。地板下方可以设置**公开可见**的低位接收平台，避免无界坠落；它只负责接住落败者，不再是竞赛地板。

`WARN→RELEASE` 倒计时必须在物理世界里可观察（灯光、图案、音效），Agent 只能读取已公开的倒计时/警示，**不能读取未来随机时间表或内部 seed**。

### 3.3 正式淘汰条件（以物理证据为准）

至少同时检查：
- 鸭子质心/躯干已经越过竞技场有效支撑高度向下；
- 接触图中不再有 **有效竞技地板** 对其提供稳定支撑，且在连续一小段时间窗内不能通过正常恢复重新上场；
- 非因摄像机裁切、单帧脚部离地、普通跳跃顶点或对手遮挡而误判；
- 淘汰证据包含 `body_id, t_sim, z, vz, supporting_tile_ids, contact_refs, video_frame_ref`。

工程起始建议：有效支撑高度线低于竞技场表面 **8–15 cm**，再加约 **0.2–0.4 s** 消抖；必须通过跳跃/被碰后腾空测试校准，不把这些数字当作物理常数。真正掉进接收区后算 ELIMINATED，不应被隐形 reset 送回场上。

### 3.4 地板调度的公平性

关卡 seed 只负责初始布局和塌陷随机计划，不负责指定哪只鸭子赢。训练/调试/评测必须以随机初始角色位置打乱，轮换出生点，对同一策略做公平比较。

**每次 release 前必须有 WARNING**；面向新手的节奏应给鸭子足够时间从危险块移动到邻块。预警时间和距离通过实测的 `walk_to` 时间分布校准，建议最初 1–3 秒再逐步调整。绝不能根据视频希望谁赢而暗中修改 release schedule。

设计一个**关卡可行性检查器**：以可走邻接图及实际时间/速度余量，检查至少存在一条从各出生点向安全区域撤离的可行路径；这只能作为保守近似，不可冒充实际物理可达性证明。需要在种子基准上测可玩性。

---

## 4. 场景和物理：TileArena（DG-01 / DG-02）

### 4.1 尺寸从 Body 数据标定，而不是凭空定死

起始建议：5×5 个方形格子；单格边长约 `0.42–0.52m`，格缝约 `0.01–0.02m`；整体约 `2.2–2.7m`。**这是首次灰盒实验区间，不是最终尺寸。** Codex 必须通过 Microduck 真实站立投影/双脚外廓、最小转弯半径、步行到邻格耗时、两鸭对向避障试验调整。太窄会出现“所有鸭子都被物理限制死”，太宽则没有紧张感。

按顺序推进：
1. 3×3 空场 1 鸭，站立、移动与越缝。
2. 4×4 空场 2 鸭，身体和地面真实接触。
3. 5×5 空场 4 鸭，全部角色站稳并移动。
4. 少量地板预警/下落；确认无硬碰撞穿透后再增加调度密度。

### 4.2 推荐实现：Free Body + 初始 Weld Support

一个格子由：`tile body + freejoint + box collision geom + weld equality to world/static fixture + warning-only decorative geoms` 构成。编译时记录 `tile_id → body/joint/eq/geom indices`。运行时只做 `data.eq_active[eq_id] = 0` 释放约束。**释放不等于直接改写 tile qpos/qvel。** 释放后真实受重力下落，旁边的鸭子如果踩在格边则会受到平台运动影响。

MuJoCo 官方允许运行时切换 equality 的 `mjData.eq_active`；以当前实际版本做最小复现，检查 weld 的初始相对姿态是否正确，防止初始化弹跳。参考：https://mujoco.readthedocs.io/en/stable/computation/ 与 https://mujoco.readthedocs.io/en/stable/modeling/ 。

如果 25 个 freejoint + weld 在 4 鸭世界性能或求解器中不稳定，可使用**带 hinge 的可释放翻板/机械支撑**，但必须依旧靠重力/力矩发生真实运动；记录与原方案的物理差别，不偷换成 kinematic 动画。

### 4.3 接触模型和高可信度检测

- `tile/duck`、`duck/duck`、`tile/tile` 真实接触；只为必要的装配自碰撞剔除配置显式 exclude。警示灯等视觉 geoms 非碰撞。
- 地板 collision geom 使用 box/convex primitive，视觉形状可做倒角、六角风格，但**不能在视觉上形成比物理 support 大得多的“隐形支撑”**；先统一方形物理格子，后面再讨论六角形。
- 支撑判断基于脚底 geom / 地板 geom 的接触及法向，校验足底接触是否真的在 tile 上；不要仅用 `duck.z > threshold` 判定存活。
- 测：数值穿透量分布、碰撞力/冲量、能量跃迁、solver warnings、地板释放后重力加速度、跌落时间；当 4 鸭 + 多格共存时严防数值爆炸。
- 不默认持续 5kHz 物理以追求“高保真”。先测 `0.005 / 0.002 / 0.001 / 0.0005 s`（必要时更细）的稳定性、穿透和 wall-real-time-factor；保持 50Hz policy，确保子步数是整数或严格同步。选满足证据要求的**最经济稳定配置**。不能把低采样率当成“无穿透”。

**建议物理质量门槛**（可按实际几何比例校准并记录）：经关键场景压测后的接触穿透 P99 ≤1.5 mm、绝对最大 ≤3 mm、无明显可见整身穿模、无 NaN/solver divergence。达不到不要在宣传片中宣称“零穿透”；必要时修 collision proxy、solref/solimp、稳定性和建模。

### 4.4 “比赛坑底”

保留可见深坑（从俯视镜头能看出高度）；下面用可见接收网或宽平台做淘汰区。入坑不算继续比赛；落到接收网时应有真实接触与落地声。摄像机可以跟拍，不可在同一物理运行里瞬移落败者到观众席。必须节制视觉效果：不要用爆炸火焰掩盖接触穿模。

---

## 5. 多机器人运行：同一物理世界里四只真正的 Duck（DG-03）

### 5.1 重用上游，而不复制一套控制器

复用 `MultiDuckComposer` 思路，但建议把“通用鸭装配”与“绳子专用装配”拆开，形成 `DuckWorldAssembler`；加载同一官方 MJCF 四次，自动加 `lavender/`, `cream/`, `sky/`, `graphite/` 前缀。每只都维护独立：Policy state、last_action、command、关节/传感器索引、runtime identity、Skill state、Practice lineage。

禁止四个仿真世界各自跑后期拼成一张图。必须是 `1 MjModel + 1 MjData + N DuckRuntime + N controllers + 1 arena`。测试四鸭相遇时真的发生 body-body 接触；不得有同名 actuator/sensor。

### 5.2 五层频率解耦

- MuJoCo：多子步物理积分，实际 dt 按实验确定；
- ONNX motor policy：官方兼容 50 Hz；
- 局部 Skill：约 10–50 Hz（站稳、跟踪目标格、避障、停止、恢复）；
- 战术决策：约 0.5–2 Hz，依据实际状态；
- ROSClaw Native Agent：只在开局/新任务/失败分析/必要时变更策略，不直接控制腿。

每只 Duck 的战术动作只允许：`GO_TO_TILE(tile_id)`、`WAIT/HOLD`、`YIELD`、`RECOVER`、`CHOOSE_NEXT_SAFE_TILE`，后续 `JUMP_TO_TILE` 需经独立验证才启用。底层使用受限 walk/turn/stand policy，按可达距离、可见 WARNING 和身体状态实现。**初版地板缝很小，允许不依赖 Jump 完成基本换格**。

### 5.3 角色风格≠输赢脚本

建议配置 `Lavender=bold`, `Cream=careful`, `Sky=curious`, `Graphite=analytical`，仅影响“同等安全时如何选择目标、距离边缘的余量、是否等待”等偏好，不改物理属性、不预置冠军。发布素材中角色人格属于剧情设定，不能冒充通过 RL 自动学出来的性格。

---

## 6. 可观察的战术智能：不要让 AI 读出未来地图（DG-04）

定义 `ArenaObservationV1`（每只 Duck 一份，版本化 schema）：

```yaml
schema: microduck.arena_observation.v1
sim_time_s: 12.80
robot:
  body_id: lavender
  current_tile_id: tile_12
  pose_estimate: {x: 0.25, y: 0.45, yaw: 0.12}
  speed_m_s: 0.21
  upright: true
  current_skill: walk_to_tile
  skill_ready: {walk: true, recover: false, jump: false}
visible_tiles:
  - {id: tile_11, state: LOCKED, risk: low, reachable: true}
  - {id: tile_13, state: WARNING, warning_age_s: 0.45, reachable: true}
  - {id: tile_17, state: FALLING, reachable: false}
nearby_ducks:
  - {id: cream, range_m: 0.41, bearing_rad: 0.8, moving: true}
objective: survive
```

数据分域：
- `Perception/State`：当前可见格子的状态、鸭子里程/姿态/接触、短期运动历史、公开音光预警，可传给 Jev/agent；
- `Oracle/Evaluator`：全部未来释放时刻、所有种子、所有身体全真值、完整接触审计和胜负事实，只在生成器、训练/评估的可信端使用；
- `Render`：第三人称、顶视角、色彩与粒子表现，不能反向注入 AI。

做架构测试：Agent/Jev 的请求 payload **不得出现** `future_tile_schedule`、`seed`、`oracle_tile_ttl`，不因计算了预测风险而把预测误写成未来真值。预测可以用最近观测推出，但标记 `predicted` 和时间戳。

---

## 7. 先有聪明的确定性基线，再试 Jev（DG-05）

### 7.1 基线必备

最初实现 `DeterministicSurvivor`：根据相邻格状态/可达时间、警告和其他鸭子占用情况选择下一个格子；提供可解释的状态机与局部避障。最低保证“鸭子尽可能找安全格子”，避免四只只会站着原地掉下去。

`tile_cost` 可从以下组成：`可达距离/时间 + WARNING 风险 + 陷落风险 + 拥挤风险 + 需要剧烈转身惩罚`；硬约束先过滤不安全/不可用格，软评分再排序。需要区分 `no_safe_option` 时的真实失败，不要虚构穿墙路线。

### 7.2 Jev 的正确位置

Jev 如果接入，**只负责战术格子选择**，不负责一帧一帧抬腿。候选为当前合法相邻格的 `tile_id` 加 `WAIT`（有限集合），由确定性硬条件过滤，再交 Jev typed-choice 选择，最终由 50Hz Skill 走到目标格。

旧 Neon Escape 曾记录约 0.7 秒请求中位延迟，仅是**历史数据**，本项目启动时重新测 p50/p95/p99，设 `deadline`、异步 single-flight 和 fallback，不把 Jev 标注成 10Hz 伺服控制。四鸭并行调用时必须限制并发，且**让仿真时钟/网络延迟在实时模式下对应**；离线快速仿真与在线真实 Jev 延迟实验须分开。

决策成功后还需对“答复已经过时、该格已释放、路径被其他鸭子占据、机体不稳”重新做 legality check；低置信/超时走基线或 HOLD，不能盲目执行。

### 7.3 公平比较

三种 brain：`baseline`、`jev`、`recorded`。相同 seed、spawn 顺序和动作空间做 paired comparison；轮换四鸭所绑定的策略，防止颜色和出生位置对结果形成系统偏差。看 `survival_time`、`win_rate`、`tile transitions`、`falls`、`stale decisions`、`p95 decision latency`、`token/episode`。没有足够样本时不宣称 Jev 最强。

**首部宣传视频不必以 Jev 为核心卖点**。即便 Jev 的战术收益未被证实，游戏本身也必须好看且可玩；技术版真实披露 Jev 与 baseline 的关系。

---

## 8. ROSClaw 真贯通：用户一句话开赛（DG-06）

最终用户在 `rosclaw chat` 输入：

> “让 Lavender、Cream、Sky、Graphite 来一场最后一块地板，地板越掉越快；不要指定胜者。赛后告诉我谁赢了、怎么赢的，并生成回放。”

Native Agent 负责理解任务、查询 Body/Skill/允许的 game options，创建 SIMULATION Episode，等待完成、读取可信 receipt、回报结果。**不在模型对话里实现 GameLoop，也不让模型对关节发写指令。**

建议 Microduck Lab 对外提供语义 API（名称是**需要实现的接口设计**，不是已存在 ROSClaw 命令）：
- `arena.create_session(config)`：只接受经 JSON Schema/Pydantic 校验的参数；
- `arena.start(session_id)`：创建/启动仿真 episode；
- `arena.observe(session_id, robot_id?)`：只读当前状态；
- `arena.status(session_id)`：只读进度；
- `arena.stop(session_id, reason)`：受控停止；
- `arena.replay(run_id)`：重放；
- `arena.report(run_id)`：包含比赛结果、失败原因、证据摘要。

遵守 ROSClaw 当前 `ToolDescriptorV2` 等契约：`PHYSICAL_ACTION` 不能直接变成 model-callable 工具。先审查 agentd/capability app/Simulation executor 允许的实际路径，在适当的 **SIMULATION action / App / ActionEnvelope** 边界上构建 Adapter。不得通过把写动作伪装成 OBSERVE 让模型绕过安全系统。若需要 Generic Capability 缺口，向 ROSClaw core 提最小 issue/PR，与游戏专用代码分离。

每个 Duck 是一个 `Body Instance`，可以共享 Microduck Physical DNA，但身份、运行状态、Practice 与能力版本必须不同。ROSClaw integration 在 `SIMULATION` 模式下生成真实 Mission / task / game run / observation / terminal receipt 关联，trace_id 在整个链路可查。

**验收**：真实 `rosclaw chat` 启动一次 2 鸭局与一次 4 鸭局，等待自然结束，回执报告与 MuJoCo 真值一致；断线/取消/重复 start/idempotency/超时场景有测试。遇到本地模型/agentd 不可达应报明确失败，不得用模拟问答假称已贯通。

---

## 9. Practice / 失败分析 / 进化：先证据，后宣传（DG-07）

每次比赛写入：
- `run_id, task_id, mission_id, trace_id, seed, config_hash, source_commit, upstream_hash, mujoco_version, body_instances, policy_sha256`；
- `tile_event_log`: WARNING、RELEASE、FALLING、LOST 的发生时间和因果；
- `duck_event_log`: SkillIntent、SkillStart、SkillEnd、身体支持变化、Falls、Recovery、Eliminated；
- `decision_log`: 每个 brain 的候选、输入状态 hash、选择、延迟/置信度、过期/拒绝原因；
- `physics_audit`: 每个关键接触对、穿透/冲量摘要、跌落前后状态、胜者判定证据；
- `replay_manifest`: 状态轨迹、逐帧 hash、视频时间轴及剪辑镜头描述。

数据分层：结构化事件进入 ROSClaw Practice/相关存储（在接口可用时）；大规模 qpos/qvel/control/视频文件保存在本地对象工件，事件只记录引用和 digest。不造第二套 ROSClaw Memory/Darwin 数据库。

**Replay 陷阱**：只保存 `qpos/qvel/ctrl` 不够。格子 release 状态在 `data.eq_active`，若要严格重演还需相应 tile 事件、约束活动状态、外部施加力/actuation、真实决策及状态版本。录制与重放都验证 tile support、Duck pose、winner、接触摘要；重放失败不能生成正式 Hero。

第二阶段（不阻断主视频）：对 `DeterministicSurvivor` 的风险参数、预测距离、tie-break、路径余量做候选搜索。先 16–32 个训练/开发 seed，冻结后再抽本地隔离 holdout（例如至少 32 个新 seed）做 Darwin 对照，要求生存表现改善且越界/穿模/未授权动作不退化；经真实 Promotion Gate 和允许的批准流程后才更新 Champion。**没有真实参数/策略更新、无真实 Darwin 接线时，不允许宣称“鸭子自主进化了”。**

---

## 10. 游戏视觉与电影化（DG-08）

**艺术方向**：延续 Neon Escape 的深色霓虹城市，但竞技场以高对比区域色为主：安全蓝/青、警告黄、释放红、已落下黑色空洞。每只鸭子固定原生色 + 头顶小名字/清晰编号，不增加会影响本体动力学的重量级饰品。

镜头：
1. `EstablishWide`：俯视/斜俯视完整 4 鸭与大部分格子；
2. `DuckChase`：关注某只鸭子与它前方即将塌陷的格子；
3. `ThreatShot`：格子预警和释放的短特写；
4. `NearMiss`：脚踩边沿、相邻格下落时的慢动作；
5. `Elimination`：鸭子真实掉落的完整动作；
6. `FinalDuel`：两鸭 + 最后几格同框；
7. `WinnerShot`：冠军在真实平台上庆祝（仅已验证动作）。

**导演只读**，只能控制 camera/lights/HUD/music，不碰机器人根位置/关节/格子约束。镜头切换基于事件时间线，拒绝 `if t in [20,21] show winner X` 这种强硬编码。必须同时产出**一镜到底参考回放**（非宣传精剪）和**Hero Cut**，方便核对。

开场 3 秒必须一眼看懂目标：`LAST DUCK STANDING / 地板不断掉落 / 只剩一个赢家`。HUD 极简：剩余鸭数、场上安全格数、倒计时、冠军名字。音效可后期制作但要标记为 foley；不把音效当成真实传感器。

首版建议剪辑：0–4s 开赛；4–13s 地板塌落；13–25s 四鸭混战；25–39s 两鸭决赛与险情；39–48s 冠军/庆祝；48–53s ROSClaw Logo + 下期预告。时长随真实事件节奏调整，不强行对齐。宣传片不能混剪多个不同 seed 伪装成**同一连续回合**；确需 montage 要有明确转场/字幕。

**发布包**：`hero_16x9.mp4`、`short_9x16.mp4`、`technical.mp4`、`hero_cover.png`、`replay_reference.mp4`、`subtitle_zh.srt`、`subtitle_en.srt`、`video_manifest.json`、`bilibili_copy.md`。专门做 1280×720 封面版本并核对移动端可读性。

---

## 11. 建议目录布局（只扩充有价值的模块）

```text
ros-claw/microduck/
├── docs/duckverse/
│   ├── 00-audit.md
│   ├── 01-architecture.md
│   ├── 02-physics-contract.md
│   ├── 03-implementation-board.md
│   ├── LAST_DUCK_STANDING.md
│   └── game-content-roadmap.md
├── configs/duckverse/
│   ├── last_duck_standing_greybox.yaml
│   ├── last_duck_standing_four_ducks.yaml
│   └── characters.yaml
├── src/microduck_lab/arena/
│   ├── __init__.py
│   ├── schema.py              # config/event/observation/result typed contracts
│   ├── world.py               # TileArena world builder, no Neon Escape mutation
│   ├── tiles.py               # body+freejoint+support equality, state machine
│   ├── assembler.py           # multi-duck composition based on existing composer
│   ├── observation.py         # actor-visible state, no oracle leakage
│   ├── candidate.py           # legal tile/skill enumeration
│   ├── controller.py          # local closed-loop walk/stop/recover
│   ├── brains.py              # baseline + Jev(optional) + recorded
│   ├── scheduler.py           # seed-based, fair release plan and warning
│   ├── referee.py             # winner/draw/elimination/contact audit
│   ├── episode.py             # single truth: simulation loop and events
│   ├── recorder.py            # physics + decision + replay manifest
│   └── rosclaw_adapter.py     # SIMULATION API integration; no duplicate agent
├── src/microduck_lab/video/
│   └── duckverse_director.py  # event-driven cinematic shots
├── scripts/
│   ├── run_last_duck_standing.py
│   ├── eval_last_duck_standing.py
│   ├── replay_last_duck_standing.py
│   └── render_last_duck_standing.py
├── tests/arena/
│   ├── test_tiles.py
│   ├── test_contacts.py
│   ├── test_multi_duck.py
│   ├── test_scheduler.py
│   ├── test_observation_boundaries.py
│   ├── test_controller.py
│   ├── test_referee.py
│   ├── test_replay.py
│   ├── test_jev_optional.py
│   └── test_rosclaw_integration.py
└── artifacts/duckverse/           # 大轨迹/片源 .gitignore；小型可复核报告可提交
```

所有新脚本路径为**实施目标**，不是宣称当前已有。`rosclaw_adapter.py` 是否位于本仓库按 ROSClaw 真正接口审查决定，但必须保持外置的 Microduck-specific 边界。

---

## 12. 逐 PR 工程实施计划（每个 PR 要能单独审）

| PR | 目标 | 必交结果 / 进入下一步条件 |
|---|---|---|
| **DG-00** | Repo/Body/ROSClaw/License 审计、架构冻结 | 四份设计审计文档、环境与依赖 hash、已验证能力矩阵；不改旧行为 |
| **DG-01** | 真实 Tile 物理 + 1 Duck | 3×3 地板：锁定可支撑，释放后真掉落，接触正确；一个 Duck 能步行换格；无根节点移动 |
| **DG-02** | Tile schedule + Replay/Referee | WARNING→RELEASE→FALLING→LOST；倒计时、真实淘汰、无 ghost support；同 seed 同输入重演一致 |
| **DG-03** | 2→4 Duck shared world | 4 只单一 MuJoCo 世界、各自独立关节/策略、真实互碰、记录合法；性能 profile |
| **DG-04** | 可玩的确定性生存 Agent | `go_to_tile` 状态机、合法候选、路线/避碰、失败分类；不是所有鸭子原地站着 |
| **DG-05** | 新地图/节奏、人物设定、可玩性 | 4 个 seed/layout 原型、不同危险节奏、随机出生公平检查、技术版预览 |
| **DG-06** | ROSClaw Native Agent SIMULATION 真接通 | `rosclaw chat` 真开赛、获得带引用的 terminal receipt、回放；失败模式可复核 |
| **DG-07** | Practice/录制/评测 + 可选 Jev | 一条真实 lineage；brain 基线对照；如 Jev 真调用显示延迟/概率/效果；无虚假自进化 |
| **DG-08** | Cinematic Director + 成片 | 16:9 / 9:16 / 技术片 / 封面 / 中英素材，逐镜头与真源对应 |
| **DG-09** | 冻结发行 + 32 个新种子 QA（建议） | 新种子质量报告、碰撞/成功/淘汰频率、公平性、已知限制；README 更新，版本发布 |
| **DG-10（后续）** | 真 Darwin/学习进化研究 | 仅在有可测退化/提升与可用 ROSClaw Promotion 接口时启动；不阻碍视频发布 |

**交付顺序不许颠倒**：DG-01 没过，不做视觉包装；DG-03 没过，不写四鸭已能竞赛；DG-04 没过，不说 AI 在自主生存；DG-06 没过，不说“ROSClaw chat 驱动”；DG-07 没过，不展示学习/进化的性能提升。

---

## 13. 质量门槛与测试矩阵

### 13.1 必须自动测的物理不变量

1. tile 锁定时负载可支持、无持续下沉；释放后正常下坠，且 `eq_active` 真变化。
2. 没有任何隐藏的平面 geom 在塌陷后支撑鸭子；验证真实地板/格子接触面。
3. 每只 Duck 在初始化外禁止被修改 floating-base qpos/qvel 来产生游戏动作；初始化定位必须标记。
4. 各鸭 joint/sensor/actuator ID 独立，四鸭同时跑不串流、不覆盖上一个动作。
5. 警告→释放事件与状态机严格单调；tile 不会无故恢复。需要重置只能开启**新 episode**。
6. 普通跳跃/前滚翻/短暂腾空不可误淘汰；落坑后不会误判仍存活。
7. Duck–Tile、Duck–Duck、Tile–Tile 的接触对正确；有真实冲量和有限穿透，无 NaN、无 solver warnings。
8. 同一世界种子 + 同一完整决策与约束事件，在相同版本环境中回放得到相同赢家与足够接近的物理轨迹。
9. agent 输入不包含未来 release schedule、seed、evaluator 私有信息。
10. 渲染机位/HUD/慢动作变化不改变运动轨迹或结果。

### 13.2 分阶段 Gate（建议阈值，按实测修订并公布）

| Gate | 最低要求 |
|---|---|
| **G1 物理** | 1 鸭 + 3×3 方格：10 次释放/坠落回归无 solver failure、无假地板支撑；留有逐帧证据 |
| **G2 两鸭** | 至少 10 个真实比赛 seed：能正常开局/塌陷/决胜或平局，记录全部结果；不容许系统崩溃 |
| **G3 四鸭** | 至少 12 个真实四鸭 seed，稳定初始化、碰撞、运行、淘汰；无名字/控制串线 |
| **G4 可玩** | 至少 3 个不同 seed 的录像可看懂“危险→响应→后果”；胜者不是预设；记录精彩事件频率 |
| **G5 Performance** | 实测 4 Duck 实时倍率、物理开销/渲染开销/政策推理耗时；达不到 1× 则离线可靠录制并报告，不伪称实时 |
| **G6 Agent** | 至少一次 live `rosclaw chat` 开赛到真实 SIMULATION receipt；任何不可用路径应显式 BLOCKED |
| **G7 Replay** | 录制决策、tile 约束状态与真实物理接触均可复核；结果不依赖视频后期 |
| **G8 发布** | 主片/短片/技术片 + 仓库发行说明同步；逐镜头确认没有重大可见穿模或物理身份错误 |

`win_rate` 与 Jev 增益只在足够 seed、出生点轮换且对照公平时估计，至少给出分子/分母和不确定性，不把一条选中的 Hero Clip 当统计证据。

### 13.3 要有恶意/极端测试

- 全地图同一时刻 WARNING：调度器应拒绝不可玩的配置或明示 Hard Mode。
- 两鸭占据同一格，格子瞬时 release：Referee 必须只按物理事实裁决。
- Jev 超时/返回不可用 tile：fallback 生效，不使 physics 暂停或执行非法动作。
- Agent 命令请求给某鸭瞬移/直接改电机：接口必须拒绝。
- 某策略文件缺失/观测 shape 不匹配：该 Body 降级/禁用，不能静默换成装饰性动画。
- 录像依赖的 hash 或 eq_active 事件丢失：严格重放失败并输出诊断。

---

## 14. 命令面（以下是“需要实现”的建议 CLI，不是现有命令）

```bash
# 用现有 lock 安装依赖与策略（务必核对真实命令）
python scripts/bootstrap.py

# 单 Duck 物理灰盒
python scripts/run_last_duck_standing.py --ducks 1 --grid 3 --brain baseline --seed 11 --headless

# 两 Duck 完整基础赛
python scripts/run_last_duck_standing.py --ducks 2 --grid 4 --brain baseline --seed 42 --record

# 四 Duck 稳定赛
python scripts/run_last_duck_standing.py --ducks 4 --grid 5 --brain baseline --seed 1024 --record

# Jev 实测：需要真实密钥、真实网络、显式 realtime
python scripts/run_last_duck_standing.py --ducks 4 --grid 5 --brain jev --seed 1024 --realtime --record

# 多种子比较与质量报告
python scripts/eval_last_duck_standing.py --config configs/duckverse/last_duck_standing_four_ducks.yaml --seeds-file artifacts/private/unseen_seeds.json --out artifacts/duckverse/eval

# 严格复播（对齐赢家、约束事件、关键 contact）
python scripts/replay_last_duck_standing.py --run artifacts/duckverse/runs/run_xxx --strict

# 生成真实录制的视频（非新仿真）
python scripts/render_last_duck_standing.py --run artifacts/duckverse/runs/run_xxx --cut hero --aspect 16:9
```

实际脚本参数和依赖路径可根据工程合理调整，但**说明文档和输出脚本必须能够复制运行**。ROSClaw chat 的步骤按已存在接口提供专门 `docs/duckverse/LAST_DUCK_STANDING.md`，不可使用凭空发明的 ROSClaw CLI 参数。

---

## 15. Duckverse 的可复用资产与第二部作品路线

不在第一部做大平台，但将下列协议做成小型可复用资产：

- `CharacterCard`：鸭子身份、颜色、角色性格、已有能力、policy digest；区别叙事设定与真实学习数据。
- `PropCard`：碰撞 geom / 运动关节 / 激活/释放动作 / 安全范围 / 录制事件。本期先做 `CollapsibleTile`，下期可有 `MovingGate`、`Boulder`、`BottleSwitch`。
- `EpisodeCard`：一部游戏的目标、角色、场景 seed、成功/淘汰规则、视频高光事件。
- `ManeuverSkill`：具身本体使用的动作契约和 success/failure 证据。
- `VideoEvent`：真实物理事件与剧集镜头的桥梁。

内容储备（暂不实施）：
1. `Duck Arena: Last Duck Standing`——多鸭物理生存；
2. `Neon Skate`——优先验证官方 Roller 变体与训练/部署配置；
3. `Duck Bowling`——踢球、撞瓶、连锁机关；
4. `Duck Delivery`——协同搬运，需要真实抓持与负载测试；
5. `Duck Trouble`——生活场景捣蛋与物体交互。

**版权提示**：官方 `microduck_rl` 把 3D 模型另行列为 CC BY-SA-NC，项目代码许可证并不会自动覆盖机器人 3D 模型与商标；素材投放/商业合作前须核查具体授权范围并保留 Pollen Robotics attribution。不要重授权上游美术。

---

## 16. 对 Codex 的最终执行要求（每轮汇报固定格式）

每完成一项 PR，统一汇报：

```text
PR / commit:
实施目的:
改动文件:
复用的上游资产与来源 commit:
场景 seed / policy hashes / MuJoCo version:
实际运行命令:
真实结果（成功、失败、物理证据、录像）:
测试数量（passed/failed/skipped）:
画面审查结论（是不是好看、看得懂）:
已知局限与退路:
下一 PR Go/No-Go:
```

**第一轮只执行 `DG-00 + DG-01`**，做到一只 Duck 可以在 3×3 真正会下坠的地板上移动；提交一个 10～20 秒灰盒录像和逐物理步接触证据。我确认物理路线有效后，再推进多鸭与电影级画面。若无人在线审查，Codex 也应按 Gate 自动执行可安全推进的下一步，但绝不跳过实际物理验证。

**最终北极星**：别人看视频时先觉得“可爱、疯狂、好玩”；看 GitHub 时发现“这个结果来自真 MuJoCo 多刚体接触、官方运动策略、可审计 ROSClaw Agent 和能复现的真实运行”。这是内容 IP 与技术价值同时成立的条件。
