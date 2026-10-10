# Jev 鸭鸭生存战

[English](README.md) · [验证与限制](VALIDATION.md) · [原始检查摘要](../../artifacts/jev-rumble)

张三、二呆、老六、卷王在同一个 MuJoCo 世界中争夺移动目标、避让危险地板并争取成为唯一幸存者。此版真正在线调用 Jev 选择战术，独立于此前的[本地规则版](../survival-rumble/README.zh-CN.md)，不覆盖已确认的跳绳、跑酷或历史复现代码。

[![Jev 鸭鸭生存战开发版](preview.jpg)](media/jev-survival-zh.mp4)

[中文解说、特写与慢动作 · 110.88 秒](media/jev-survival-zh.mp4)。同一局连续记录，末尾仅老六真实存活；用户已于 2026-10-10 确认发布。[高清影片与完整回放数据](https://github.com/ros-claw/microduck/releases/tag/jev-survival-2026-10-10)。

## 决策与运动

每两秒提交一次公开局势，把四鸭的状态、当前姿态、对手位置速度、已公开的地板倒计时与可达动作交给 Jev。开场一次 HTTP 请求包含四个有类型的选择问题，各自说明控制哪只鸭、性格是什么；淘汰后只查询仍存活的角色。它是**一次共享批量请求**，不能称为四个独立语言模型智能体。

Jev 可以选择抢点、预判截路、避让、退守中心或短暂停留。可选目的地由当前地图的可达性产生；本地寻路器执行选中的目的地，不把它偷偷换回固定抢点规则。返回动作必须属于原候选集，概率与置信度必须有效；过期、已不可达的指令拒绝执行。置信度如实记录，不把尚未校准的数值阈值当作安全证书或无故拒绝合法动作。没有有效指令时停止追逐，而不是把本地规则冒充成 Jev。

`jev-requests.json` 保留实际请求、原始返回、模型版本、用量、网络耗时、提交和交付的仿真时间、接受结果。50 Hz 决策日志引用实际执行的请求编号。API 密钥只从环境变量 `TYPESAFE_API_KEY` 或本机 `~/.config/microduck/typesafe.key` 读取，不进入证据。

底层复用上游 ONNX 行走和站立策略：61 维观测、14 维关节目标，50 Hz 控制；物理步长 0.25 毫秒，鸭子电机限 ±0.6405 Nm。Jev 不直接输出关节角，也没有在本次工作中训练出新的基础运动能力。

## 倒地恢复

旧控制器缺少独立恢复状态，摔倒时仍可能执行行走和转向。本版在躯干朝上余弦低于 0.82 时立即切换零速度站立策略，暂停战术追逐；只有朝上余弦超过 0.94、有实际脚底支撑、水平速度低于 0.16 m/s，持续至少 0.18 秒，才继续执行战术。恢复入口朝上余弦低于 0.3 时先尝试站立 0.8 秒；倾倒过程中被检测到的鸭子先保留 2 秒自然起身机会，避免过早切技能打断正在改善的动作。未站稳时再切换已有 `alpha_sitstand` 坐站策略 0.8 秒，随后回到站立；每次倒地只插入一次，切换时机不跟随瞬时速度抖动。恢复完全通过限力矩关节电机完成，不抬起根部、不传送、不关闭碰撞。

在与比赛相同的原生碰撞及头身包围盒条件下，四种初始化倒地姿态、每种三个扰动，共 12 个配对样本：持续给 0.35 m/s 行走和 1.2 rad/s 转向指令为 **0/12** 在 3.2 秒内恢复；单纯零速度站立为 **11/12**；最终恢复技能组合为 **12/12**。另 12 个校准扰动样本也为 **12/12**，这 24 个平躺初始化测试合计中位站稳时间 **1.77 秒**、最大 **3.14 秒**，中位水平漂移 **0.108 米**。详见[最终控制器校准](../../artifacts/jev-rumble/final-recovery-calibration.json)、[基础对照](../../artifacts/jev-rumble/recovery-calibration.json)。控制器冻结后，再用新种子 20、21、22 做 12 个留出样本，**12/12** 通过，最长 **2.82 秒**，之后未再调参。这些是人工初始化姿态测试，不能推广为任意移动碰撞都能恢复，也不是整局胜率对照。

失败过程也保留：翻滚插入只有 5/12 恢复；按瞬时姿态/速度触发坐站策略使切换时机漂移，24 个样本只通过 19 个，故未采用。旧在线版本还出现过约 18 秒的卡住恢复，不能拿精选片段掩盖。三个真实录制倒地状态的受控对照还显示：过早切技能分别耗时 3.28、2.18、4.06 秒，保留自然起身机会为 1.38、1.66、1.76 秒。因此最终控制器根据恢复开始时的姿态区分初段长度。这些对照重放对手控制和公开破碎时间表，并重建上一动作，不是新的在线 Jev 比赛；不能把它们当作在线胜率。最终控制器采用经过校准的恢复阶段；此轮复用已有学得的坐站策略，没有新训练或姿态重置。

**动态恢复尚未彻底解决**：成片对局中五次完成的恢复实际耗时为 4.9、4.8、10.74、3.72、14.46 秒；最长一次期间有 36 段身体接触。不能把初始化测试的通过率说成实战都能三秒起身。完整数据与失败保留在[验证记录](VALIDATION.md)。

## 渐进破碎与公平性

此前终局成批收圈被替换为明确的游戏危险规则：

- 第 6 秒预警第一块，第 10 秒最早脱落；每块至少公开预警 4 秒，同时最多只有一块待脱落。
- 一块脱落后才预警下一块，因此两次脱落至少相隔 4 秒。实际脚底向上支撑力积累疲劳，用于候选优先级；外围优先，中心不脱落。
- 只能移除不会切断剩余地图连通性的块，始终保留通往中心的路径。预警后仍需及时走开；“存在路径”不等于保证每个姿态都来得及逃脱。
- 若仍存活的鸭子实际倒在即将脱落的板上，统一公开延长一次、最多 1.2 秒。恢复宽限不停止物理、不禁止对手接触，也不能永久保地板。
- 若脚下已公开预警且当前 Jev 指令没有有效撤离路线，本地控制器立即向安全相邻板撤离，日志和视频标为 `LOCAL_EVACUATE`，不会冒充 Jev 决策。
- 移动目标只落在仍锁定的地板上；抢点只记统计，不按得分判冠军。只剩三块板后，中心台真实电机以 15 秒渐入轻微摇摆，最大目标倾角 0.08 rad、各限 ±4 Nm。

地板通过解除世界支撑约束开始下落，随后受重力与碰撞控制；不是材料断裂力学模拟。选择待脱落地板不使用角色姓名、得分或预定赢家。真正跌落才由裁判淘汰；唯一存活者在任意仍受支撑的地板上真实承重站稳 0.75 秒即可胜出（窗口承重比例 ≥90%，最长失载间隔 ≤30 ms），不再被迫赶往中心继续冒险。全灭为平局，超时无冠军，不保证每局都恰好有赢家。完整对局预算扩展至 120 秒，让 4 秒逐块预警有时间完成，视频按事件剪辑到两分钟内。

所有接触都纳入最大穿透检查，包括头身碰撞、已淘汰身体、下层接落台。机器人碰撞是原生几何加保守包围盒，并非渲染三角形精确碰撞。没有根部轨迹控制或外力推鸭。

## 运行与核验

先按仓库说明安装资产与依赖。输出目录必须不存在，在线 API 需要自行配置密钥：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_jev_game.py \
  --seed 61204 --capture --out artifacts/jev-rumble/local
PYTHONPATH=src .venv/bin/python scripts/check_jev_rumble.py \
  --run artifacts/jev-rumble/local --out artifacts/jev-rumble/checks-local.json
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_jev_rumble.py \
  --run artifacts/jev-rumble/local --out artifacts/jev-rumble/replay-local.json
```

生成相同风格的视频（先安装 `requirements-commentary.txt` 中的可选配音依赖到 `.tts-venv`）：

```bash
PYTHONPATH=src .venv/bin/python scripts/render_jev_rumble.py --run artifacts/jev-rumble/local \
  --plan-only --out out/jev-local-plan.json
PYTHONPATH=src .venv/bin/python scripts/commentate_jev_rumble.py --run artifacts/jev-rumble/local \
  --plan out/jev-local-plan.json --out out/jev-local-speech.json
PYTHONPATH=src .venv/bin/python scripts/render_jev_rumble.py --run artifacts/jev-rumble/local \
  --speech out/jev-local-speech.json --out out/jev-local-silent.mp4
PYTHONPATH=src .venv/bin/python scripts/sonify_survival_rumble.py --run artifacts/jev-rumble/local \
  --video out/jev-local-silent.mp4 --speech out/jev-local-speech.json --out out/jev-local-zh.mp4
PYTHONPATH=src .venv/bin/python scripts/check_jev_media.py --run artifacts/jev-rumble/local \
  --video out/jev-local-zh.mp4 --out artifacts/jev-rumble/media-local.json
```

在线返回具有非确定性，再运行相同种子不保证相同比赛。闭环核验使用**已记录的网络返回与交付时间**，重新生成请求局势、重新验证答案和合法性，再比较每一步物理状态、控制、接触、战术决策。它不会调用新 API，也不会把回放称为新的在线 Jev 比赛。

录制全量接触可能使仿真慢于实际时间。日志分别记录网络墙钟耗时、响应交付的仿真年龄与整局实时系数，不把仿真中的快速响应误说成网络低延迟。视频仅回放同一局的记录状态，慢动作、姓名、字幕和合成体育解说都不参与控制；成片控制在两分钟内。

完整 MJB、轨迹和全量接触保留在本机采样目录，不纳入 Git。仓库包含审计与网络返回的压缩归档、校验结果、源代码快照及审阅影片；仅 clone 仓库不能重放本机的完整选局数据，可自行采样，或从 [Release](https://github.com/ros-claw/microduck/releases/tag/jev-survival-2026-10-10) 下载 scene.mjb、trajectory.npz、contacts.jsonl.gz 和 replay-metadata.tar.gz，将前三项放入同一个目录，再将元数据包解压到该目录，运行上述核验命令。

已发布：[jev-survival-2026-10-10](https://github.com/ros-claw/microduck/releases/tag/jev-survival-2026-10-10)。此前原始审计与影片元数据保留录制时的开发状态，当前发布状态以此页为准。小样本展示与确定性回放不构成大规模胜率或真机能力证据。
