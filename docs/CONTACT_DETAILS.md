# Physical gameplay: contact details / 物理闯关细节

The English detail cut follows the complete six-encounter course, then replays
three useful interactions: clearing and landing beyond the gap, brushing the
rotating rod, and pushing a ball into targets to unlock the exit. Cameras move;
robot and obstacle motion come from recorded simulation states. Native robot
materials are retained. The short finish stays at normal speed.

英文细节版保留六段完整闯关，慢动作只放在跨坑落地、旋转杆擦碰、推球撞瓶这三个有
价值的物理交互上。镜头移动和回放不改变鸭子、杆子或道具的轨迹，模型使用资产原色。
终点动作只作为正常速度的简短尾声，不是项目的主要能力或视频标题。

## What happened at the rod? / 旋转杆是否穿模？

The previous V5 seed-402 film did **not** contain a rod–duck collision. Exact
5 kHz input replay finds a minimum capsule-to-body-envelope separation of
**17.123 mm**. The duck took a side route; perspective can make the paths look
crossed. That film cannot be used as evidence that the rod struck the robot.

The new film uses the previously recorded seed **704**, which includes a real
force-bearing brush against `duck/trunk_base/hazard_proxy`. Four 0.2 ms force
samples produce **0.04294 N s** normal impulse, with **116.55 N** peak force.
The minimum geometric separation across the encounter is **+0.164 mm**:
contact starts within the **0.2 mm soft-contact margin**, before overlap. No
negative rod/body separation was observed in this encounter. This is a brief
brush, not a staged knockdown or a demonstration of recovery after a hard hit.

上一版实际是从侧面避开，最近间隙 17.123 毫米，没有杆子击中鸭子的物理证据。新版
采用已记录的 704 号闯关，存在真实接触力：4 个物理步的法向冲量合计 0.04294 N·s，
峰值 116.55 N。最近几何间隙仍为正的 0.164 毫米，因为 0.2 毫米接触余量会在几何
重叠之前产生力；这一段没有检测到杆子与身体碰撞包络的几何穿透。它是短暂擦碰，
不能描述为被重击后恢复站立。

The rod is a colliding, rendered capsule. Each rigid robot part has a massless,
conservative box enclosing its visual/native geometry, enabled against hazards
only. Native geometry still handles ground support and self-contact; original
robot mass and inertia are preserved. These are collision approximations,
**not exact triangle-level visual-mesh collisions**. An articulation regression
checks that the boxes enclose the visual parts. The cyan outline in the contact
freeze-frame shows only the contacted box and is a render-only annotation.

旋转杆的可见胶囊也是碰撞几何；鸭子身体各刚性部件使用包住可见形状的保守碰撞盒，
仅参与障碍物碰撞，不增加质量，不替代原生地面支撑或自身碰撞。不能把这种近似描述
成逐三角面精确碰撞。特写中青色线框显示真实参与接触的包络，线框本身不参与物理。

The contact freeze-frame uses a separately captured **5 kHz state**, not a
nearby 200 Hz movie frame. MuJoCo's force/geometry outputs describe the state
at the start of an integration step; the audit records that convention and
stores the matching pre-step qpos/qvel. Recomputing forward dynamics for that
state reproduces the **116.55 N** contact force. The dense capture and proof
are hashed into the movie manifest.

## Continuous motion training / 连续动作训练

The old short hop/stand switches and direct head gestures were physically
simulated but looked twitchy in human review. They are superseded here by one
fine-tuned PPO policy for the entire rhythmic finish. Continuous 1.15 Hz head
and body pose commands, alternating foot-lift rewards, balance, measured
position/heading feedback and an increasing action-rate penalty encourage
coordinated stepping and sway. The actor retains the shared **61-dimensional**
observation and **50 Hz** control contract. A separate SmoothPD fine-tune uses
the same target conditioner in training and deployment: previous target plus
`clip(0.25 * (requested - previous), -0.08, 0.08)` at each 20 ms control tick.
The previous-action observation stays raw; no new observation slot is added.
The target-rate bound is an explicit controller constraint, not a learned
achievement. Actual pose motion must independently have <1% spectral energy
above 8 Hz, remain upright and supported, lift both feet and avoid excessive
drift. Observation normalization is baked
into ONNX through the training repository's standard export script.

旧版短跳、站立交替和直接摆头虽然经过物理计算，但观看反馈是抽搐，不能以通过物理
门槛来宣称视觉效果成功。新版训练单一 PPO 策略持续执行有节奏的步态与摆动，头部和
身体目标连续变化，并奖励交替抬脚、平衡、位置保持、降低相邻动作变化。观测仍为
61 维，电机控制 50 Hz，导出 ONNX 包含训练时的观测归一化。平滑版微调在训练、部署两边采用同一个目标
递推：每 20 毫秒向新目标前进 25%，单步变化最多 0.08 弧度。目标限速是明确的控制器
约束，不是假称策略自己学出了这个上限；实际关节运动仍须通过独立的高频能量、抬脚、
站姿、支撑和漂移检查。观测中的上一动作保留网络原始输出，没有增加观测通道。

An initial BAM-actuator candidate failed the position-servo deployment probe:
its RMS target rate was 58.97 rad/s, maximum one-step change 3.502 rad and drift
95.8 mm. It is retained as a failed experiment, not shown as a successful
policy. The replacement uses the repository's existing XML PD-servo family
with the runtime's **±0.6405 N m** current-equivalent limits. Warp training
uses 1 ms physics and zero contact margin because its mesh multi-contact
implementation rejects positive margins; the course retains 0.2 ms physics
and its existing contact margins. CPU deployment checks are therefore required.
This is a simulation policy, not a hardware-qualified emotional behavior.

最初 BAM 执行器候选在位置伺服部署中失败，记录保留；更换为仓库已有的位置伺服模型，
与闯关的电流等效力矩限制一致。训练与部署的物理步长、碰撞余量仍有差别，因此必须
经过实际 CPU 物理部署验证，不能只凭训练奖励判断有效，更不能据此宣称真机迁移成功。

## Evidence and scope / 证据与范围

See [the evidence index](../artifacts/neon-escape-v6/README.md) for selected-policy
results and failed probes. The producer replays historical live-Jev seed 704
exactly through 26.32 s, then continues physics with a new actuator-only finish.
No robot root pose/velocity edits, external robot forces, target teleports or
cross-trajectory pose splices are used. This is a selected development
continuation, **not a new live API run or additional unseen-seed trial**. Earlier
frozen-cohort scores remain unchanged.

关卡前缀是已有 Jev 记录的精确输入重放，后缀从同一动态状态继续仿真。没有改写鸭子
根部位置和速度、施加鸭子外力或拼接不同轨迹。新尾声不是新的在线 API 测试或未见
种子测试，不能增加此前成功率。规划读取特权仿真状态，不是摄像头视觉感知。

Publication requires exact actuator-input replay, full-contact inspection at
every 0.2 ms step and independent contact-causality checks. Full-contact gates
remain max penetration <1.5 mm and P99 <1 mm in every category, with zero solver
warnings. Small compliant overlaps in other contacts are reported separately;
the claim of no rod/body geometric overlap applies to this rod encounter only.
Audio is synthesized Foley, not a simulated microphone.

## Reproduce / 复现

Extract the release evidence to `capture/`; no API key is needed for replay.
Training source and commands are in [training/README.md](../training/README.md).

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/continue_parkour_victory.py \
  --source capture \
  --branch-time 26.32 --joy-policy policies/continuous_joy.onnx \
  --output artifacts/neon-escape-v6/local-new
.venv/bin/python scripts/replay_parkour_run.py capture
.venv/bin/python scripts/audit_parkour_contacts.py capture
.venv/bin/python scripts/audit_parkour_actuation.py capture
.venv/bin/python scripts/audit_arcade_causality.py capture
.venv/bin/python scripts/audit_sweeper_clearance.py capture --impact-time 15.769
.venv/bin/python scripts/verify_sweeper_frame.py capture
.venv/bin/python scripts/render_parkour.py --source capture --goal escape \
  --style industrial --cut technical --blur 1 --output out/local-details.mp4
```

The release capture itself contains the exact historical prefix, so it can be used
as the continuation source at 26.32 s. Using the archived policy reproduces the
same physical trajectory; metadata records the new parent identity. Ordinary input
replay and rendering require no Jev or retraining. Tuning probes are explicitly
distinguished from reliability tests.
