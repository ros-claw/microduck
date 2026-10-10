# Strike & Escape / 连锁撞瓶逃生

This experimental example turns the chase into six physical encounters:
**roll under → dodge a dropped crate → jump a void → pass a rotating arm →
push a ball and topple three targets → roll again and escape**. A physical
strike unlocks the exit. The pursuer can knock the alley guides loose and then
strike the closed door. Videos show simulation time, with three slow-motion
segments per cut.

这次增加的是能改变关卡状态的玩法：鸭子推动球，球和球瓶发生连锁撞击，三个目标
实际倒下以后，出口电机才开门。巨球继续追赶，撞开护栏，鸭子还需要完成第二次
翻滚与冲门。组合计数来自六个关卡验证结果，不是按时间刷分或新增了六个神经网络。

## Control and interaction / 控制与交互

Jev selects encounter maneuvers, including `PUSH_BALL` and the second
`ROLL_CENTER`. An explicit task state machine determines encounter order.
Local model-based previews use the current joint state, velocity and previous
motor action to compare transitions. The default first roll retains a moving, foot-phase-gated entry. The final
roll compares moving and bounded brake/settle entries; the gap also tries a
larger set of approach/settle sequences. An optional first-roll preview was
found to regress in a separate frozen cohort and remains experimental. Previewed self-contact is checked at every
0.2 ms step. Execution continues through the existing learned motor policies;
predicted poses are never copied into the running robot.

Jev 负责选择下一关动作；局部执行器根据当前状态比较衔接方案。推球复用行走策略，
球的运动来自鸭子接触、重力和地面接触。行走、站立、翻滚和长跳四套 ONNX 权重保持
不变。这是已知关卡、仿真完整状态下的模型规划；尚未实现视觉感知或通用游戏智能。

| Element | Physical implementation |
|---|---|
| Duck | Native model/materials; unchanged learned motor weights and current limits |
| Void | Real 15 cm floor gap with no hidden support |
| Bowling ball | Free body, radius 8.5 cm, mass 25 g; no motor or applied drive force |
| Three inline targets | Free 8 g bodies; high-friction base contacts prevent sliding from substituting for toppling |
| Alley guides | Physical 1.04 m guides with weld holds; a measured boss-guide impulse above 0.02 N·s releases each hold at the next control tick |
| Exit | Contact-triggered electronic lock drives a slide gate through a motor limited to ±6 N |
| Pursuer | Rolling 0.8 kg sphere, bounded forward/lateral drive forces |

The alley intentionally guides the ball; this is a toy physical puzzle, not
an unrestricted precision bowling benchmark. Guides, pins, their floor
contacts and all duck contacts remain in the contact audit.

保龄球段有真实护栏辅助，因此不能包装成无辅助的精准投球能力。护栏释放和开门都是
接触触发的控制逻辑，随后由动力学求解运动；不是通过改写球或鸭子的轨迹实现。

## Causal verification / 因果验证

A target qualifies only after a force-bearing contact chain from the duck to
the ball and then to that target, possibly relayed through a previously struck
target. Duck–ball impulse must exceed 0.001 N·s; each causal target impulse must
exceed 0.0005 N·s; its local-up projection must fall below 0.35. A duck/boss hit
on a still-standing target, or a boss hit on the ball before unlock, invalidates
that target/shot. Touching an already-qualified fallen target is recorded by
the whole-run ledger. Unlock requires all three qualified targets.

`scripts/audit_arcade_causality.py` initializes the saved model once and replays
recorded inputs. It independently reconstructs contacts at 5 kHz, verifies that
gate commands follow an already-qualified strike, checks that guide releases
follow measured boss impacts, and verifies zero applied external force on the
ball and targets. Recorded-input replay and full-contact publication gates
are separate requirements.

独立检查会重算“鸭子 → 球 → 球瓶”的接触链，并逐帧检查开门和护栏释放条件。只撞到
球瓶而没有倒下、巨球代替鸭子完成撞击、或鸭子补撞尚未倒下的目标，都不能算成功。

Intentional ball/target touches carry no HP penalty, but remain fully counted
in penetration/force evidence. Arm, crossbar and pursuer hits still cost HP.
The six encounter count permits an upright arm survival and a completed exit
roll with a brush; it does not label either as collision-free. The first
roll-under and crate dodge require no contact with their corresponding hazard.
The original full-stunt goal retains its knockdown/recovery requirement.

## Captures and evaluation / 记录与评估

Final film, once-only cohorts, full contact limits and wall-clock costs are
listed in `artifacts/neon-escape-v4/README.md`. Earlier failures and controller
snapshots remain available. Distinct seed groups and live API timings are not
a paired before/after experiment and must not be used to claim a causal uplift.

每组新种子只跑一次。旧失败不删除，不与后续版本合并计算成功率。每条记录保存源码
快照，精选视频也单独报告伤害、规划耗时和仿真时间。所有接触类别继续使用相同门槛：
**最大穿透 <1.5 mm，P99 <1 mm，无求解器警告**。规划预览不能代替实际记录的发布验收。

## Reproduce / 复现

```bash
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -m microduck_lab.demos.parkour_rehearsal \
  --policy policies/parkour_long_jump_v2.onnx --brain jev --hard-contacts \
  --predictive --difficulty arcade --seed 16 --seconds 40 --capture \
  --sweeper-mass .4 --sweeper-torque .2 --sweeper-phase .9 \
  --output artifacts/neon-escape-v4/local-new-run
.venv/bin/python scripts/replay_parkour_run.py artifacts/neon-escape-v4/local-new-run
.venv/bin/python scripts/audit_parkour_contacts.py artifacts/neon-escape-v4/local-new-run
.venv/bin/python scripts/audit_arcade_causality.py artifacts/neon-escape-v4/local-new-run
.venv/bin/python scripts/render_parkour.py \
  --source artifacts/neon-escape-v4/local-new-run --goal escape --style industrial \
  --cut hero --output out/local-strike.mp4
```

API credentials stay outside Git. A fresh API run can differ; exact reproduction
uses the published scene/input capture. Rendering rejects captures without a
matching input replay, full-contact pass and independent causal proof. The
`--legacy-handoffs` option retains earlier gap/exit integration behavior;
`--forecast-entry-roll` enables the unsuccessful experimental entry planner. See source
snapshots for the exact historical implementation of each cohort.

Remaining work includes skill-transition reliability, planning latency, a
certified 20 cm / 1.2 s dodge, perception and hardware validation. Foley is
synthesized. Frozen V1 and all earlier releases are preserved.
