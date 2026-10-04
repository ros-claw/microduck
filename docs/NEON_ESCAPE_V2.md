# Neon Escape V2 — methods and evidence / 方法与证据

**Experimental. The physical baseline passes replay and contact checks; the full
V2 specification is not complete.** A selected rule-controlled run rolls under
a bar, changes lane, jumps a real 15 cm void, is knocked down by a sweeper,
recovers in **1.08 s**, escapes and leaves the pursuing sphere behind a physical
door. It is not a live-Jev demonstration or a quick-dodge certificate.

**实验版本，尚未完成全部 V2 验收。** 已有规则控制的完整物理基线通过输入回放和
全接触审计：行进翻滚、换道、跨越真实 15 cm 坑、被扫杆击倒、1.08 秒起身、逃生，
巨球由真实门板挡住。不能把这条记录说成在线 Jev 成功，也不能当作快速闪避已达标。

V1 remains unchanged at `a1219f309bbfd4f2edb6ffb3b8ecba0cd3eb8d64`:
[51-file freeze](../artifacts/neon-escape-v2/v1-freeze.json).
V2 lives separately in `microduck_lab.parkour`.

## What changed / 实施内容

- **Scale and gap:** standing visual width 0.142 m; lanes 0.20 m; track 0.60 m.
  Separate floor boxes and interrupted rails leave an actual void. A visible
  0.90 m recovery bay provides space around the sweeper. Original asset materials
  are retained. Dimensions come from the asset rather than guessed body length.
- **Physical props:** a hinged bar, free-joint crate released from a weld after
  an 0.85 s warning, torque-limited sweeper, force-driven sphere and a sliding
  door limited to 6 N. Raising the bar hinge lever from 0.22 to 0.32 m lets the
  sphere push it open; the bar remains low enough to obstruct a standing duck.
- **Control:** learned motor policies at 50 Hz; MuJoCo at 5 kHz; joint torque
  limit 0.6405 Nm. Maneuvers finish on observed rotation, support, orientation
  and velocity. Timeouts mean failure. Moving roll entry uses measured left-foot
  contact and upward body velocity. Recovery uses motors, without root resets.
- **Planning:** Jev selects the next encounter maneuver, with a 2–4 s tactical
  horizon; the executor controls exact roll and jump timing. Responses are
  prefetched and revalidated at handoff. Actual execution can still pause when
  a predicted route becomes invalid; this remains a limitation.
- **Rendering:** recorded 200 Hz states, event-driven cameras, 0.72 m lookahead,
  original robot colors, and optional four-sample motion blur. Each cut has at
  most three slow-motion segments. Audio is synthesized Foley, not a microphone.

缩窄跑道、真坑、自由落箱、有限扭矩扫杆和有限力巨球已落地。所有动作通过电机策略
执行，初始化后不写入机器人根姿态。镜头读取记录姿态仅用于渲染；物理真实性另由
记录电机输入重新推进仿真的回放证明。Jev 不是 50 Hz 反射控制器。

## Contact audit / 接触真实性

The publication gate checks **every force-bearing contact category** at 5 kHz:
maximum geometric penetration <1.5 mm, P99 <1 mm, no solver warnings, matching
capture hashes and successful input replay. Hazard-only numbers are insufficient.

| Selected physical baseline | Maximum | P99 |
|---|---:|---:|
| Duck × floor | 1.467 mm | 0 mm |
| Duck self-contact | 0.259 mm | 0.234 mm |
| Duck × hazard | 0 mm | 0 mm |
| Props × floor | 0.863 mm | 0 mm |
| Props × props | 0.260 mm | See raw audit |

[Full contact audit](../artifacts/neon-escape-v2/hard-floor/full-stunts/all-contacts.json)
· [input replay: 5,732 states, zero position/velocity error](../artifacts/neon-escape-v2/hard-floor/full-stunts/replay.json)
· [run and source hashes](../artifacts/neon-escape-v2/hard-floor/full-stunts/audit.json).

A positive contact margin can generate force before geometric overlap: zero
negative distance does not mean no collision. Impulses, damage, falls and recovery
are recorded separately. These are compliant contacts, not a proof of zero overlap
under arbitrary conditions. The floor maximum is close to the threshold.

Native ground and self geometry are preserved. Massless per-body boxes expanded
by 0.2 mm are used only for hazard contact; they preserve robot mass and inertia.
Explicit floor pairs use a 2 ms reference time and 0.2 mm margin. The filmed
baseline used 2 ms self-contact; current development uses 1.2 ms self-contact
with zero margin. Those profiles must not be pooled into one success rate.

完整审计同时检查鸭子—地面、自身、障碍，以及道具接触。通过的基线地面最大重叠
1.467 mm，接近阈值，并非“绝不穿模”。正 margin 中也有真实接触力；击倒和起身
必须有真实事件证据。当前开发参数与基线略有不同，历史结果不能直接算到新配置上。

## Skills and remaining gates / 技能与未完成项

| Requirement | Evidence and limitation |
|---|---|
| Real long jump | Current hard profile, settled launch 6 cm before edge: 12/15/18/20 cm gaps passed **9/10, 10/10, 9/10, 3/10** development trials. Whole-course approach remains sensitive. |
| Moving roll | Foot-phase entry works in component tests; lateral drift at handoff remains significant. The filmed baseline completes a full inversion and returns to supported upright motion. |
| Quick dodge | **Not passed.** A full 20 cm lane generally takes about 1.6–2.3 s. Short integrated dodges can start close to the destination lane; duration without displacement is misleading. |
| Recovery | Selected genuine sweeper knockdown recovers in **1.08 s**. Static pose batteries do not certify arbitrary moving impacts. |
| Chase | Baseline records **20.676 s** continuous pursuit and a real closed-door impact. |
| Running | Commands and observed speed are recorded separately. End-to-end pace still suffers from alignment and braking. |
| Live Jev | Real API runs and failures are retained. One complete live run failed the strict self-contact gate (1.554 mm maximum, 1.179 mm P99). It is not published as qualified. |
| Generalization | Frozen hard-contact live cohort **201–206: 2/6 valid escapes, 0/6 full-stunt publication passes**. No reruns. Historical native-ground cohort 101–106 had 4/6 escapes; that is a different configuration. |

硬地面长跳单项已有进展，但从跑步制动切到起跳仍然不稳定。20 cm／1.2 s 快闪没有
通过；不能拿翻滚后只剩 5–7 cm 的小幅换道冒充完整一车道快闪。在线 Jev、动作交接
与动态击倒的稳定性仍需继续解决，不能宣传整关高成功率。

### Frozen hard-contact evaluation / 冻结配置测试

[Seeds 201–206, exact snapshot and complete results](../artifacts/neon-escape-v2/unseen-hard-201-206/summary.json):

| Seed | Outcome | Full contact gate |
|---|---|---|
| 201 | Fell into gap; recovery failed | Failed |
| 202 | Fell into gap | Failed |
| 203 | Valid escape, no actual knockdown/recovery | Passed |
| 204 | Reached finish but exhausted HP | Passed |
| 205 | Valid escape, no actual knockdown/recovery | Passed |
| 206 | Fell into gap | Failed |

All six input replays passed. These seeds vary obstacle placement; they are a
small test, not a broad distributional benchmark. API timing also affects physical
handoffs. Failures include excessive self-contact during falls, so low penetration
in the selected film must not be generalized to every episode.

新六种子在线测试仅 2/6 有效逃生，0/6 满足全套动作发布门槛。六条输入回放均通过，
但掉坑时仍可能出现超阈值自身接触。这个结果说明当前整合不稳定；两条逃生记录也
不能补拍或拼接倒地恢复，伪装成完整在线成功。

## Failure analysis / 从失败中得到的结论

1. **Soft native contacts hid a major defect.** Earlier live development finished
   in 21.14 s, but full auditing found ground penetration 17.684 mm/P99 6.165 mm
   and self-contact 3.785 mm/P99 3.458 mm. Diagnostic videos remain local.
2. **Harder contact changes skills.** Reusing old roll timing or launch distance
   failed. With current contact, a 15 cm jump passed 10/10 at launch distance
   4 or 6 cm, 9/10 at 8 cm and 4/10 at 10 cm. These are development sweeps.
3. **Stand is not a brake.** Switching a still-moving body to the standing policy
   near the edge allowed continued drift. The executor now separates short
   positioning pulses, zero-speed locomotion braking and standing stabilization;
   an unsafe launch aborts explicitly. The approach is still experimental.
4. **Geometry before solver tuning.** An early 8 cm rail intersected the sweeper
   at reset; those tests are invalid. Current rails are 2.5 cm. A low bar hinge
   blocked the boss; raising its pivot corrected the physical leverage.
5. **Old jump reuse failed.** The historical task initialized the root at z=0,
   placing soles below the floor. The new task starts at z=.12, verifies standing
   physics, initializes PPO from the official standing actor and uses normalized
   export. The selected checkpoint is `model_500.pt`.
6. **Roll retraining was not adopted.** A short hard-contact PPO transfer used a
   zero-margin training profile required by its Warp collision configuration.
   CPU deployment checkpoints failed transfer tests; increasing training reward
   alone would not qualify them. The original rollout policy remains selected.
7. **Planner envelopes must match physics.** Current route envelopes contain 52
   qualified entry-state buckets from 162 development trials, including failures.
   The live executor rejects mismatched lane/contact profiles and rechecks the
   swept path against obstacles and track boundaries at handoff.

这些失败记录保留在仓库。没有删除失败种子、改写策略输出姿态或降低接触门槛来取得
好看的视频。训练、单项扫参、整关和未见种子分别报告。

## Reproduce / 复现

Use the repository environment and asset checkouts. API credentials stay outside
Git (`TYPESAFE_API_KEY` or `~/.config/microduck/typesafe.key`). Jev runs incur API
usage. Outputs refuse to overwrite an existing run audit.

```bash
export OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/freeze_neon_v1.py
.venv/bin/python -m pytest tests/test_parkour_physics.py tests/test_neon_arcade.py -q
.venv/bin/python -m microduck_lab.demos.parkour_rehearsal \
  --policy policies/parkour_long_jump_v2.onnx --brain jev --hard-contacts \
  --sweeper-mass .4 --sweeper-torque .2 --sweeper-phase .9 --capture \
  --output artifacts/neon-escape-v2/local-new-run
.venv/bin/python scripts/replay_parkour_run.py artifacts/neon-escape-v2/local-new-run
.venv/bin/python scripts/audit_parkour_contacts.py artifacts/neon-escape-v2/local-new-run
```

This runs the **current experimental controller**, not a guarantee of reproducing
the older selected baseline. Exact historical source archives and physical model
hashes accompany each run. `audit.json:passed` is the encounter/stunt gate;
publication also requires matching `all-contacts.json` and `replay.json`.

```bash
.venv/bin/python scripts/render_parkour.py \
  --source artifacts/neon-escape-v2/hard-floor/full-stunts \
  --output out/microduck_neon_escape_v2_physical_baseline_hero.mp4 --cut hero
.venv/bin/python scripts/render_parkour.py \
  --source artifacts/neon-escape-v2/hard-floor/full-stunts \
  --output out/microduck_neon_escape_v2_physical_baseline_technical.mp4 \
  --cut technical --blur 1
```

See [evidence index](../artifacts/neon-escape-v2/README.md),
[policy provenance](../policies/parkour_long_jump_v2.json), and
[training reconstruction](../training/README.md).
