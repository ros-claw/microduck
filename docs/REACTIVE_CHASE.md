# Reactive Chase — state-dependent execution / 根据状态选择动作

This revision addresses the rigid route and stop-start behavior of the V2
physical baseline. Jev still selects encounter maneuvers. A local physics model
now evaluates short action sequences from the **current simulator state** before
committing to a gap approach or sweeper route. Existing learned motor policies
execute the selected sequence; no new policy weights or root-pose animation were
introduced.

这次主要改的是动作衔接和判断，不只是加快播放或改变镜头。Jev 选择关卡动作，局部
物理模型尝试不同的起跳衔接和绕杆路线，再由原有电机策略执行。预测只推进仿真副本，
不会把预测出的姿态写回正在运行的鸭子。

**Scope:** model-based planning with privileged simulator state and a known
course. It is not camera perception, a hardware demonstration, or a real-time
planning certificate. The films play recorded **simulation time**. In the selected
seed-12 run, gap planning took **2.727 s** and sweeper planning **5.021 s** of wall
time; the 25.92 s complete capture took **36.41 s** to execute. Jev requests took
751–958 ms. These computation delays are not included as frozen frames in the
simulation-time film.

**重要范围：** 预测使用仿真完整状态和已知物理模型，不是视觉理解。视频按仿真时间
播放，不能当作实时真机能力：所选记录的跨坑／扫杆规划分别耗时约 2.7／5.0 秒，
25.92 秒完整仿真记录用了约 36.41 秒墙钟时间。计算时的停顿没有插入仿真时间视频。

## Changes / 改动

| Problem | Implementation |
|---|---|
| Fixed jump-distance trigger fails after different approaches | Preview bounded advance → brake → settle → learned jump sequences from current joint state, velocity and previous action. Slow arrivals get longer approach candidates. A failed forecast is retained, not called a safe plan. |
| Robot walks directly into the arm | Compare routes on both sides, at 0.30/0.37 m offsets, with the moving torque-limited arm included. Prefer a predicted contact-free route; if only upright routes with contact remain, select lower predicted impulse and explicitly record that fallback. |
| Short contacts disappear from coarse forecasts | Sweeper forecasts inspect contact forces at every 0.2 ms step. Publication independently replays and audits all contacts at the same rate. |
| Hanging crates appear permanently harmless | Before release timing is known, reserve the entire future drop column. Check continuation past the crate, not only the lane-change endpoint. |
| Large fixed lane shifts overshoot the narrow road | Re-measure routes targeting ±0.14 m; use foot envelopes for ground support and full-body envelopes for obstacle collision. The 0.60 m road stays the same width. |
| A brief entry-state mismatch discards a valid Jev response | Retain the answer for at most 0.35 s while rechecking legality. Never execute it while illegal, extend its expiry, or carry it into another encounter. |
| Boss becomes trapped after being knocked sideways | Chase mode adds a bounded ±0.35 N lateral centering force. The sphere still rolls through contact physics; it is not translated or teleported. |
| Static obstacle placement and weak pursuit | Chase mode places the crate at y=±0.12 m, starts the sphere 0.30 m closer and raises its target speed from 0.40 to 0.44 m/s. |
| Empty-looking set | Renderer-only industrial scenery, physical-gap warning markings, speed ticks, outlined crate and a dark red pursuit sphere. Native duck materials and all collision geometry remain intact. |

规划不再假设“完成换道就一定能通过落箱区”。左右落箱使用同一套控制器，根据观测
重新筛选路线。±14 cm 是绕过障碍的路线目标，**不能冒充已通过 20 cm／1.2 s 快闪**。
新包络来自 162 次开发试验，53 个入口条件组通过；这些不是独立未见整关试验。

## Selected live records / 两条在线记录

Both use real Jev API responses, the same controller and unchanged motor weights.
They are selected development records, not a success-rate estimate.

| Record | Drop zone | Jev crate choice | Finish, simulation time | HP |
|---|---|---|---:|---:|
| `final-dev-11` | y = −0.12 m | Left route | 17.62 s | 3/3 |
| `final-dev-12` — filmed | y = +0.12 m | Right route | 17.90 s | 3/3 |

Both have real gap crossings, no duck–hazard force-bearing contacts, and a sphere
impact against the closed finish door. All recorded input replays pass exactly:
5,128 states for seed 11 and **5,184 states for seed 12**, zero position/velocity
error. This proves the saved trajectory follows the recorded actuator inputs in
this simulator configuration; it does not prove generalization.

同一个控制器在两种落箱位置下分别选择左绕、右绕；两条都保留全部生命值，并记录了
真实跨坑和巨球撞门。它们是精选开发记录，不能只展示两条成功就宣传稳定成功率。

For the filmed seed-12 run, the full-rate audit reports:

| Contact category | Maximum overlap | P99 |
|---|---:|---:|
| Duck–floor | 0.866 mm | 0.019 mm |
| Duck self-contact | 0.228 mm | 0.124 mm |
| Props–floor | 0.842 mm | 0 mm |
| Props–props | 0.635 mm | 0.004 mm |
| Duck–hazard | No force-bearing contact | Not applicable |

[Audit](../artifacts/neon-escape-v3/final-dev-12/all-contacts.json) ·
[Input replay](../artifacts/neon-escape-v3/final-dev-12/replay.json) ·
[Run, decisions and predictions](../artifacts/neon-escape-v3/final-dev-12/audit.json).

**Two different evaluation goals:** the original `stunts` goal still requires a
real knockdown/recovery and observed hazard contact. The new `escape` goal rewards
avoiding the hit; it does not require deliberately being knocked down. Both retain
exactly the same penetration thresholds: max <1.5 mm and P99 <1 mm, across every
force-bearing category, with matching capture hashes, no solver warnings and an
input-replay proof. A clean escape is not counted as a four-stunt pass.

“逃生”和“全套特技”分别验收。原来的真实击倒起身门槛保留；这次不再为了表演恢复而
故意走向扫杆。接触穿透阈值没有降低，无接触不写成“完成了碰撞恢复”。

## Frozen evaluation / 冻结配置评估

The frozen cohort produced **5/6 valid escapes**, but only **3/6 escapes passed
all publication contact limits**. All six input replays passed. Every successful
escape retained 3/3 HP.

| Seed | Escape | Strict escape publication | Finding |
|---|---|---|---|
| 301 | Pass | Pass | Clean escape |
| 302 | Fail | Fail | Waiting/route failure, exhausted HP and fell |
| 303 | Pass | Fail | Self-contact max 2.700 mm; P99 also exceeds 1 mm |
| 304 | Pass | Pass | Clean escape, right route |
| 305 | Pass | Pass | Clean escape |
| 306 | Pass | Fail | Self-contact max 1.970 mm; P99 also exceeds 1 mm |

新六种子：5/6 有效逃生，严格发布检查为 **3/6**。成功逃生的五条均为满生命值；
303、306 的自身接触仍超标，不能作为通过物理验收的视频发布。302 的等待和路线
执行失败也保留，没有换种子替代。

Results for the fixed, once-only live cohort 301–306 are recorded in
[`unseen-301-306`](../artifacts/neon-escape-v3/unseen-301-306/summary.json).
The source and policy hashes were frozen before testing; all failures are kept.
This more difficult course differs from V2's historical 201–206 cohort, so their
percentages must not be presented as a controlled before/after comparison.

六个新种子每个只跑一次。新关卡增加了落箱位置变化和追逐压力，与旧关卡不是同分布，
不能把两个成功率直接相减，宣称算法提升了多少个百分点。

An intermediate matched development ablation on the *classic* course used seeds
0–2: the earlier execution failed all three, while the intermediate predictive
revision escaped in all three. This small development comparison is retained in
`control-*` / `route3-*`; it is not the final held-out score.

## Reproduce / 复现

```bash
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest tests/test_parkour_physics.py tests/test_parkour_preview.py -q
# Uses Jev credentials outside Git; API usage is incurred.
.venv/bin/python -m microduck_lab.demos.parkour_rehearsal \
  --policy policies/parkour_long_jump_v2.onnx --brain jev --hard-contacts \
  --predictive --difficulty chase --seed 12 \
  --sweeper-mass .4 --sweeper-torque .2 --sweeper-phase .9 --capture \
  --output artifacts/neon-escape-v3/local-new-run
.venv/bin/python scripts/replay_parkour_run.py artifacts/neon-escape-v3/local-new-run
.venv/bin/python scripts/audit_parkour_contacts.py artifacts/neon-escape-v3/local-new-run
```

A new live run can differ because API results and wall-clock arrival times affect
handoffs. For exact saved-state reproduction, use the published MJB/input capture
rather than calling the API again.

```bash
.venv/bin/python scripts/render_parkour.py \
  --source artifacts/neon-escape-v3/final-dev-12 \
  --output out/microduck_neon_escape_v3_reactive_hero.mp4 \
  --cut hero --style industrial --goal escape
```

The English action/technical cuts have at most three slow-motion segments each.
Audio is synthesized Foley. The full capture, historical source and audits are
provided separately from Git as a release asset. V1 and the earlier physical
baseline are preserved. Remaining work includes planning latency, robustness
outside the calibrated entries, a genuinely fast 20 cm dodge and real perception.
