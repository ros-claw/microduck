# Microduck: Neon Escape / 霓虹逃脱

An independent **ROSClaw Physical Arcade** prototype: Jev selects a bounded motor skill, existing Microduck ONNX policies actuate a physical body, and MuJoCo decides the outcome. The rope-skipping reference implementation is unchanged.

[Watch the English gameplay film](https://github.com/ros-claw/microduck/releases/download/neon-escape-2026-09-26/microduck_neon_escape_en.mp4)

## What works

- Actual TypeSafe API calls, with returned model ID, action distribution, confidence, token usage and measured latency saved in each decision record. No fake Jev responses in evaluation or film.
- Walking, closed-loop lane changes, braking with heading stabilization, and a complete forward roll followed by supported standing. Robot motion comes from 14 servo targets; no robot root-pose assignment, velocity injection or mocap.
- One fixed high bar plus **six dynamic hazard types**: lifting gate, rotary sweeper, gravity-released guided crate, sliding wall, pendulum and a force-driven rolling boulder. Scene motors have force limits; all hazard–robot collision pairs are enabled.
- Seeded obstacle order, lane, movement phase and bar clearance. Three lanes permit alternative routes, but the generator does **not** yet prove dynamic reachability of every level.
- Read-only 5 kHz collision audit, failure latch, crossing-based points, and a roll bonus requiring a full measured rotation, actual body inversion, upright completion and at least 50 ms foot support.
- Recorded-decision replay, with exact equality checks on the original position/upright/action trace, contact audit and skill transitions. Rendering uses 200 Hz saved states, not pose interpolation.

The demonstration is **simulation**, not a physical hardware deployment. “Laser” bars are stylized solid collision geometry, not an optical laser model. Lane paint and buildings outside the track are visual decorations.

## Results, including failures

The displayed seed 102 run reaches the finish in **39.98 simulation seconds**, clears six course obstacles without hazard contact, and completes one verified full roll. The pursuing ball is not counted as a cleared obstacle. In this layout it rolls into the first high bar and is physically blocked at x ≈ 0.744 m; it does not chase the duck all the way to the finish. The film includes a close-up of this contact. A true final-stage boulder chase remains future level design.

| Development seed | Jev | Rule baseline, same final scene |
| --- | --- | --- |
| 0 | Fail: sweeper contact, then timeout | Clean finish, 33.92 s |
| 101 | Reaches finish after gate contact: **fail**, 50.64 s | Clean finish, 35.62 s |
| 102 | **Clean finish, 39.98 s** | Clean finish, 37.38 s |

A separate fresh-seed check was then run without tuning the controller to its outcomes:

| Fresh seed | Jev outcome |
| --- | --- |
| 9001 | Clean finish, 37.22 s |
| 9002 | Clean finish, 42.66 s |
| 9003 | Reaches finish after crate contact: **fail**, 40.12 s |

That is **2/3 clean finishes on this small fresh-seed set**. All three complete a verified roll. No robustness claim is made from only three trials.

These are exploratory trials, not a reliable success-rate estimate. The baseline is currently better on this small set. Its tactical loop runs at 5 Hz, while live Jev is limited by actual network latency; baseline physics is allowed to run faster than wall time. This is not an intelligence-only or equal-latency comparison. Both use the same physical timestep, layout, skill runtime and collision scoring.

The successful film is explicitly a selected run; other outcomes remain in the [evaluation summary](../artifacts/neon-escape/evaluation.json) and individual audit files. Stiffer contacts reduce but do not eliminate numerical penetration: the failed seed 0 reaches **3.75 mm** at the sweeper. No contact was detected in the successful seed 102 run; this is not a proof of continuous collision detection or hardware fidelity.

## Real API measurement

A fixed hypothetical decision state was queried 1,000 times with four concurrent HTTP workers:

| Metric | Observed |
| --- | ---: |
| Valid responses | 1,000 / 1,000 |
| Latency p50 / p90 | 722 / 761 ms |
| Latency p95 / p99 | 780 / 943 ms |
| Input tokens | 588,000 |
| Returned model | `jev-1.13.0` |
| Same-state modal choice | JUMP, 1,000 / 1,000 |

This benchmark describes this machine, connection, request and time. Four-way throughput is **not** a single robot's decision frequency. Live gameplay uses one outstanding request, a **1.5 s deadline**, and roughly 1–1.4 completed decisions/s. Physics continues while a request is pending. No retries or unbounded request queues are hidden inside the control loop.

The first benchmark's strict sum-to-one validator rejected 27 rounded distributions. A direct API check observed sums of 0.99; the corrected validator permits only the accumulated two-decimal rounding tolerance and preserves original probabilities. Both benchmark files are retained. These were local validation failures, not 27 HTTP failures. No billing amount is inferred from token usage; actual billing was not fetched.

The HTTP adapter follows the official [TypeSafe API reference](https://docs.typesafe.ai/api). Unlike the proposal's illustrative JSON, Noul returns a probability in `noul`, not a separate boolean with confidence. Choice and Score have different response shapes. `act_now` and risk are recorded for analysis; execution currently uses the chosen action, confidence and escalation probability.

## Control and observations

```text
Current bounded state snapshot (no seed or future motor schedule)
  → candidate filtering
  → asynchronous Jev Choice + Score + two Nouls
  → deadline / confidence / escalation / current-legality check
  → committed motor skill at 50 Hz
  → MuJoCo at 5 kHz
  → physical contact and completion audit
```

The API sees current body state, hazards within a bounded longitudinal observation window, geometry, current hazard velocity, visible lane clearance, measured skill limits and mission progress. This is structured **simulation-state perception**, not camera vision. The full level seed and motor phase are saved only in the audit and not sent to Jev.

A roll commits for 2 seconds plus 0.8 seconds of stabilization. Lane commands commit for 1 second; braking has a short settling commitment. An API reply cannot interrupt a committed maneuver. A stale response or newly illegal choice is discarded. Model uncertainty triggers braking; a second, current-state geometric check can also brake before the next motor tick. This filter is conservative geometry, not a formal reachability or safety proof.

The current confidence threshold is **0.25**, lowered from the exploratory 0.5 setting after repeated stationary failures. This is an experimental threshold on TypeSafe's confidence field, **not** a 25% probability of safe motion, and it has not been calibrated to task success. The failed seeds above are part of assessing this trade-off. `need_system_two > 0.7` also brakes; no second model is silently invoked.

## Skill findings and current limits

- The walking policy has a substantial low-speed dead zone. Pure lateral commands barely move the body; lane changes therefore steer and walk.
- Switching directly to standing from walking produced roughly 25 cm of continued forward motion. The arcade brake uses zero forward walking command with yaw stabilization before standing.
- At a 31 cm start distance, 22–26 cm bar centers caused contacts, while 28 and 30 cm cleared. A separate full-width 28.5 cm bar cleared at tested distances 31, 35, 37, 39 and 42 cm. The generator uses 28.5–29.5 cm; candidate filtering restricts pose and start distance.
- Existing rope-hopping policies can hop in place. Forward hopping was unstable in the first transfer probe. **Parkour jumps, gap crossings, general fall recovery, a new quick-dodge policy, skate mode, Survival Arena and automatic retraining are not implemented or validated here.** Jump is absent from live candidate sets.
- This version reuses existing learned policies; it does not claim to have trained a new motor policy or demonstrated autonomous Darwin improvement. Roll rewards measure actual rotation and landing, not elapsed skill time.
- The scene is deliberately small. The high bar is a skill tutorial; the subsequent dynamic hazards offer route choices. Neither this design nor these measurements prove that Jev is necessary or superior to rule-based control.

## Run

From the repository root, after the normal asset/bootstrap setup:

```bash
# Actual Jev. Put the key in TYPESAFE_API_KEY, or a mode-0600 file at:
# ~/.config/microduck/typesafe.key
# Never put a key in the repository or command-line arguments.
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.demos.neon_run \
  --brain jev --realtime --seed 102 --seconds 60 --capture \
  --output artifacts/neon-escape/my-live-run

# Offline rule baseline, same physical interface:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m microduck_lab.demos.neon_run \
  --brain baseline --seed 102 --seconds 60 --capture \
  --output artifacts/neon-escape/my-baseline

# Reconstruct the published successful run from its actual decisions:
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_neon_run.py \
  --source artifacts/neon-escape/final-jev-seed102 \
  --output artifacts/neon-escape/my-replay

MUJOCO_GL=egl .venv/bin/python scripts/render_neon_escape.py \
  --source artifacts/neon-escape/my-replay --output out/my-neon-run.mp4

# This makes billable API calls; the count and concurrency are explicit.
.venv/bin/python scripts/jev_spike.py --queries 1000 --workers 4
```

Live API responses can vary, even with the same world seed. To reproduce the published trajectory, use the recorded-decision replay, which fails if its physical trace or audit differs. Source and policy hashes are included in final run reports. Large MJB/NPZ state caches remain local and can be regenerated; the public audit contains the decisions needed for replay. Robot and policy assets retain their upstream licenses; see the repository asset/license documentation.

## 中文说明

这是独立的新游戏线，保留跳绳参考实现。**真实 Jev 调用已经接入，并已有 39.98 秒无碰撞通关记录**：鸭子先翻滚过杆，再通过策略控制的转向与步行绕开动态障碍，所有动作经过身体动力学和碰撞求解。没有替鸭子写根节点位置，没有把规则基线冒充 Jev。

目前包含一根固定高杆和六类动态障碍：升降门、旋转扫杆、重力释放的导轨箱、横移墙、摆锤、受力滚动的大球。计分取决于实际越过障碍且没有接触；翻滚还必须测到完整旋转、恢复直立和足部支撑。

成功视频选用了种子 102。开发测试里 Jev 仅 1/3 干净通关，同样三个种子的规则基线为 3/3，**暂不能宣传 Jev 优于规则或稳定掌握了整个游戏**。另外冻结控制器后测试了三个新种子：9001、9002 干净通关，9003 碰箱子失败，即 **2/3**，样本仍很小。失败接触也仍有有限穿透，种子 0 的扫杆接触最大约 3.75 mm，不能笼统宣称零穿模。

1,000 次真实请求测得中位延迟约 722 ms、p99 约 943 ms，所以这里采用异步单请求管线和 1.5 秒截止时间，没有虚构 100 ms 或 10 Hz。画面中的延迟与置信度来自真实返回。当前置信度阈值 0.25 属于试验参数，尚未做任务成功率标定。

当前大球会被第一根高杆实际挡住，视频展示了这个接触结果，并非全程追到终点的 Boss；终局追逐关还需单独设计。现阶段完成了可运行的动作选择、物理关卡、技能边界试验、碰撞审计与录像复现。**跨沟长跳、通用跌倒恢复、专门训练的新技能和自动进化仍是后续工作**；不具备可靠证据的技能不会放进 Jev 的候选动作里。
