# Neon Escape V2 — physical development / 物理能力开发

V2 is under development. **There is no certified V2 hero run yet.** V1 remains
unchanged at `a1219f309bbfd4f2edb6ffb3b8ecba0cd3eb8d64`; its implementation,
failures and released video are checksummed in
[`v1-freeze.json`](../artifacts/neon-escape-v2/v1-freeze.json).

V2 正在开发，**尚无通过验收的 V2 完整宣传片**。这里记录可复现的物理实验，
不把训练奖励、单次成功或录制剪辑当作整关成功率。

## Current blocker / 当前阻塞

A live Jev development run completed roll → dodge → real-gap jump → actual
knockdown/recovery → finish in 21.14 s. Its 5,832 saved states replayed exactly
from actuator inputs. **This is not a release-qualified run.** A subsequent
5 kHz audit of *all* contacts found large native duck/floor and self-contact
compliance that the earlier hazard-only gate missed:

| Contact category | Maximum penetration | P99 |
|---|---:|---:|
| Duck × hazard | 0.013 mm | 0 mm |
| Duck × ground | **17.684 mm** | **6.165 mm** |
| Duck self-contact | **3.785 mm** | **3.458 mm** |
| Props × ground | 0.855 mm | 0 mm |
| Props × props | 0.420 mm | 0.139 mm |

[Full contact evidence](../artifacts/neon-escape-v2/live-development-8/all-contacts.json).
These are geometric overlaps, not an explanation that makes the visual defect
acceptable. The two rendered cuts are retained locally as **native-ground
diagnostics** and are not published as successful V2 films. The renderer now
requires a matching input-replay proof and full contact audit before export.

真实在线开发录制完成了全套动作，但全接触审计发现原生地面和自身接触过软。
因此暂停正式成片发布，保留失败证据。不能用障碍碰撞的低穿透数字代表所有物理接触。
目前正修复硬地面上的技能迁移；这会改变翻滚和着陆后的交接状态，旧电机策略不能
未经测试直接沿用。

A frozen six-seed live Jev cohort (101–106) produced **4/6 escapes**, **6/6
real-gap crossings**, and **1/6 passes of the earlier four-stunt gate**. That
older gate did not include native ground/self penetration, so none of those
counts certifies the stricter publication gate. No failed seed was rerun or
removed. The exact controller/policy snapshot and hashes are retained in
[`unseen`](../artifacts/neon-escape-v2/unseen).

## Design / 方法

The new `microduck_lab.parkour` package is isolated from V1. The track width is
three times a lane width derived from 1.4 × the actual standing asset width,
rounded up to a centimetre. The current model measures **0.142 m visual footprint
width, 0.20 m lane width, 0.60 m track width**. Earlier evidence used the native
collision footprint (0.125 / 0.175 / 0.524 m); those batteries are historical
results, not certification of the wider-lane controller. Height and footprint length are different quantities;
we report speed in m/s instead of inflating a “body lengths per second” number.
Floor boxes stop at each gap. Neither rail nor decorative geometry bridges the
void. Native asset materials are preserved.

场地按实测碰撞外壳缩放，而不是照抄估计值。真坑由分离的平台构成，没有隐藏地面；
侧护栏也在坑边截断。保留原生模型配色。地面和自身接触保持原生几何；高速障碍接触
使用逐刚体贴合的无质量盒状外壳，额外外扩 0.2 mm。碰撞位掩码将两套接触分开，
回归测试确认机器人质量和惯量不变。撞击区有明确可见的 0.9 m 宽恢复平台。

Props use physical mechanisms: a hinged crossbar; a free-joint crate held by a
releasable weld with an 0.85 s warning; a torque-limited velocity-controlled
sweeper; a driven rolling sphere; and a bounded-force sliding finish door.
Duck joints remain torque limited to 0.6405 Nm. Simulation uses 0.2 ms steps
and 50 Hz policy execution. Root pose writes are restricted to experiment
initialization; recovery and maneuvers use motor policies.

横杆可被推开；箱子解除焊接约束后完全自由；扫杆只接受有限扭矩；巨球通过力驱动滚动。
翻滚、刹车、换道和恢复采用状态完成条件，计时器只能导致超时失败。右向换道可使用
上游 61 维观测的左右镜像映射；它改变策略输入输出，不改变仿真姿态。

## Evidence / 验收记录

The table below records the earlier native-ground component configuration.
Full-course and held-out results are reported separately above. Development
sweeps retain failures; changing contact parameters invalidates transfer claims.

| Capability | Current evidence | Gate status |
|---|---|---|
| V1 freeze | 51 tracked implementation/evidence files unchanged | Passed |
| Real gap | Downward ray sees no floor in the gap; executable regression test | Passed |
| Moving roll | 10/10 perturbed starts; 1.28–1.32 s to full rotation + upright foot support; zero bar contacts | Passed in this battery |
| Recovery | 12/12 static face-up/down/left/right starts, 0.8–1.5 s | Static battery passed; live development recovery 1.98 s, but native ground failed |
| Quick dodge | One 17.5 cm lane: mirrored right 1.06–1.08 s; left 1.20–1.22 s | Borderline left; 20–30 cm gate not certified |
| Rotating impact | 36/36 tested speed/height/angle combinations; max geometric penetration ≈0.023 mm | Component gate passed |
| Boss vs props | Real contact opens bar and moves released crate; physical rails keep ball on course | Component passed; one live run maintained 9.21 s continuous chase and hit the closed door |
| Old `jump.onnx` | Initial 80 gap/launch/duration combinations: 0 successful crossings | Not certified |
| New long jump | 12/15/18/20 cm gaps: 40/40 full-rate airborne-crossing + stable-handoff trials; another 20/20 launch-window trials | Development component passed |
| Tactical prefetch | Live development run: Jev chose roll, left route and jump, 711–752 ms; all three executed through the finish | Integrated development run; frozen unseen results above |
| Films | No V2 hero/technical film released | Pending physical gates |

对应证据见 [`artifacts/neon-escape-v2`](../artifacts/neon-escape-v2)。
恢复测试的 `moving_impact` 初始轻推没有真正击倒鸭子，**不能计入倒地恢复通过数**。
早期扫参只检查站稳等宽松指标，不作为技能认证；最终必须验证整个空中轨迹和对岸支撑。

### Collision finding / 碰撞根因

The rotating impact test initially reached **2.91 mm** penetration. Explicit
Duck × Hazard contact pairs prevent the softer native duck/floor contact
reference from being mixed into obstacle contact. Sweeper pairs use a 0.8 ms
reference time, impedance `[.99,.999,.0005,.5,2]`, a 0.2 mm collision margin,
and bounded friction. The resulting maximum measured *geometric penetration*
is about **0.023 mm**; most impacts are resolved inside the positive margin.
Normal impulse still records those contacts: zero negative distance does **not**
mean zero physical interaction. The solver is compliant, not mathematically rigid.

接触参数按障碍类型单独设置，地面接触保持原生。正的接触 margin 也可能产生真实接触力，
因此审计同时记录法向冲量与几何穿透，不能拿“没有负距离”冒充“没有碰撞”。
此结果覆盖试验所列速度、角度和姿态，不能外推到任意高速冲击。

See MuJoCo’s [solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
and [explicit contact pairs](https://mujoco.readthedocs.io/en/stable/XMLreference.html#contact-pair).
`nativeccd` denotes convex collision detection; it is not a continuous-time
anti-tunneling switch.

### Jump investigation / 旧跳跃问题

The old checkpoint’s saved training configuration has root initialization at
`z=0`. The current inherited factory reproduces sole sites approximately
**0.117 m below the floor** at reset. This is incompatible with a standing
parkour launch. Our first development training inherited that defect; it was
stopped and retained as a failed experiment. The V2 factory explicitly starts
at `z=.12`; the official standing policy then remains upright in a 3 s Warp
rehearsal. PPO is initialized from the official standing ONNX with numerical
actor-output parity, and final exports still use the upstream normalized
exporter.

旧策略并未被删除，但它的历史腾空数字不能直接作为跑酷能力证明。尤其翻倒时脚也会升高；
新的奖励和部署验证要求身体直立、无地面支撑的腾空以及对岸稳定落脚。新训练不会把原地
跳绳的“禁止前移”惩罚带进长跳任务。

### Jev measurements / Jev 实测

A matched 90-request benchmark uses three **synthetic encounter fixtures**, not
live gameplay. All three profiles returned valid responses and matched the
fixture preference in 30/30 requests each:

| Profile | p50 | p95 |
|---|---:|---:|
| Choice | 719 ms | 913 ms |
| Choice + uncertainty Noul | 712 ms | 851 ms |
| Four questions | 714 ms | 785 ms |

This small sample does **not** show a latency advantage for Choice-only. V2
uses Choice + one uncertainty question because the latter participates in
handoff rejection; unused risk/act-now questions are omitted. Jev chooses the
next encounter maneuver; the physical executor controls exact launch timing.
No claim of 50 Hz model inference is made.

这次实测不支持“问题更少就一定更快”。保留真正参与执行判断的不确定性问项，
通过提前请求与交接时重验来处理约 700 ms 的响应时间。`live-development-3` 完成在线翻滚、
闪避、跨坑、进门，但没有真实击倒恢复，因此仍未通过完整 Hero 验收。

## Reproduce / 复现

From the delivery repository with its asset checkouts and Python environment:

```bash
export OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/freeze_neon_v1.py
.venv/bin/python -m pytest tests/test_parkour_physics.py -q
.venv/bin/python scripts/parkour_runtime_battery.py
.venv/bin/python scripts/parkour_recovery_battery.py
.venv/bin/python scripts/parkour_dodge_moving.py
.venv/bin/python scripts/parkour_sweeper_lab.py
.venv/bin/python scripts/parkour_boss_lab.py
# Requires TYPESAFE_API_KEY or ~/.config/microduck/typesafe.key; incurs API usage.
.venv/bin/python scripts/benchmark_parkour_jev.py
```

Legacy gap screening:

```bash
.venv/bin/python scripts/parkour_gap_battery.py
```

The API key stays outside the repository. `policies/parkour_long_jump_v2.json`
records the normalized ONNX provenance and development qualification scope.

```bash
.venv/bin/python -m microduck_lab.demos.parkour_rehearsal \
  --policy policies/parkour_long_jump_v2.onnx --brain jev --capture \
  --output artifacts/neon-escape-v2/local-run
.venv/bin/python scripts/replay_parkour_run.py artifacts/neon-escape-v2/local-run
```

The rehearsal is a development tool: its `passed` field requires a true recovery,
continuous boss pursuit, door impact, surviving HP, a supported finish, validated
gap flight and bounded hazard penetration. It is an encounter/stunt gate, not
a full contact certificate. Publication additionally requires `all-contacts.json`. A finished route alone is insufficient.

### Further failure analysis / 后续根因

The former 8 cm side rails intersected a 5 cm sweeper **at initialization**.
Those tests are invalid scene configurations, not desirable knockdown stunts.
Lowering the rails to 2.5 cm eliminated that initial overlap. A 30-case running
contact battery then measured max 0.141 mm, worst per-trial P99 0.036 mm.
The dropped boss additionally required explicit prop/floor contact pairs:
its peak floor penetration fell from 12.97 mm to 0.565 mm in the live development
run. Duck/floor contact parameters were kept unchanged.

不能靠调求解器掩盖出生时的几何相交。旧失败数据保留，但不会用于宣传“精彩碰撞”。
正接触 margin 内仍然可以有实际接触力，所有这类接触都会扣血并记录冲量。

Input replay initializes the state once, then advances physics using recorded
actuator commands, equality releases and bounded forces. An audited development
capture matched all 5,100 saved states with zero position/velocity error. This
is distinct from the renderer reading poses to display the already recorded run.


## Hard-contact transfer / 硬接触迁移

A controlled 18-trial moving-roll comparison on the current 20 cm lanes varied
only ground solver parameters. Keeping native impedance `[.9,.95,.001,.5,2]`
and setting a 2 ms reference with 0.2 mm margin gave 3/3 component completions
(max ground penetration 0.716 mm); the higher-impedance 2 ms profile gave 1/3.
The integrated run still failed its roll timeout, so **3/3 is not a robustness
claim**. The evidence is in `floor-transfer-visual-footprint.json`.

硬接触配置改善穿透，却改变已有策略的运动结果。正在基于官方 Roulade 任务微调，
使用原生几何、有限力矩和运动起步；训练结果必须重新经过 CPU 部署和整关验证。
20 cm 快闪、连续动作交接、硬地面倒地恢复和可靠整关仍是未完成项。

```bash
.venv/bin/python scripts/parkour_floor_transfer.py \
  --output /tmp/floor-transfer.json
.venv/bin/python scripts/audit_parkour_contacts.py \
  artifacts/neon-escape-v2/local-run
```
