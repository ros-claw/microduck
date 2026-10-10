# Reactive Chase evidence / 证据索引

[Methods, latency and limitations](../../docs/REACTIVE_CHASE.md). Runs use multiple
successive controller versions; do not pool them into one success rate.

| Record | Meaning |
|---|---|
| `probe-*` | Development prediction experiments. Early probes incorrectly aborted during the jump crouch; superseded, not qualification evidence. |
| `predictive-201`, `predictive-avoid-201` | Initial integration failures; known development seed. |
| `control-0..2`, `route3-0..2` | Intermediate matched classic-course ablation: 0/3 versus 3/3 escapes. Not the final held-out score. |
| `rule-*`, `route2-*` | Earlier controller attempts. `route2-*` failed report serialization (NumPy boolean); missing audits are not valid evidence. Diagnostic log retained. |
| `chase-live-*`, `chase-aware-*`, `refined-*`, `inner-*` | Development progression, including incorrect hanging-crate forecasts, blocked routes, insufficient jump approaches and boss trapping. |
| `support-route-qualification` | 162 trials measuring full-body and separate foot envelopes, nominal ±20 cm routes. |
| `inner-route-qualification` | 162 trials of ±14 cm routes; 53 qualified entry buckets; selected deployment envelopes. |
| `final-dev-11`, `final-dev-12` | Two selected live, clean escapes with opposite crate routes. Both pass exact input replay and full-contact limits. Seed 12 is filmed. |
| `unseen-301-306` | Final frozen live cohort, one attempt per seed: 5/6 escapes, 3/6 strict escape-publication passes. All six replays pass. |
| `diagnostics` | Logs explaining failed/superseded experiments. |

All large captures remain local except the selected seed-12 capture distributed
in the [Reactive Chase release](https://github.com/ros-claw/microduck/releases/tag/neon-escape-reactive-2026-09-28).
Its bundle contains the scene, recorded inputs/trajectory, historical source,
policy weights and audits. The other JSON hashes do not imply that every binary
capture is downloadable. API credentials are excluded.

每条记录附带其自己的源码快照。旧失败没有删除，新六种子没有重跑挑选。精选视频采用
“逃生”验收，不要求故意撞倒；原来的“全套特技”验收仍然单独保留。两种验收的接触
穿透阈值相同。规划耗时和仿真时间分别报告，不将渲染速度当实时计算性能。
