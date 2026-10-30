# Duck Fu evaluation / 功夫鸭评估

[English methods](README.md) · [中文方法](README.zh-CN.md) · [Machine-readable summary](../../artifacts/combat/complete-summary.json)

This separates physical validity, motor ability, tactical comparison and human viewing acceptance. A watchable film alone does not pass the other gates. 本页分别报告物理、运动能力、战术对战和真人观感，不用精选视频替代统计。

## Frozen motor package

Contact drive: `duckfu_contact_v1_100.onnx`; lateral step: `duckfu_evade_v1_200.onnx`; combat walking: `duckfu_locomotion_v1_200.onnx`. Every final competition uses the same body package on both teams. Frozen file hashes appear in the configs and source lock.

Forty fresh initial states, four paired variants each; twelve simulated seconds per row. Seeds are separate from development and game evaluation. [Protocol](../../configs/combat/eval_motors_v5.yaml) · [All 160 rows](../../artifacts/combat/complete-motor-validation.json.gz).

| Measurement / 指标 | Result / 结果 |
| --- | --- |
| Standing-target attack: actual contact, ≥1 cm target displacement and final stable attacker | **40/40**; Wilson 95% interval 91.2–100% |
| Median standing-target displacement | **117.8 mm** |
| Evasion: first attack avoided and defender stable | **37/40**; defender final stability 40/40 |
| Counter course: first attack avoided | 35/40 |
| Counter contact after facing back | 37/40; defender final stability 39/40 |
| Numerical proposal | **160/160 pass**; maximum penetration 0.297 mm |
| Matched-body ablation without dedicated contact motor | 0/40 effect-and-stable attacks |

The ablation uses the new combat-walking body package with its dedicated drive motor removed. It is not a claim that every upstream walking policy scores zero. A defender's self-actuated evasion/counter displacement is not reported as attack strength. Counter contact is not automatically an effective strike, a successful complete sequence or a game win.

The twelve-command CPU walking test includes reverse, lateral and reverse+side+turn, followed by zero-command braking. All twelve ended supported and stable; largest observed head offset was 0.323 rad. These are twelve command cases, not forty independent locomotion trials. Broader stability is measured by the complete-game evaluation.

[Normal-speed motor sample 1](media/duck_fu_duel_greybox.mp4) · [sample 2](media/duck_fu_motor_02.mp4) · [sample 3](media/duck_fu_motor_03.mp4) · [missing-counter example](media/duck_fu_motor_failure.mp4). Selection is the first three avoided-and-counter-contact examples and first missing counter in the frozen validation; it is disclosed in the [selection record](../../artifacts/combat/motor-film-selection.json). All four regenerated 96,000 integration steps each with zero state error.

## Recovery / 起身恢复

All attempts and incomplete episodes remain in the raw reports. Reset-pose results are not mixed with contact-driven falls. The competition uses standing/sit-to-stand retries; the rejected rolling prior is disabled. The R3 reset-pose course uses a radius-0.75 m legacy disc with its own initial friction perturbation, not the final team octagon. The actual 8 kHz profile commands are recorded in the [profile provenance](../../artifacts/combat/recovery-profile-provenance.json); the original harness did not embed the resolved Physics dataclass in each row.

| Battery | Attempted | Actual eligible falls | Recovered ≤3 s | Recovered ≤10 s | Completed P50 / P90 |
| --- | --- | --- | --- | --- | --- |
| R0, nonpenetrating reset poses | 40 | 40 | **39/40** | 40/40 | 1.33 / 1.62 s |
| R3, near edge + initial friction perturbation | 40 | 40 | **40/40** | 40/40 | 1.39 / 1.92 s |
| Nominal R1, contact stress coach then commanded retreat | 160 | 50 | **32/50** | 50/50 | 2.74 / 3.87 s |
| R2, continued-interference course | 160 | 48 | **24/48** | 48/48 | 2.97 / 5.47 s |

The R1 coach uses a disclosed rolling stress course, with actual measured contact before the target fall, then commands a one-second retreat and zero speed. A retreat command does not prove actual clearance; residual contact can remain. This is a nominal stop-attacking stress protocol, not clean proof that every recovering duck was physically isolated. Its **64%** within three seconds misses the early 80% target. The 110 courses without a qualifying contact fall are retained, not discarded as unsuccessful recoveries or counted as successes. R2 has 112 courses without a qualifying contact fall; all attempts remain in its separate report. R2 is a harder, continuing-command stress protocol rather than an equal-difficulty substitute for R0.

[R0 rows](../../artifacts/combat/recovery-v5-R0-28001.json.gz) · [R3 rows](../../artifacts/combat/recovery-v5-R3-28101.json.gz) · [R1 rows](../../artifacts/combat/recovery-v5-R1-29001.json.gz) · [R2 rows](../../artifacts/combat/recovery-v5-R2-29201.json.gz)

## Tactical training and paired cross-play

Rule distillation is initialization, not self-play. Solo training subsequently made twelve online PPO updates × sixteen games; the corrected final team run made eight updates × eight complete games, resuming a real actor/critic/optimizer checkpoint. Opponents are frozen for each episode. The team training run's 64/64 games passed numerical checks; no batch needed rejection in that final run. Earlier rejected team training remains in the experiment archive.

The final 1v1 matrix compares rule, distilled and PPO tactics against rule-pressure, rule-counter and historical-distilled opponents. Every cell uses **32 base seeds × both sides = 64 games**, with arena rotation. The frozen team actor uses the same paired design against three opponents, with a separate paired rule baseline. Seeds used for training, motor qualification and final cross-play are distinct. The final team checkpoint was selected by its predeclared update count, before looking at held-out wins.

See the final machine-readable summary and paired bootstrap reports for W/D/L, no-contests and uncertainty. Paired confidence intervals resample the 32 base seeds; they do not treat two sides or different opponents on one seed as independent observations. Numerical rejections remain in scheduled-game denominators. A PPO training run is not an automatic Champion promotion; regressions and lack of gain must remain visible.

### 1v1 frozen matrix

Each row is 32 base seeds × two sides; W/D/L are from RED's perspective.

| Candidate vs opponent | W | D | L | No-contest | Numerical pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule vs rule_pressure | 8 | 49 | 7 | 0 | 64/64 |
| rule vs rule_counter | 12 | 20 | 32 | 0 | 64/64 |
| rule vs historical_distilled | 12 | 36 | 16 | 0 | 64/64 |
| distilled vs rule_pressure | 18 | 26 | 20 | 0 | 64/64 |
| distilled vs rule_counter | 16 | 34 | 14 | 0 | 64/64 |
| distilled vs historical_distilled | 21 | 25 | 18 | 0 | 64/64 |
| learned vs rule_pressure | 6 | 47 | 11 | 0 | 64/64 |
| learned vs rule_counter | 12 | 29 | 23 | 0 | 64/64 |
| learned vs historical_distilled | 11 | 31 | 22 | 0 | 64/64 |

[Paired bootstrap report](../../artifacts/combat/duel-paired-summary.json).

### 2v2 frozen matrix

Each row is 32 base seeds × two sides; W/D/L are from RED's perspective.

| Candidate vs opponent | W | D | L | No-contest | Numerical pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| team_learned vs rule_pressure | 6 | 53 | 5 | 0 | 64/64 |
| team_learned vs rule_counter | 10 | 33 | 20 | 1 | 63/64 |
| team_learned vs historical_distilled | 3 | 55 | 6 | 0 | 64/64 |
| rule vs rule_pressure | 4 | 59 | 1 | 0 | 64/64 |

[Paired bootstrap report](../../artifacts/combat/team-paired-summary.json).

The solo PPO actor changes mean win rate versus the rule baseline by **−1.56 percentage points**, paired bootstrap 95% interval **−7.29 to +4.17**. The teacher-distilled actor gives **+11.98 points**, interval **+5.21 to +18.75**, across the same frozen opponent pool. Team PPO versus the paired rule baseline on the common pressure opponent gives **+3.13 points**, interval **−7.81 to +12.50**. These results do not establish a stable PPO gain. The distilled intermediate is promising, but choosing it after seeing this holdout would require a fresh validation before promotion. **No tactical Champion is promoted.**

Across all **832 scheduled games**, **831 passed** numerical checks. One team-vs-counter game (base seed 50018, swapped side) reached **3.0228 mm** stage contact depth and was automatically declared no-contest at 12.46 s. It is retained in the denominator and in a [continuous silent diagnostic film](media/duck_fu_numerical_rejection.mp4), not counted as a win. The successful selected team film is a different match, seed 50006, with **0.595 mm** maximum penetration.

The original frozen evaluation actor source is `ebc6d26`. A subsequent **post-match reporting-only** correction (`c6e0dec`) separates interrupted windows from completed misses and distinguishes recovery-guard events from referee down-count clears. Raw reports are preserved; the derived v2 reports recompute only event attribution. Fresh complete 1v1 and 2v2 captures after the fix had **bit-identical controls, physical states, contacts, rewards and referee results**. Final films and their zero-error replays use the corrected source. Earlier motor/diagnostic clips use the archived ebc source. [Duel parity](../../artifacts/combat/duel-attribution-physics-parity.json) · [Team parity](../../artifacts/combat/team-attribution-physics-parity.json) · [Frozen source archive](../../artifacts/combat/source-snapshots/competition-ebc6d26.tar.gz).

## Physical audit and reproducibility

All integration steps check finite state, clock, solver warnings, zero undeclared external forces, model mass/actuation invariance and torque limits. Penetration is measured on the **penetrating-contact subset**; categories duck–duck, duck–stage, self and receiver have separate maxima, world positions and geometry IDs. The numerical proposal is max ≤3 mm and pooled penetrating P99 ≤1.5 mm. Passing it does not by itself establish visually exact contact.

Circular-cylinder/libccd rim failures reached 8.84 mm. Native convex detection improved the local rim but failed other self/inter-robot contacts in complete matches. The final fixed octagon/libccd/8 kHz profile passed both previously failing sixty-second regression seeds, with maxima 0.324 and 0.711 mm. [Rejected and fixed regression records](../../artifacts/combat/rejected-and-fixed-regressions.json.gz) preserve those failures.

Closed-loop replay reruns all controllers, independent motor histories, physics, contacts, decisions and referee; it does not merely load poses. The 45-second final 1v1 reference regenerated **360,000 integration states with zero error** and identical semantic logs. The selected team match regenerated **399,360 states with zero error**. Per-frame film audits are recorded in the complete summary. The technical film's contact hold is an actual solved upper-leg contact from the raw log, not an assumed head hit or a force sustained for two seconds. Slow motion uses actual recorded source states, and every encoded frame has a source-step mapping.

A fresh delivery environment installed the locked dependencies and pinned upstream assets, passed **185 tests**, and ran a complete match through the CLI. Applying the complete training patch to its clean base passed **five training contract tests**. Frozen rope/parkour/survival actor directories are unchanged. Hardware transfer is untested.

## Timestep sensitivity / 步长敏感性

The representative 0.3 s fixed-input contact test compares the same recorded motor tape at dt and dt/2, including impulses, penetration, root displacement and instantaneous down/ring-out predicates. This is a local component test, not a resumed match. See the [full contact comparison](../../artifacts/combat/final-duel-contact-convergence-v2.json).

An additional **complete closed-loop** check on duel seed 32001 changed from a 45 s draw at 0.125 ms to a BLUE win at 9.34 s at 0.0625 ms. Both runs passed the penetration/force/solver checks, but their observations and subsequent actions diverged. **Game outcome is not timestep-invariant.** The benchmark and selected films describe the frozen 8 kHz profile; exact replay proves reproducibility within that profile, not unique physical ground truth or hardware validity. The changed result is retained in its [manifest](../../artifacts/combat/final-duel-dt2-manifest.json), not suppressed as a numerical no-contest.

## Remaining limits / 已知边界

- Component convex hulls approximate concavities; native ground/self shapes remain upstream approximations.
- Dynamic recovery can exceed three seconds; continued interference and edge falls remain harder.
- Misses, draws and numerical no-contests are legitimate outcomes. A film selection is not a random sample.
- No trained short kick, qualified rolling escape, hardware deployment, network-LLM reflex loop or autonomous evolution is claimed.
- ROSClaw typed SIM executor integration is implemented, but official daemon-authorized combat receipt validation is pending. The executor does not manufacture authorization receipts.
- **Human viewing gate: PENDING_HUMAN_REVIEW.** No five-person comprehension score is invented; publication remains subject to user acceptance.
