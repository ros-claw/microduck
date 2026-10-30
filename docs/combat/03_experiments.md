> Historical first-round record. Subsequent training, complete matches and current qualification are in [RESULTS.md](RESULTS.md) and the [current methods](README.md). Earlier unqualified statements below describe their named versions.

# Experiments and failed qualification

These are development experiments. Existing survival results are never counted
as combat results. All viewed/tuned states remain development data.

## Command-space calibration

The first 12 command tests used a 3 s command window and 1 s settling/braking
on either side. Small commands did not reliably start a gait: +0.2 m/s forward,
−0.2 m/s backward and +0.15 m/s sideways produced essentially no net movement.
+0.30 m/s sideways did move; ±1 rad/s pure turning largely twisted the planted
stance and relaxed back, instead of making an effective turn. Combining a
translation with turning produced a usable reorientation candidate.

These are measured input-response findings, not universal robot speed limits.
[Full command data](../../artifacts/combat/command-range-calibration.json.gz).

## Reactive duel calibration

Initial dodge commands were below the measured response region. A second
mistake triggered evasion during approach, allowing the coach to aim again
before committing. The corrected baseline responds to visible closing motion
and commits the attack heading; it does not inspect enemy skill phases.

With the final per-component hulls, 12 paired initial states × standing,
evading and countering defenses produced 36 runs:

| Development condition | First exchange contact | Defender counter contact |
| --- | --- | --- |
| Stand | 12/12 | — |
| Evade | 0/12 | — |
| Evade then counter | 0/12 | 10/12 |

A contact is not a qualified effective strike. Attacks against a standing duck
mostly displaced it by only millimetres. This fails the intended fighting-body
standard, even though posture was stable and the physical contacts are real.
All 36 retained runs passed the initial numerical proposal; largest contact
penetration was about **0.183 mm**. This is this experiment's result, not a
universal guarantee. [Raw rows](../../artifacts/combat/first-exchange-calibration.json.gz).

The first three initial states and a failed counter state were captured at
4 kHz, with continuous 50 fps normal-speed movies. The earlier trained motor
checkpoint R1/50 also produced three recorded miss→evade→counter-contact
sequences. These are disclosed development samples, not a success-rate claim.
One baseline and one candidate sequence were fully regenerated with zero
integration-state error, including contact logs and controller histories.

## Targeted motor training and rejection

R1 trained 150 PPO iterations after corrected 64-env/5-iteration smoke and
normalization/export parity. It started from imported walking actor weights,
not an original resumed walking PPO checkpoint. An early three-case probe of
checkpoint 50 moved a standing duck about 18 mm, 39 mm and effectively zero.
The larger test is what decides qualification.

The frozen last checkpoint (149) then received 40 new paired initial states,
four variants per state, 160 runs. Thresholds were fixed first: actual contact,
≥1 cm standing-target displacement, and final upright/support/low-speed hold.

| Variant | Contact | Effect plus stable finish | Physics proposal pass |
| --- | --- | --- | --- |
| Existing motor, standing defender | 40/40 | 0/40 | 40/40 |
| R1/149, standing defender | 27/40 | **2/40** | 40/40 |
| R1/149, evading defender | 0/40 | not an attack-effect test | 40/40 |
| R1/149, countering defender | 0/40 | 9/40 counter contacts, effectiveness unqualified | 40/40 |

The ≥80% attack-body target **failed**. R1/149 is retained as a rejected
candidate, not promoted. These data were inspected during diagnosis and are
now development data, not a final holdout. The comparison is of complete motor
skill variants including their head-command adapter, not a policy-only ablation.
Defender movement in evade/counter rows is self-actuated; it is not reported as
impact-caused displacement. [Full rejection](../../artifacts/combat/rejected-candidate-149-validation.json.gz).

R2 removed the close-range reward annuity and paid target progress only after
supported stable braking. Its corrected smoke and normalized export passed.
It still had no useful target-progress reward in the early run, so it was
stopped at iteration 55. Its saved checkpoint 50 was nevertheless exported and
measured in CPU before rejection: 40 new initial states × 4 variants, all 160
passing the numerical proposal; 27/40 standing-target contacts, **13/40**
effect-and-stable attacks and 17/40 counter contacts. This is still below the
80% body gate. Its seeds differ from R1, so this is not a matched causal
policy-improvement estimate. [R2 result](../../artifacts/combat/rejected-candidate-r2-50-validation.json.gz).

## Return to task definition

The CPU teacher reachability test found that the original 0.28 m / 0.18 m tall
free target was first detected at **1.1595 s of a 1.2 s drive**. Moving a taller
target closer created earlier contact, but baseline movement of the same
737 g target remained small. A separate, explicitly lower-friction target
probe improved one close case to about 5 mm, still below qualification.
These target variations are **lab conditions**, not changes to any duel's
friction, mass or torque.

The next motor experiment needs a measured near-contact curriculum with
explicit target-mass/friction stages, meaningful contact time and stable exit,
followed by full-mass/full-friction CPU duck tests. It must retain the current
referee and numerical thresholds. No new reward revision or self-play was
launched after this diagnosis in this round.

[Reachability](../../training/combat/target-reachability.json.gz)
· [Friction probe](../../training/combat/target-friction-reachability.json.gz)
· [Training source, budget and failures](../../training/combat/README.md).

## Gates

Physics: passed on the reported batteries, with convex approximation limits.
Motor: **UNQUALIFIED**. Gameplay: not a qualified competition. Film:
**PENDING_HUMAN_REVIEW**. No 2v2, self-play, Jev combat, ROSClaw combat receipt,
autonomous evolution or hardware deployment is claimed.

## Entry calibration after the training failure

The first actor source was frozen at `a330f0f`. A bounded explicit
`dash_start_distance_m` was then added; existing defaults remain 0.40 m.
Four development seeds at each of 0.30/0.33/0.36 m showed that the closest
entry broke all four initial evades. 0.36 m retained evasion and was frozen
for a new 40-state / 160-run validation. It produced 34/40 standing-target
contacts, **14/40** effect-and-stable attacks, 37/40 first-attack evades and
21/40 counter contacts. All 160 passed the numerical proposal; max penetration
was 0.345 mm. This is still far below the body target, not a successful motor
promotion. Different validation seeds prevent a policy-only causal comparison.

[Compact final result](../../artifacts/combat/final-summary.json)
· [Full final rows](../../artifacts/combat/final-motor-validation.json.gz)
· [Calibration measurements](../../artifacts/combat/entry-distance-measurements.json.gz).

The latest examples also expose undesired head elevation. Training removed
head-pose tracking while retaining root-upright rewards; head movement was
therefore underconstrained. This is a task/reward design gap to fix in the
next curriculum, not evidence of an intended fighting technique. No third
reward version was trained in this round.

## Continuation: from the failed lab to complete competition

The later implementation preserves the failures above. Restoring the native head objective and using a full-mass target close enough to contact produced a useful new drive motor. A separate lateral motor addressed the native sidestep's insufficient displacement. A third, matched-PD walking curriculum addressed head drift during reverse + side + turn commands after motor switches. The selected walking checkpoint remained supported and settled in all twelve CPU command/stop cases; its largest head deviation was approximately 0.323 rad. Earlier checkpoint 100 barely translated backward and was rejected in favor of checkpoint 200.

A measured zero-command exit was then needed. Requiring an exact HOME head pose stalled attacks: the contact motor's zero-command equilibrium has about 0.33 rad head-pitch offset under gravity. The final exit requires stable support, low planar speed, bounded head displacement and low measured head motion for a dwell. It does not overwrite joints or blend outputs.

A physical low-head wind-up of 0.36 s gives the defender an observable cue before contact drive. Only actual head pose and observed closing history trigger defenses. With the final motor package on forty fresh paired starts, contact drive passed **40/40** effect-and-stable attacks, median standing-target displacement **0.1178 m**; evade passed **37/40** first-attack-avoided and stable, while counter contact occurred **37/40**. All 160 rows passed numerical checks. The matched-body no-contact-drive ablation passed 0/40 attack-effect tests; this is an ablation within the new motor package, not a same-setting comparison against every upstream walking policy. The historical stronger native baseline is retained in its original report.

### Solver and platform failures

| Rejected configuration | Observed failure | Response |
| --- | --- | --- |
| libccd + circular cylinder, 4 kHz | A full team run reached 8.84 mm stage penetration at the cylinder rim | Preserve the failing capture; benchmark the exact local input tape |
| Native convex detector, 4/8/16 kHz | The local rim improved, but full matches still exposed self or duck contact failures; more substeps alone did not solve them | Reject as the final competition profile |
| Upstream walking under capped combat PD | Reverse/orbit switching could drive the head into its chassis | Train the actual command mixture under the deployed body model |
| Rolling recovery prior | A retry produced >3 mm native self penetration | Disable this unqualified primitive, keep collisions and the down count |
| libccd + fixed octagonal convex-mesh stage, 8 kHz | Both previously failing full sixty-second regression seeds passed, maxima 0.324 and 0.711 mm | Freeze for larger motor, training and held-out game checks |

No rejected match is converted to a winner. The fixed octagon is an explicitly supported game design, not a hidden barrier or runtime shape change; public boundary distance and action masks use its actual faces.

### Recovery and tactical learning

With rolling retries disabled, the new R0 battery recovered all 40 starts within the ten-second observation window; 39/40 met the three-second target. The near-edge/initial-friction R3 battery completed all 40, all within three seconds. Dynamic contact stress is reported separately, including attempted courses with no qualifying fall, numerical rejection and incomplete recovery. A coach commanded to retreat is not automatically proof that it physically cleared the recovering duck.

The 1v1 tactical actor completed actual online PPO after rule distillation. Initial four-player training was stopped before an invalid batch could update the actor; its last valid actor/critic/optimizer checkpoint was retained. The final team run resumes that checkpoint on the frozen corrected motor/physics package. Frozen cross-play compares rule, distilled and PPO tactics with the **same** motor package and keeps numerical no-contests. Training reward or an attractive clip is not a promotion criterion. Current counts, frozen weights and all final rates are in [RESULTS](RESULTS.md).
