# Duck Fu — duel development lab

**English** · [简体中文](README.zh-CN.md)

Two native Microducks share a fixed MuJoCo world. A disclosed rule-based coach
commits to a dash; a public-state defender evades, faces it again and attempts
a counter. Existing learned motors and a newly trained candidate move the
joints. **This is an early lab, not a qualified fighting game.**

[Earlier R1/50: 10 s continuous motor candidate](media/duel-candidate-01.mp4)
· [Initial state 2](media/duel-candidate-02.mp4)
· [Initial state 3](media/duel-candidate-03.mp4)
· [Existing-motor baseline](media/duel-greybox.mp4)
· [Failed counter](media/duel-failure.mp4)
· [Actual collision-hull view](media/duel-collision.mp4).

All samples are silent, no-FX and have a fixed unobstructed camera. Labels
identify coach/defender roles; there is no network model choosing reflexes and
no script moving roots. The candidate examples use the **R1/50** motor and
previously viewed development states. They are not a success-rate claim.

Current R2/50, 0.36 m entry: [continuous sample 1](media/near-duel-01.mp4)
· [sample 2](media/near-duel-02.mp4) · [sample 3](media/near-duel-03.mp4)
· [missing-counter failure](media/near-duel-04.mp4).
These are the first three completed counter-contact sequences and first missing
counter from the frozen near-entry validation, selected for diagnosis. They
are not an unselected sample or qualification proof.

## Method and evidence

The new combat module leaves frozen rope/parkour/survival actors unchanged.
Motor policies receive the standard 61D observation and produce 14 joint
position offsets at 50 Hz. Local maneuver logic commits to attack direction,
responds to visible relative motion and hands control to recovery when needed.
All ducks decide from the same snapshot before physics advances at 4 kHz.

Duck–duck collision uses **70 massless visual-component convex hulls per duck**,
with separate masks to avoid duplicate native/proxy responses. Original native
self/ground contacts, inertia, materials and ±0.6405 Nm torque caps remain.
This fixes important coverage gaps without adding broad invisible AABBs, but
component convexification still approximates concave surfaces. Contacts,
impulses and penetration categories are audited every physics step.

In 12 paired development states, standing defenders were contacted 12/12;
evading defenders avoided the first attack 12/12. Counter contacts occurred
10/12 in the existing-motor baseline. Contact alone does not establish useful
attack strength. The new final R1/149 motor failed the larger 40-state body
criterion: **2/40** effect-and-stable attacks, so it is rejected. R2/50 then passed 13/40 on 40 different new states, still below 80%.
Near-entry calibration followed by 40 further new states gave 14/40
effect-and-stable attacks, 34/40 standing-target contacts, 37/40 successful
first-attack evades and 21/40 counter contacts. The body gate still failed.
Two reward versions were investigated; the target course needs better calibrated contact
time and curriculum. **Motor Gate: UNQUALIFIED. Film Gate: PENDING_HUMAN_REVIEW.**

The baseline and candidate selected clips each regenerated 40,000 physical
steps with zero integration-state error and identical contact/controller logs.
The accepted survival baseline also regenerated all 401,126 steps with zero
error. Final first-round test suite: 163 passed; separate training contract tests:
2 passed. No four-duck combat, self-play, Jev combat, combat ROSClaw receipts,
real hardware or autonomous evolution is claimed.

[Source audit](00_source_audit.md) · [Scope/rules](01_scope_and_rules.md)
· [Physics/coverage](02_physics_contract.md) · [Experiments/failures](03_experiments.md)
· [Training](../../training/combat/README.md) · [Capabilities](../../configs/combat/capabilities.yaml).

## Reproduce a recorded candidate

Install the delivery dependencies and pinned upstream assets from the main
README. The optional audit scripts need `pip install -e '.[dev,combat-audit]'`.
From this checkout's root, use a fresh output directory:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_combat.py \
  --config configs/combat/reference_01.yaml --brain rule --seed 101 --capture \
  --out artifacts/combat/local/my-reference
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_combat.py \
  --run artifacts/combat/local/my-reference
PYTHONPATH=src .venv/bin/python scripts/render_combat.py \
  --run artifacts/combat/local/my-reference --out out/my-duckfu.mp4
```

The rule brain is the tactical controller; this command still executes the
explicit candidate motor recorded in the configuration. `--brain learned` or
`jev` fails because those tactical modes are not implemented/qualified.
For initial states 2/3 use their reference YAML and seeds 102/103.

`combat_probe.py --config configs/combat/duel_lab.yaml --suite first_exchange
--count 12 --jobs 4 --out <fresh-dir>` runs the 36-case development comparison.
`eval_combat.py --protocol configs/combat/eval_protocol.yaml --out <fresh-dir>`
repeats the **failed R1/149 motor protocol**, not a game win-rate evaluation.
Historical R1 reference captures use actor source snapshot
`a330f0f544e4cb0ae8dc4e62ff64a1ca3e2c2e7c`. Current near-reference captures
use the latest source and `near_reference_01.yaml` / seed 4002 (then
`near_reference_02.yaml` / 4005 and `near_reference_03.yaml` / 4006).
Full state/contact captures stay local; videos, compact evidence, configurations
and candidate weights are tracked. New captures are replayed by advancing the
actual controllers and physics, not by merely setting recorded poses.

No Release or tag is created for this work. Pollen Robotics assets and model
terms remain separate from project code; see the repository attribution.

Large raw reports/logs are losslessly compressed; the [compact final summary](../../artifacts/combat/final-summary.json) is directly readable. Source snapshots preserve old replay versions.

Replay source versions: R1 films use `a330f0f544e4cb0ae8dc4e62ff64a1ca3e2c2e7c`;
the R2 near-entry films use `c2776f564f4d46ac53d92493518975c0fb9a9a23`.
The final audit adds world coordinates to worst-depth records. Its four
recaptures have bit-identical integration states, motor histories, contacts
and decisions to those R2 films, and the final implementation was independently
replayed again. [Equivalence](../../artifacts/combat/final-capture-equivalence.json)
· [Final replay](../../artifacts/combat/final-reference-replay.json)
· [Worst-contact positions](../../artifacts/combat/worst-contact-positions.json).
Use the matching historical source worktree/snapshot when verifying an older
manifest; the verifier intentionally rejects changed actors.
