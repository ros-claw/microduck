# Circus Director: first-stage implementation

The design input is `~/workspace/tennis/更酷的demo尝试.md`. Work remains in the MicroDuck repository; the TennisForge project is not modified.

## Scope and acceptance

Latest follow-up: [formation geometry, timing, and independently validated static duo](CIRCUS_GEOMETRY_AND_TIMING.md). `rehearse --config configs/circus/duo_matched.json` runs the selected configuration; `--config` also accepts a saved trial result and uses its `config` field.

**Measured status:** [September 17 results](CIRCUS_RESULTS_2026-09-17.md). Full relay and the 30% increase remain unpassed.

The first stage follows the document's priority: a four-duck **5 / 3 / 5 relay**, plus a measured **30% speed challenge**. Double-under, travelling rope, seesaw, assisted jumping, and cooperative transport remain subsequent research stages. The document's example success percentages and frequencies are illustrative, not measurements.

The existing three-duck result remains a regression baseline. A fourth robot standing in the scene, two robots spawned inside a rope, or an edited sequence of independent clips is not a relay success.

A complete relay requires, in one uninterrupted physical rollout:

1. Sky clears five consecutive full rope cycles.
2. Graphite physically moves from outside into the moving-rope region, without teleporting, disabling contacts, or tripping either jumper.
3. Both robots clear three consecutive matching cycles, each satisfying the original sole-clearance and landing rules.
4. Sky physically leaves the region while Graphite continues.
5. Graphite clears five consecutive cycles.
6. A servo-driven bow and a supported upright return complete the finale.

Stage transitions consume independently observed contacts, positions, and cycle records. A timer cannot mark a skip successful. Entry/exit faults latch failure; failed cycles reset the consecutive count. Static-duo tests are explicitly labeled as feasibility probes and cannot promote the relay skill.

## Implementation

- `sim/classic_rope.py`: routes rope collisions to every jumper, including Graphite, before model compilation. Reference handle locations translate with custom turner positions when widening the scene. Rope–turner exclusions remain inherited from the baseline; rope–jumper and inter-jumper contacts stay enabled.
- `sim/skip_metrics.py`: the same physical evaluator can audit either jumper by name, retaining the baseline criteria.
- `demos/honest_skip.py`: optional additional robots and a policy-step callback; the standard three-duck invocation retains its defaults.
- `circus/mission.py`: a declared, bounded bilingual grammar converts the canonical relay and percentage-speed request into structured plans. This is not an unrestricted natural-language or LLM planner.
- `circus/relay.py`: the event-driven 5 / 3 / 5 stage gate.
- `circus/trial.py`: real MuJoCo rehearsals, phase-gated walking or hopping entry, separate jumper audits, inter-jumper collision tracking, trajectory samples, and failure evidence.
- `circus/promotion.py`: a holdout gate requiring at least four distinct seeds, disjoint from practice seeds, a common protocol, and successful task-specific evidence.
- `scripts/circus_director.py`: plan, rehearse, and parameter-practice commands; appends real trial receipts and creates `champion.json` only after the holdout gate passes.

The current Practice backend searches physical controller/layout parameters. Entry probes include a goal-directed hopping controller through the existing centering-command interface; this changes velocity requests, not body poses. Safe waiting and exit regions use the whole robot collision geometry and a conservative swept-rope bound. It does **not** perform automatic PPO retraining. A failed search reports `NEEDS_LEARNING`; it must not manufacture a learned skill or turn a training score into a champion. A champion for one subskill never certifies the complete relay.

## Commands

Use the installation and asset-root instructions from the main README. All commands run from this repository's root.

```bash
.venv/bin/python scripts/circus_director.py plan \
  'Sky 先跳 5 下，Graphite 进去，两只一起跳 3 下，Sky 退出，Graphite 再跳 5 下。' \
  --output artifacts/local/relay-plan.json

.venv/bin/python scripts/circus_director.py plan 'faster by 30%' \
  --output artifacts/local/speed-plan.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/circus_director.py rehearse \
  --kind relay --plan artifacts/local/relay-plan.json --seconds 30 --output-dir artifacts/local/relay

# Diagnostic video; failures remain visible and are not labeled a success.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/circus_director.py rehearse --kind entry \
  --dx 0 --seconds 25 --video out/entry-diagnostic.mp4 \
  --output-dir artifacts/local/entry-video

# Use a fresh directory; search stops advancing cadence at a failed level.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/circus_director.py practice \
  --kind speed --increase-percent 30 --seeds 0,1 \
  --holdout-seeds 701,702,703,704 --output-dir artifacts/local/speed-practice
```

Rehearsals preserve a source snapshot keyed by the loaded protocol hash, and refuse to overwrite an existing evidence file. `practice` uses held-out seeds only after locking a candidate. Receipts contain the evidence file hash and evaluator protocol hash. Rehearsals record policy hashes, layout, seed, physical verdicts, and actual achieved cadence. Requesting 4.03 Hz without actually sustaining it cannot pass the speed challenge. A validated lower curriculum level does not imply the full requested increase was achieved.

## Research context

[Marope](https://arxiv.org/abs/2606.08064) uses learned decentralized rope manipulation and centralized scheduling for cooperative long-rope skipping. [Co-jump](https://arxiv.org/abs/2602.10514) studies mechanically coupled cooperative jumping by quadrupeds. These are references motivating later work, not implementations or results reproduced by this project.

## Subsequent gates

| Stage | Required evidence before calling it a skill |
| --- | --- |
| Moving-rope entry and exit | Real boundary crossing, continuous contacts enabled, successful landing and continuation |
| Double-under | Two full underfoot passages in a single uninterrupted airborne interval, followed by supported landing |
| Travelling rope | Measured group translation with valid skips and bounded relative formation error |
| Seesaw | Passive physical plank, measured balancing torques from robot contacts, successful supported crossing |
| Assisted jump | Matched solo control and cooperative trials showing a measured height gain |
| Free-form director | Capability-grounded planning, executable skill contracts, real learning backend, independent full-show rehearsal |

None of these future gates is certified by the existing three-duck score.
