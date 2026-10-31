# Duck Fu: tactical 2v2

**English** · [简体中文](README.zh-CN.md) · [Main project](../../../../README.md)

[![Tactical 2v2](media/cover.jpg)](media/tactical_en.mp4)

[English film — 63.88 s, 1080p, 50 fps](media/tactical_en.mp4). This edited showcase separates **Bout A** and **Bout B**, labels every slow replay and retains the real draw. Commentary, contact-triggered sound and the rhythm bed are post-produced. They are not simulated microphone recordings. Native robot materials are retained.

## Method

The new opt-in `controller_revision: 3` coordinates public-state roles over the existing frozen learned walking, lateral-step, contact-drive and recovery motors. There is **no new tactical neural network, Jev call or motor-training run in this round**.

The counter style pairs a fighter drawing pressure with a wing moving around an opponent. Pressure style keeps separate lanes. Exposed or isolated opponents invite flank pressure; the second enemy must still be screened. An ally near the edge requests withdrawal and cover. Target and role changes take effect at maneuver boundaries; displaying a role describes the executed assignment, not a certified tactical success.

Continuous bounded walking inputs replace coarse turning commands. A public head pose or approaching velocity can trigger a lateral step. Measured own support, upright state, joint motion and motor history govern the handoff to turning and a return bump. Committed drives have longer contact commitment than short bumps. Physical collisions remain enabled between opponents and teammates.

An exposed side is geometric: a solver-qualified response with the attacker over 60 degrees off the target's facing direction **at attack launch**. It is not a damage multiplier or a special weak-point collider. A bump is not a knockout; actual ring-out or the existing down count decides eliminations. No root pose/velocity edits, external pushes or health-based knockback are introduced.

## Results and limits

The frozen actor is `7cafc1a`. Four new base seeds `200001–200004`, each with a side swap, produce eight initial-state pairs; swaps are correlated, not 16 independent seeds. Both revisions use the same pinned motors, optional learned recovery and physics settings. This compares a combined controller change, not an isolated ablation.

| Across eight bouts per revision | Revision 2 | Revision 3 |
| --- | ---: | ---: |
| Solver-qualified response events | 92 | 192 |
| Responses from a flank at launch | 43 | 65 |
| Upright, uninterrupted evade/miss/counter sequences | 1 | 3 |
| Own-team nearby supported ally at a response | 67 | 123 |
| DOWN events | 20 | 19 |
| Ring-outs | 8 | 4 |
| Wins / timeout draws | 1 / 7 | 0 / 8 |

All 16 bouts pass the existing numerical gates; maximum penetration is **1.7254 mm** against the unchanged 3 mm ceiling. Both selected 60-second captures regenerate **480,000 integration states each with zero error**, identical contacts and identical referee output. The captured validation bout's control/body/phase arrays also equal its original non-capture run.

The historical counter detector counts four new-controller sequences, but one contains an intervening fall. A **post-hoc stricter audit** excludes recovery, DOWN or elimination and requires upright ≥0.94 throughout. It leaves three sequences. The film's selected faster sequence stays above 0.996 upright. This clarification does not alter actors, rewards or results. Nearby support is a geometry proxy, not proof that a teammate caused or saved a win. No improved win rate or completed counter-speed target is claimed.

Local 0.4-second fixed-input timestep probes remain finite and numerically accepted at 0.125/0.0625 ms. The counter window differs by **0.86 mm** in final root position and **1.10%** in duck impulse; the edge window differs by **4.01 mm / 19.25%**. The edge scramble remains timestep-sensitive; these probes do not establish a timestep-invariant winner or hardware fidelity. There are **231 passing tests**.

## Film selection and reproduction

Bout A is seed `200003`, unswapped, selected for two uninterrupted upright counters; the faster one is highlighted. Bout B is development seed `190002`, unswapped, selected for withdrawal cover, a measured side response and real ring-outs of ANCHOR at 25.40 s and SHADE at 25.60 s. Both sources end in a timeout draw. This is entertainment selection, not an estimate of typical play or a stitched fictional bout. `NEW BOUT`, A/B source labels and source clocks disclose the edit. Cameras and the live-target minimap only read recorded states.

Use the [locked environment and assets](../../README.md), from the repository root, with new output directories:

```bash
export MICRODUCK_ROOT="$PWD/.assets"
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_combat.py \
  --config configs/combat/v2/selected_tactical_counter.yaml --seed 200003 \
  --capture --out artifacts/combat/local/my-counter
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_combat.py \
  --config configs/combat/v2/selected_tactical_edge.yaml --seed 190002 \
  --capture --out artifacts/combat/local/my-edge
PYTHONPATH=src .venv/bin/python scripts/build_combat_tactical_film.py \
  --counter-run artifacts/combat/local/my-counter \
  --edge-run artifacts/combat/local/my-edge --out out/my-tactical-plan.json
PYTHONPATH=src .venv/bin/python scripts/render_combat_film.py \
  --plan out/my-tactical-plan.json --out out/my-tactical-silent.mp4
PYTHONPATH=src .venv/bin/python scripts/sound_combat_film.py \
  --plan out/my-tactical-plan.json --silent-video out/my-tactical-silent.mp4 \
  --tts .venv-commentary/bin/edge-tts --out out/my-tactical-en.mp4
```

Optional speech dependencies are pinned in `requirements-commentary.txt`, installed in a separate environment. Rendering works without speech or a network connection. Rebuild plans after relocating sources; archived plans contain original absolute paths. The complete silent source captures remain local and can be regenerated, rather than committing multi-gigabyte integration arrays.

[Paired results, replay and physics evidence](../../../../artifacts/combat/v2/tactics/) · [Source controller](../../../../src/microduck_lab/combat/tactics_v3.py) · [English subtitles](media/tactical_en.en.srt). This is a development sample awaiting viewing acceptance. No Release, tag or merge is created.
