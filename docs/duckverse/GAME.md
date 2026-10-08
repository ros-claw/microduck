# Last Duck Standing / methods and reproduction

[简体中文](GAME.zh-CN.md) · [Publication assets](PUBLICATION.md)

Four Microducks share one MuJoCo world. A seeded, occupancy-independent floor schedule gives visible warnings before free-body tiles fall. Each robot chooses a currently legal adjacent tile and walks there with existing learned motor policies. Support and posture determine elimination and victory. Character names identify instances; they are not learned personalities.

## Control and physics

- **Shared dynamics:** one `MjModel` / `MjData`, four independently indexed 14-joint runtimes, 56 position actuators, native inertias and original materials. Each robot weighs 0.73724318 kg. Torque bounds are ±0.6405 Nm. Root assistance and applied external forces are zero.
- **Floor:** 0.48 m square, 0.8 kg free-body tiles separated by 12 mm seams. World welds hold them initially. Release changes only `eq_active`; gravity and contacts produce the fall. A visible receiver catches fallen bodies 0.81 m below the arena. It never counts as valid arena support.
- **Contact:** native all-collisions geometry remains active, including native self contacts. Additional massless head/torso boxes conservatively enclose visual mesh bounds with 0.5 mm padding. These extra boxes exclude their own robot, but interact with other robots, tiles and the receiver. No duck–duck or duck–floor collision is disabled.
- **Solver:** MuJoCo 3.12.0, 4 kHz integration, Newton 80 iterations, 0.5 ms soft-contact reference, legacy libccd collision path. This particular calibration passed crowded landing cases that failed with the native path; it is not a universal solver comparison. Mesh collision uses convex geometry and finite soft-contact penetration, not exact visual triangles. See [MuJoCo collision documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection).
- **Motor layer:** official `alpha_stand.onnx` / `alpha_walking.onnx`, 50 Hz updates. After actual elimination, measured joint positions become servo hold targets so the standing policy does not fight the receiver. No root position or tile trajectory is rewritten.
- **Tactics:** current ideal simulator pose, loaded foot support, public tile state/countdown and nearby opponent poses. The actor never receives the seed, shuffled release list or future release timestamps. It selects adjacent tiles with sufficient remaining support time using distance, yaw, connectivity and crowding costs; it checks validity again while walking. This is deterministic feedback, not a newly trained survival policy or onboard vision.
- **Referee:** loss follows physical height or sustained loss of valid tile support. Victory requires exactly one remaining robot, loaded stable support and upright trunk posture. All losses are processed before declaring a result; zero survivors yields a draw. Match-over stops future scheduled releases; already falling bodies continue physically. The winner holds its learned stand policy for two more simulated seconds.

Square, perimeter ring, thick cross and clipped-corner terraces are supported. Each has steady (5 s warning / 1.1 s release spacing) and rapid (3.5 s / 0.85 s) cadences. Warning windows overlap. Seeds rotate participant names across cardinal spawn locations; all actors use the same tactical parameters and motor policies.

## Measured results

| Battery | Seeds | Physics quality pass | Winner / draw |
| --- | --- | --- | --- |
| Two robots | 501–510 | 10/10 | 4 / 6 |
| Four robots | 601–612 | 12/12 | 4 / 8 |
| Frozen heldout | 20000–20031 | 32/32 | 12 / 20 |

Quality requires finite state, no time reset or solver warning, no applied external forces, bounded motor torque, maximum contact penetration ≤3 mm and penetrating-contact P99 ≤1.5 mm. All 54 final matches pass. Worst observed penetration is **1.797736 mm**; heldout penetrating-contact P99 is at most **0.001 mm**. Quality pass is a stability/contact criterion, not game-winning success. Twenty heldout games draw because support can expire with no legal adjacent escape. The baseline does not solve that limitation.

Heldout games include eight layout/cadence combinations, four spawn rotations each. Twenty-two of 32 have duck–duck contact. Winners by identity: lavender 1, cream 2, sky 2, graphite 7. This small sample does not establish equal win rates or a superior personality. Full outcomes, per-run contact extrema and performance remain in [QA summary](../../artifacts/duckverse-game/qa-summary.json) and the three battery reports.

Four-robot heldout runs average **0.669× simulated time / wall time** without full raw recording. ONNX batch P95 is at most 1.80 ms across four robots. Full-rate recording and rendering are offline; video playback speed is unrelated to simulation throughput. [Render benchmark](../../artifacts/duckverse-game/render-performance.json) separates rendering from physics.

## Reproduction

Use the release checkout and pinned native assets described in the root README. MuJoCo 3.12.0 is required for the archived strict replay; policies and source files are hash-bound.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_duckverse.py \
  --seed 101 --players 4 --layout square --cadence steady --capture \
  --out artifacts/duckverse-game/my-seed101

.venv/bin/python scripts/replay_duckverse.py artifacts/duckverse-game/my-seed101

# New, non-existing output directories; complete fixed batteries
.venv/bin/python scripts/eval_duckverse.py --suite two --out artifacts/duckverse-game/my-two
.venv/bin/python scripts/eval_duckverse.py --suite four --out artifacts/duckverse-game/my-four
.venv/bin/python scripts/eval_duckverse.py --suite heldout --out artifacts/duckverse-game/my-heldout

.venv/bin/python scripts/render_duckverse.py \
  --run artifacts/duckverse-game/my-seed101 --out out/duckverse_local.mp4
```

Full-rate capture stores a binary scene, integration states and controls at every 0.25 ms step, all contacts with signed distance/force/position/normal, public decisions at 50 Hz, constraint/referee events and audit digests. A capture uses substantial disk space. Evaluation without `--capture` retains decisions and audit metrics while omitting heavy raw arrays/contact streams.

Strict replay regenerates the entire closed loop and compares model bytes, states, controls, every decision/contact row, constraints, events and referee output exactly. Selected seeds **101, 20008, 20017** replay with zero state/control error. Live native ROSClaw captures have separate audited receipt linkage. Gzip wrapper timestamps can differ across identical runs; semantic contact rows are compared rather than wrapper-byte equality. This verifies simulator consistency, not hardware or an independent dynamics model.

## Boundaries

No Jev call, new motor training, autonomous evolution or Darwin promotion occurs in this game. Optional personality preferences, multi-step route search, trained recovery, learned survival tactics and a live game UI remain future work. Earlier single-duck DG-01/02 and previous rope/parkour demos remain frozen and retain their own limitations. [Failed contact experiments](DEVELOPMENT.md) are preserved separately from final qualification.
