# DG-02: public warnings, seeded release, contact referee / 地板调度与物理裁判

**Delivered: one-duck scheduled physics and strict semantic replay.** The existing DG-01 calibration is unchanged. This milestone does not implement a four-duck tournament, robust general survivor, Jev, ROSClaw chat or Darwin.

[English paired demonstration](https://github.com/ros-claw/microduck/releases/download/duckverse-dg02-schedule-2026-10-08/duckverse_dg02_en.mp4) · [Physics/replay evidence](https://github.com/ros-claw/microduck/releases/download/duckverse-dg02-schedule-2026-10-08/duckverse_dg02_evidence.tar.gz) · [Frozen paired measurements](../../artifacts/duckverse-dg02/final/summary.json)

[![Warning response and hold control](../../out/duckverse_dg02_en.jpg)](https://github.com/ros-claw/microduck/releases/download/duckverse-dg02-schedule-2026-10-08/duckverse_dg02_en.mp4)

## Implemented behavior

Each free-body tile progresses monotonically **LOCKED → WARNING → RELEASED → FALLING → LOST**. Warning precedes release by four seconds; release changes only equality activity. FALLING requires actual z<-0.027 m and vz<-0.03 m/s; LOST requires z<-0.15 m. These are grid-height-specific calibration thresholds, not universal physics constants. Warning is a recorded, renderer-visible state; it changes no contact geometry or support.

The scheduler freezes a seed-based permutation before physics starts. Tile 1 warns first for this one-body calibration, followed by the other eight in seeded order. Warnings begin at 2 s and repeat every 5 s; there is at most one warning at a time. The plan never depends on the duck's position or desired video result. This is reproducible calibration, **not spawn-balanced tournament scheduling or a proved playable layout generator**.

A small deterministic responder reads current pose, measured supporting tiles, current visible tile states and warning age. It selects an adjacent LOCKED tile by distance/yaw cost, then commands the existing 50 Hz learned walking policy. Approach speed decreases with distance to avoid overshoot. It does not receive seed, future order, future release time, oracle TTL or evaluator fields. This is privileged simulator-state feedback, not a vision demonstration.

Elimination requires torso/root z<-0.12 m **and no active floor support continuously for 0.3 s**. Contacts qualify as support only when the native foot receives positive normal force with upward normal against a still-constrained tile. Short airborne phases and supported recovery are not automatically eliminated. Evidence includes timestamp, body ID, z/vz, supporting tile IDs and contact-step reference. The receiver and already released tiles never count as arena support.

The referee processes all bodies before resolving outcome. Unit tests cover simultaneous DRAW and require a final survivor to retain valid support for 0.2 s before WINNER; a last body still falling cannot be prematurely crowned. **Only single-body physics has been exercised here**: its terminal outcomes are ELIMINATED or TIME_LIMIT, never a manufactured champion. Multi-body winner rules still need shared-world physical calibration at DG-03.

## Frozen results

Seeds **11–20**, paired current-warning response vs hold, identical world/initial perturbation/schedule within each seed. Horizon 14 s, robot joint perturbation ±0.005 rad. This is a development comparison; the responder and timing were tuned on seed 11.

| Controller | Alive at 14 s | Physically eliminated | Physics quality passes | Strict input/semantic replays |
| --- | ---: | ---: | ---: | ---: |
| Current-warning responder | 10/10 | 0/10 | 10/10 | 10/10 |
| Hold position | 0/10 | 10/10 | 10/10 | 10/10 |

Maximum penetration across all 20 runs: **2.764 mm**. Zero solver warnings/time resets, finite recorded states, actuator torque ≤0.6405 Nm, no external applied force. No heldout win rate, long-term survivor reliability or AI superiority claim.

Selected seed 11 responds twice: warning on tile 1 → move to tile 4; warning on tile 4 → move to tile 7. Both evacuated tiles then really fall. In its paired HOLD run, the original support falls and the referee eliminates the duck. The film explicitly labels **two separate runs A/B**, not one edited episode. The hold release has a 0.5× detail replay from recorded states without interpolation. Both views show the same state; audio is absent.

An additional **48 s all-tiles-collapse run**, seed 11, completes all nine WARNING/RELEASED/FALLING/LOST sequences. It ends ELIMINATED at **46.521 s**, torso z=-0.636 m with no arena support. Maximum penetration **2.007 mm**, zero solver warnings, 96,000 physics steps and all controller decisions/events/outcome replayed exactly. The duck remains on an isolated final tile with no legal adjacent escape; this is a real failure, not a winning game.

Reference suite: **97 passed / 0 failed / 0 skipped**. The old 51-file V1 freeze is intact. Original DG-01 source/runtime/scripts and old skipping/parkour policies are unchanged. New code passes Ruff.

## Replay and observation boundaries

The verifier initializes the recorded full integration state once, regenerates the seeded scheduler, and advances MuJoCo using recorded motor controls. It does **not** apply per-frame pose placement or blindly trust recorded equality arrays: release events must generate the same actual eq_active state. It independently recomputes directional foot support, public observations, deterministic responder choices, tile transitions, referee events and terminal outcome. Contacts, root qpos/qvel and event lists agree exactly in the frozen runs.

Model, trajectory, contact stream and compressed decision log are mandatory and hash-checked. Missing files/hash entries, altered outcomes, inconsistent schedules, extra/missing frames or mismatched support fail verification. Tests deliberately corrupt outcome and remove a required replay hash. Public decision payloads for all 20 runs were also scanned for oracle leakage.

Per-step energy is recorded in the trajectory for inspection; no universal energy-jump bound is claimed. Contact samples are pre-integration; torso/tile diagnostics and elimination evaluation are at t+dt with the just-evaluated contact graph (0.5 ms alignment). Standard dt=0.0005 s, policy 50 Hz, MuJoCo 3.12.0; collision parameters/native assets/torque limits inherit DG-01.

## Commands actually used

Use the release checkout/bootstrap instructions in the root README. Every output directory must be new.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/eval_tile_survival.py \
  --out artifacts/duckverse-dg02/local-paired

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_tile_survival.py \
  --seed 11 --duration 48 --out artifacts/duckverse-dg02/local-full

OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/replay_tile_survival.py \
  --run artifacts/duckverse-dg02/local-full

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/render_tile_survival.py \
  --response artifacts/duckverse-dg02/local-paired/warning-11 \
  --hold artifacts/duckverse-dg02/local-paired/hold-11 \
  --out out/duckverse_local_dg02.mp4

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
.venv/bin/python scripts/freeze_neon_v1.py
```

The release includes full input trajectories for the filmed pair and the 48 s full collapse, one shared model, all 20 paired contact/decision streams, audit/replay reports, source archive and manifests. Other paired state trajectories remain local and are reproducible using the frozen source, seed and official policy digests. Large binary files are not committed to Git.

## Failures and limits

- `probe11b`: fixed 0.45 command until reaching centre overshoots by about 0.37 m after stopping; apparent survival partly comes from drifting onto another tile. Rejected as controller evidence. Distance-dependent approach speed fixes the selected two straight transitions. [Original capture report](../../artifacts/duckverse-dg02/probe11b/audit-summary.json) retained.
- Early startup API/log-writing errors are retained as explicit STARTUP_FAILED records, not counted as physical trials.
- The short cohort primarily validates the first warning response. It does not qualify all turn directions, crowded routing, recovery/jump or arbitrary no-safe-option cases. The full-collapse final isolation ends in genuine failure.
- Actor observations use simulation truth. No new Jev calls, ROSClaw Body instances, receipts, Practice or training updates are supplied.
- Long episode and paired seeds are development evidence, not a tournament benchmark. Native collision primitives/convex meshes retain DG-01 approximation limits and nonzero numerical penetration.

**Next Go:** DG-03 two-duck shared-world initialization, independently bound runtime indices, body collisions and profiling. **No-Go:** claiming a four-duck autonomous tournament or cinematic completed game before those tests.
