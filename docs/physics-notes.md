"""Physics findings from building the 3-duck rope skipping demo.

All numbers measured in this repo's simulation (MuJoCo 3.12, official
robot_allcollisions.xml, official ONNX policies).

## Rope turning physics at Microduck scale

- Microduck: 25 cm, ~0.8 kg, 14 XL330 servos (kp≈0.55, ±0.96 Nm), head ≈ 38%
  of body mass.
- A rotating rope loop of belly radius R needs ω²R ≳ g to stay taut at the top.
  For R=0.15–0.2 m that means f ≥ 1.3–1.6 Hz. Human jump ropes run at R≈1.2 m,
  f≈2 Hz — at duck scale gravity is relatively 6× stronger.
- Vertical mouth pumping saturates the neck/head pitch servos: τ ≈ I·α ≈
  0.8 Nm at ±25 mm / 1.25 Hz, vs the 0.96 Nm limit. Horizontal (yaw) motion is
  cheap; vertical is not.
- Standing turners: vigorous lateral head sway (≥1 cm at ~1 Hz) topples the
  stand policy within ~4 s. Body-pose channel (trained ±30 mm) does not track
  1 Hz commands (policy is quasi-static) — zero measurable effect.
- Sitting turners (sitstand policy, tripod base) are 100% stable under head
  sway, but mouths sit at z≈0.165 m (0.19 craned) — small loop, marginal ω²R.
- A duck jaw latched to a rope is an energy sink: the position servo resists
  rope-induced head motion and dissipates a tossed rotation within ~1 s.

⇒ Design: sitting turners + declared coach wind-up/rate regulation
  (skills/rope_coach.py) + mouths tracking measured rope phase (real
  actuation, load-bearing). Gives 5–10 s windows of clean rotation at
  1.35–1.5 Hz with 100% turner stability — plenty per practice episode.

## Jump (procedural)

- Scripted crouch→extend→tuck→land through the leg position actuators gives
  2.5–3.3 cm feet clearance with the stand policy holding balance afterward.
  Rope belly grazes at 0–1 cm, so 2.5 cm clears.
- Trip = rope under feet while grounded. After a trip the jumper usually stays
  down: no standup policy is released in the official ONNX set (StandUp exists
  as a training task only) — a documented capability gap. Hence the session
  pauses the rope when the jumper is down >1.2 s (turner etiquette), and
  Darwin's safety gate uses recovery-aware max-down-time.

## MuJoCo gotchas hit (regression tests cover these)

- `composite type="cable"` pins its first/last bodies to the world — latching
  it to a mouth silently pins the head to the world anchor.
- `connect` equality auto-computes the body2 anchor at qpos0; attached bodies
  sit at the origin at qpos0, so the latch "remembers" a wrong offset.
  Fix: set `eq_data[3:6] = 0` post-compile (anchor = rope endpoint origin).
- Free-joint bodies ignore their frame offset under spec attach; set
  `data.qpos` explicitly at spawn.
- Rope ball-joint damping 0.02–0.06 is ~10³× overdamped for a gram-scale rope;
  use 1e-4.
- `actuator_trnid`-based qpos/dof indexing is required for namespaced models.
"""

## Jump / skipping findings (the honest part)

- The scripted tuck-hop reaches ~2.5 cm foot clearance for ~0.1 s, peaking
  ~0.6 s after trigger (crouch 0.18 + extend 0.07 + flight 0.24 + …). The lead
  time must be ≈0.6 s — an early lead grid (0.28–0.44 s) was systematically
  miscalibrated until we profiled the maneuver.
- The position servos (kp 0.55, ±0.96 Nm) cannot ballistically launch the body:
  trunk apex rises <1 cm. Real rope skipping needs a trained jump policy.
- The swing's belly does NOT reach the hanging arc bottom: dynamic shortening
  keeps it ~4 cm up. Floor-grazing requires the rope long enough that the
  resting belly lies on the floor (L≥0.62 at mouth z 0.165) — but then the
  resting rope won't pump up. We run "swing" with L=0.55.
- Full rotation mode additionally suffers rope coil-up around the craned heads
  (no swivel on the mouths) — guarded by the coil detector + drop-regrab-reset.
- Judge honesty: verdicts need a ±0.12 s windowed feet-peak (50 Hz sampling
  misses the hop apex), real rope-contact evidence, and an upright check 0.5 s
  after the pass. The agent context (rhythm model) never sees oracle state.


---

## CORRECTION (2026-09-04): classic overhead rope IS physically feasible

The earlier draft concluded the full overhead loop was geometrically impossible
("mouth at 19 cm < duck 25 cm"). That was WRONG — it confused the endpoint
height with the belly's rotation radius. A loop rotating about the endpoint
axis with belly radius R reaches top ≈ h_endpoint + R. For R ≈ 0.17 m the top
is ≈ 0.35 m > the 25 cm duck. The 0904 discussion caught this; the Oracle
Rope Feasibility Lab (CR-02) then PROVED it in physics.

**Oracle experiment** (`scripts/oracle_rope_lab.py`, no ducks, two ideal mocap
endpoints at h=18 cm, sep=0.50 m, `mujoco.elasticity.cable` rope):
- The rope forms a stable full-rotation loop: **top ≈ 35 cm, bottom ≈ 1–2 cm,
  sustained 8+ revolutions**, at multiple configs (L∈[0.78,0.86],
  density∈[100,400]). FEASIBLE = YES.
- Working reference: L=0.82 m, density=400 (≈3.7 g rope), bend=1e3.

**The mechanism that was missing** (not "gravity too strong"):
1. **Swing-up is the barrier.** A naive fixed-frequency circle drive NEVER
   inflates the loop (belly stays at R≈5 cm no matter the amplitude/frequency).
   A phase-tracking drive (endpoints lead the measured belly angle by +90°,
   amplitude ramping) pumps the pendulum over the top and captures it into
   rotation — like pumping a swing.
2. **Endpoint phase difference Δφ**: in-phase (Δφ=0) is correct for a planar
   loop; Δφ=180° completely fails (0 revolutions, the ends fight). Measured.
3. **Air drag was never the issue** — MuJoCo defaults density=0/viscosity=0
   (no fluid forces). Dissipation is joint/bend damping + contacts.
4. The rope self-selects a rotation speed set by drive amplitude × rope mass;
   a speed governor (lead-angle → 0 at target ω) is needed to hold a
   hopper-friendly ~1.2–1.5 Hz instead of free-running to 5–8 Hz.

**Composite-cable gotcha (CR-01)**: the cable's first body is welded to the
world (`B_first` has no joint). To hold BOTH ends on mocap carriers, wrap the
composite in a freejoint body (which needs a small geom — a massless free body
falls through the connect constraint) and connect wrapper→carrierA,
B_last→carrierB.

## CR-03: jump-rope handle + wrench budget (feasible)

A lightweight handle welded into the beak (jaw_soft) — the duck holds the
handle, not the cord (like a real skipper). The rope end connects to the handle
tip via a point-connect = a passive swivel (no coiling).

Measured:
- **Leverage works**: 0.4 rad of head pitch → 4.4 cm vertical tip throw; 0.4 rad
  head yaw → 4.3 cm lateral. The rope is driven by cheap head ROTATION, not
  translation of the 300 g head. (Head-neutral standing pose; NOTE the sweep
  degenerates if the head is craned fully down — the down-pointing handle's tip
  approaches the rotation axis.)
- **Wrench budget PASSES easily**: the rope's reaction force at the endpoint
  during sustained oracle rotation is RMS ~1.43 N / max ~3.0 N (measured from
  the connect constraint force). At a 5 cm handle + head lever (~0.13 m from
  the neck joints), that's ~0.07 Nm RMS / ~0.18 Nm peak at the neck — ~15% of
  the XL330 envelope (0.96 Nm). The rope load is NOT the constraint.

The remaining blocker is NOT torque or geometry — it's that a hand-tuned
IK/servo circle-tracker can't drive the rope while keeping a standing duck
balanced (the 0904 doc's §3/§6 point: rope turning must be a LEARNED
whole-body policy — that's CR-05, the real remaining research).

## CR-05 progress (turner RL + classic skip world) — honest state

- turner_motion RL policy TRAINED (Mjlab-Turner-Flat-MicroDuck): the duck
  learns to swing the mouth-held handle tip along a target circle while
  balancing. turner_circle_track 0.0003 → 0.96 (normalized), zero falls.
  The no-rope precursor to the learned rope turner.
- build_classic_world: elastic cable + 3 ducks + mocap carriers, rope↔turner
  contact excluded (turners brace; only the jumper's contact is scored). The
  phase-tracking swing-up spins a real overhead loop over the middle duck
  (rendered, verified visually).
- REMAINING BLOCKER for the clean classic skip: the jumper's hop drifts forward
  out of the grazing zone, and the hop apex/timing needs the RL rope_hop v2
  (in-place periodic hop, CR-06) + a PLL + a rotation-speed governor (the
  free-running phase-tracking drive overspins to ~2.3 Hz; the naive governor
  broke the swing-up — needs a proper swing-up→capture→govern state machine).
