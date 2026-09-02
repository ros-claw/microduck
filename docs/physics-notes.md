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

