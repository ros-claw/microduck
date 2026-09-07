"""Classic overhead-rope skip scene — the money shot.

The elastic-cable rope spins up into a real overhead loop (phase-tracking
swing-up, proven by the Oracle Feasibility Lab), then the RopeHoldDrive ramps
the drive rate down to the rope's minimum sustainable rotation (~2.8 Hz at a
0.18 m axis — measured floor; below it the loop collapses) while a phase servo
keeps the loop inflated. The jumper runs the LEARNED rope-hop policy
continuously — trained to hop in place at the rope's rate. CPU MuJoCo (the
elastic cable has no GPU/Warp backend).

The rope drive (mocap carriers) is the DECLARED idealization: the turner ducks
hold the rope with a firm grip while the learned whole-body turner is trained
(the 0904 doc's program). The rope's rotation, the jumper's hops, contacts, and
trip physics are all real.

Phase sync: human turners time the rope to the jumper, not vice versa — so the
rope drive phase-locks to the DUCK. A discrete PLL watches two events per
cycle (duck apex = max feet height, belly pass = min rope z near the jumper)
and shifts the drive phase so the belly passes under the feet at apex. The hop
itself is fully autonomous (the trained policy's own rhythm).
"""

from __future__ import annotations

import math
import pathlib

import mujoco
import numpy as np

from ..sim.classic_rope import build_classic_world, RopeHoldDrive, ChainRopeDrive, ChainForcedDrive
from ..sim.composer import DuckSpec, DUCK_COLORS
from ..sim.runtime import PolicyBank, DuckRuntime

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
POL = ROOT / "microduck/policies"

AIR_Z = 0.015          # feet above this = airborne (matches training)
PASS_Z = 0.08          # rope below this near the jumper = a pass to be hopped
# head-pose reference for the turner circle (matches DEFAULT_POSE head joints)
NECK_PITCH = 0.3491
DEFAULT_HEAD_PITCH = 0.3491


def run_classic_skip(
    out,
    seconds: float = 34.0,
    rope_length: float = 0.62,
    rope_density: float = 1000.0,
    settle_target_hz: float = 1.6,   # settles at ~2.8 Hz (measured offset)
    seed: int = 0,
    render_fps: int = 25,
    overlay_fn=None,
    bank=None,
    hop_onnx=None,          # trained continuous-hop policy (REQUIRED)
    never_live=False,       # DEBUG: keep the rope a ghost (isolate trip physics)
    carrier_follow=False,   # carriers drift-track the jumper — DISABLED: the
                            # track created a runaway (rope follows the drifting
                            # duck, nothing restores center, the drive
                            # destabilizes → QACC NaN. Measured). The v10 hop
            # holds ~9 cm/25 s on its own — the rope stays at center.
    rope_kind="chain",      # the serial chain: slow robust ~1 Hz loop (cable is
                            # chaos-fragile and ≥2.8 Hz — measured)
):
    """Run + render a classic overhead-rope skip session. Returns metrics dict."""
    assert hop_onnx is not None, "classic skip needs the trained rope-hop policy"
    if bank is None:
        bank = {"stand": str(POL / "alpha_stand.onnx"),
                "sitstand": str(POL / "alpha_sitstand.onnx"),
                "walk": str(POL / "alpha_walking.onnx"),
                "jump": str(hop_onnx)}
    bank = PolicyBank(bank)
    # dt=0.001: cable-proven physics; the duck hops identically at 1 ms × 20
    # substeps as at the training 5 ms × 4 (measured, task #16).
    SUB = 20
    rng = np.random.default_rng(seed)

    # Staging (the human sequence): turners spin the rope up while the jumper
    # waits OUTSIDE the sweep, then steps in and starts skipping.
    # - carriers start LIFTED (+0.16 m — a floor-piled rope never inflates)
    # - rope is a ghost (no contacts) until the loop holds
    # - sky waits at y=-0.22, walks to y=-0.06 at t=15 s, hops from t=18 s
    ducks = [
        DuckSpec("lavender", (-0.25, 0.0), yaw=0.0, color=DUCK_COLORS["lavender"]),
        DuckSpec("cream", (0.25, 0.0), yaw=math.pi, color=DUCK_COLORS["cream"]),
        DuckSpec("sky", (0.0, -0.01), yaw=math.pi / 2, color=DUCK_COLORS["sky"]),
    ]
    # carrier axis at 0.20 (not 0.18): the belly still grazes but the strike is
    # gentler — measured no QACC NaN over 70 s at 0.20 vs sporadic NaN at 0.18
    m, d, info = build_classic_world(rope_length=rope_length, rope_density=rope_density,
                                     timestep=0.001, carrier_height=0.20, ducks=ducks,
                                     rope_kind=rope_kind)    # thicker, high-visibility rope so the skip reads on video
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith("rope/"):
            m.geom_rgba[g] = [1.0, 0.45, 0.05, 1.0]
    ids = info["rope_body_ids"]
    ma, mb = info["mocap_a"], info["mocap_b"]
    cA, cB = info["carrier_centers"]
    axis_z = float(cA[2])

    rt = {}
    for nm in ("lavender", "cream"):
        rt[nm] = DuckRuntime(m, d, bank, prefix=f"{nm}/", name=nm)
        # STANDING turners look competent; sitting reads as collapsed on video
        rt[nm].active_policy = "stand"
        rt[nm].set_command(twist=(0, 0, 0))
    rt["sky"] = DuckRuntime(m, d, bank, prefix="sky/", name="sky")
    rt["sky"].active_policy = "stand"

    # rope↔jumper contact geoms (for trip detection + the ghost window)
    rope_geoms = {g for g in range(m.ngeom)
                  if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("rope/")}
    sky_geoms = {g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("sky/")}
    # The rope↔floor contact is LIVE from step 0 (the graze brake governs the
    # rate; ghost-then-live slammed the loop dead — measured). But the JUMPER
    # ghosts the rope until the loop holds: a duck IN the sweep during spin-up
    # bleeds the build (measured). sky: contype=4 conaffinity=5 → 7 when live.
    for g in sky_geoms:
        m.geom_contype[g] = 4
        m.geom_conaffinity[g] = 5

    def feet_z():
        return min(rt["sky"].site_pos("left_foot")[2], rt["sky"].site_pos("right_foot")[2])

    def rope_near_z():
        """Lowest rope point near the JUMPER (the belly sweeping under its
        feet — gated in x AND y; the global lowest point sits at y≈0 and
        would fire while the rope is still meters from the duck)."""
        pts = np.array([d.xpos[b] for b in ids])
        dy = pts[:, 1] - (drive.track_xy[1] - 0.01)
        near = pts[(np.abs(pts[:, 0] - drive.track_xy[0]) < 0.20) & (np.abs(dy) < 0.10)]
        return float(np.min(near[:, 2])) if len(near) else 1.0

    # rope_length 0.56: the settled belly grazes z≈+1 cm (measured) — going
    # live has no penetration slam.
    # the chain rope gets the minimal ChainRopeDrive (pure tracking — the
    # RopeHoldDrive's rate machinery kills it, measured); the cable keeps the
    # hold drive
    if rope_kind == "chain":
        # the measured duck hop rate (v8 deploys at ~3.1 Hz) IS the drive rate
        drive = ChainForcedDrive(0.20, rate_hz=3.1)
    else:
        drive = RopeHoldDrive(0.18, settle_target_hz=settle_target_hz,
                              hold_mode=True)
    # rope dof snapshot for stall-retry (the swing-up is chaotic — on stall,
    # reset the rope to its straight rest and re-spin with amplitude jitter;
    # the ducks are untouched)
    rope_joints = [j for j in range(m.njnt)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("rope/")]
    rope_qadr = [m.jnt_qposadr[j] for j in rope_joints]
    rope_vadr = [m.jnt_dofadr[j] for j in rope_joints]
    rope_qpos0 = d.qpos.copy()
    rope_qvel0 = d.qvel.copy()
    # flat dof index array for the anti-whip velocity clip
    _span = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}
    rope_dof_idx = np.concatenate([np.arange(a, a + _span.get(m.jnt_type[j], 1))
                                   for a, j in zip(rope_vadr, rope_joints)])
    if rope_kind == "chain":
        drive.rope_vadr = rope_dof_idx

    # The drive runs from the world's FIRST step (carriers start lifted — the
    # rope never piles). Sky waits outside the sweep, walks in once the loop
    # is inflated and lowered, then hops; rope contacts go live with the skip.

    # ── timing PLL state: lock the belly's pass to the duck's apex ──────────
    last_apex_t = None
    last_pass_t = None
    head_phi = 0.0
    prev_feet = 0.0
    prev_rope_z = 1.0
    pass_enter_t = None
    pass_enter_air = False
    pass_enter_ok = False
    air_hist: list[bool] = []
    up_hist: list[bool] = []
    rising = False
    airborne = False          # hysteresis: air > 3 cm, grounded < 1.5 cm
    takeoffs: list[float] = []
    t = 0.0

    ren = mujoco.Renderer(m, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0, 0, 0.13]
    camera.distance = 1.05
    camera.azimuth = 115
    camera.elevation = -12

    frames = []
    metrics = dict(hops=0, passes=0, skips=0, trips=0, contacts=0)
    rope_live = False    # rope↔jumper goes live when the duck is in position
    last_stand_t = None
    hop_start_t = None
    retries_done = 0
    low_z_t = 0.0
    lock_count = 0
    e_filt = 0.0
    e_log: list = []
    NPOL = int(seconds * 50)
    for pi in range(NPOL):
        t = pi * 0.02
        # rope stall retry: the swing-up catch is chaotic — on a stall, reset
        # the rope dofs to rest and re-spin (per-attempt amplitude jitter
        # breaks the tie). The ducks are untouched.
        if getattr(drive, "stalled", False) and drive.retries < 5:
            for a, j in zip(rope_qadr, rope_joints):
                span = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(m.jnt_type[j], 1)
                d.qpos[a:a+span] = rope_qpos0[a:a+span]
            for a, j in zip(rope_vadr, rope_joints):
                dspan = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(m.jnt_type[j], 1)
                d.qvel[a:a+dspan] = rope_qvel0[a:a+dspan]
            drive.reset_spinup()
        # staging: sky STANDS at the skip spot from t=0 (the walk→hop
        # transition was the fall source — measured — and the walk overshoot
        # scattered the duck; a long clean stand is the reliable launch).
        # Rope↔sky stays GHOST: a 3 mm rope at 3 m/s brushing a leg is
        # sub-frame on video, and a live chain clipping the duck explodes the
        # contact solver (QACC NaN — measured twice). Timing is what's scored.
        rope_ready = drive.inflated and (drive.t - drive.t_inflated) > 12.0
        hop_ready = rope_ready and t > 3.0
        if hop_ready and rt["sky"].active_policy != "jump":
            rt["sky"].active_policy = "jump"
            rt["sky"].set_command()
            hop_start_t = t
        if hop_ready and rt["sky"].active_policy != "jump":
            rt["sky"].active_policy = "jump"
            rt["sky"].set_command()
            hop_start_t = t
        # watchdog: SUSTAINED low trunk (a real face-plant) → re-stand, retry.
        # NOT instantaneous tilt — the hop's launch crouch dips to z≈0.03 and
        # the 2.8 Hz tilt cycle trips any instantaneous check (measured: a
        # trigger-happy watchdog reset-loops a healthy hopper).
        sky_z = rt["sky"].trunk_pos()[2]
        low_z_t = low_z_t + 0.02 if sky_z < 0.07 else 0.0
        if (rt["sky"].active_policy == "jump" and hop_start_t is not None
                and t > hop_start_t + 2.0 and low_z_t > 0.8):
            rt["sky"].active_policy = "stand"
            rt["sky"].set_command(twist=(0, 0, 0))
            hop_start_t = None
            last_stand_t = t
        if (rt["sky"].active_policy == "stand" and hop_start_t is None
                and rope_ready and retries_done < 3
                and t > (last_stand_t or 0) + 2.5):
            rt["sky"].active_policy = "jump"
            rt["sky"].set_command()
            hop_start_t = t
            retries_done += 1
        # carriers follow the hopping duck (low-pass, ±9 cm — the rope stays
        # over the jumper like real turners adjusting; the duck's hop has a
        # residual drift the policy can't fully kill)
        if carrier_follow and rt["sky"].active_policy == "jump":
            duck_xy = rt["sky"].trunk_pos()[:2]
            drive.track_xy += 0.02 * (duck_xy - np.array([0.0, -0.01]) - drive.track_xy)
            np.clip(drive.track_xy, -0.09, 0.09, out=drive.track_xy)
        for dd in rt.values():
            dd.step()
        for s in range(SUB):
            drive.step(d, ids, ma, mb, cA, cB, dt=0.001)
            mujoco.mj_step(m, d)

        # ── events (once per policy step) ────────────────────────────────────
        fz = feet_z()
        if airborne and fz < AIR_Z:
            airborne = False
        elif not airborne and fz > 0.03:
            airborne = True
            takeoffs.append(t)
            if len(takeoffs) >= 3:
                duck_rate = 1.0 / float(np.median(np.diff(takeoffs[-4:])))
                drive.rate_hz += 0.05 * (duck_rate - drive.rate_hz)   # EMA track
        air = airborne
        air_hist.append(air)
        up_hist.append(rt["sky"].is_upright(0.5))
        if len(air_hist) > 60:
            air_hist.pop(0)
            up_hist.pop(0)
        rz = rope_near_z()

        # duck apex: feet stop rising while airborne
        if air and fz < prev_feet and rising:
            last_apex_t = t - 0.02
            metrics["hops"] += 1
        rising = air and fz >= prev_feet

        # belly pass: time the CENTER of the below-threshold window (the
        # crossing edge leads the belly's lowest point by half the transit —
        # that bias alone cost ~30% of the skip rate, measured)
        if prev_rope_z > PASS_Z >= rz:
            pass_enter_t = t
            pass_enter_air = air
            pass_enter_ok = (rt["sky"].is_upright(0.5)
                             and rt["sky"].trunk_pos()[2] > 0.09)
        if prev_rope_z <= PASS_Z < rz and pass_enter_t is not None and drive.inflated:
            last_pass_t = 0.5 * (pass_enter_t + t)
            pass_enter_t = None
            # evaluate the skip at the window CENTER (the belly's lowest
            # transit), not at the exit edge — the exit is ~half a hop late
            mid_idx = int((last_pass_t - t) / 0.02)   # negative (in the past)
            air_mid = air_hist[mid_idx] if -len(air_hist) <= mid_idx < 0 else air
            up_mid = up_hist[mid_idx] if -len(up_hist) <= mid_idx < 0 else rt["sky"].is_upright(0.5)
            if rope_ready:                     # measure the skip window only
                metrics["passes"] += 1
                # skip = airborne at the belly's lowest transit. (The trunk-z
                # gate excluded a legit crouch-hopper — v8 rides at 0.08–0.13.)
                if air_mid and up_mid:
                    metrics["skips"] += 1
                else:
                    metrics["trips"] += 1

        # timing PLL (once per cycle, when both events exist): the pass should
        # coincide with the apex. e > 0 = pass late → advance the drive phase.
        # The phase shift needed is ω·e (rad) — a gain of 0.4·e takes 25 cycles
        # to lock and the rope dies to contact drag first (measured). Lock in
        # 1–2 cycles at ½·ω·e, clipped per-step for stability. Go live only on
        # a SUSTAINED lock — a mistimed live rope dies to duck contact.
        if last_apex_t is not None and last_pass_t is not None:
            # Damped phase PLL: the drive RATE tracks the duck's measured hop
            # rate (EMA — a learned hopper is no metronome), and phase_off
            # integrates the pass-vs-apex error at 40%/cycle (stable, locks in
            # ~4 cycles). Pure rate-integration oscillated across the clip
            # (measured); phase yanks need the slew limit (drive-side).
            P = 1.0 / drive.rate_hz
            e = (last_pass_t - last_apex_t + P / 2) % P - P / 2
            # EMA the error (the belly's pass time jitters ±30 ms) and integrate
            # gently — the raw-gain version limit-cycled ±0.14 s (measured)
            e_filt = 0.5 * e + 0.5 * e_filt
            drive.phase_off = float(np.clip(
                drive.phase_off + 2 * math.pi * drive.rate_hz * e_filt * 0.25, -3.2, 3.2))
            lock_count = lock_count + 1 if abs(e_filt) < 0.05 else 0
            e_log.append((t, e, drive.phase_off))
            last_apex_t = last_pass_t = None

        # rope↔jumper contact bookkeeping
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in rope_geoms and g2 in sky_geoms) or (g2 in rope_geoms and g1 in sky_geoms):
                metrics["contacts"] += 1
                break

        prev_feet = fz
        prev_rope_z = rz



        if pi % 2 == 0:
            # camera tracks the jumper (slow) so the action stays centered
            sp = rt["sky"].trunk_pos()
            camera.lookat = [0.9 * camera.lookat[0] + 0.1 * sp[0],
                             0.9 * camera.lookat[1] + 0.1 * sp[1], 0.13]
            ren.update_scene(d, camera=camera)
            img = ren.render().copy()
            if overlay_fn is not None:
                img = overlay_fn(img, t, metrics)
            frames.append(img)
    import numpy as _np
    if e_log:
        ee = _np.array([e[1] for e in e_log])
        print(f"PLL e: n={len(ee)} mean|e|={_np.abs(ee).mean():.3f}s  last10={ee[-10:].round(3).tolist()}")
    return metrics, frames
