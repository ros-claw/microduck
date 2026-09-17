"""Honest classic rope-skip: the rope's ends ride the turner ducks' handle tips
and the rope is driven by the turners' BODIES (the learned turner policy swings
the mouth-held handle in the drive circle while the legs balance) — no mocap
carriers, no kinematic idealization. The jumper runs the trained rope-hop
policy. A timing PLL nudges the shared drive clock's phase so the belly passes
under the jumper's feet at apex.

This is the CR-05 answer: rope connected to the ducks' beaks, driven by the
ducks' whole-body motion, measured overhead loop (3.1 Hz is the physical rate
at this scale — the loop needs the speed to stand over the head).
"""

from __future__ import annotations

import math
import pathlib

import mujoco
import numpy as np

from ..sim.classic_rope import build_classic_world
from ..sim.runtime import PolicyBank, DuckRuntime, TurnerDuckRuntime, RopeTurnerDuckRuntime

from ..paths import ASSET_ROOT as ROOT
POL = ROOT / "microduck/policies"

AIR_Z = 0.015
PASS_Z = 0.08


def run_honest_classic_skip(
    out,
    seconds: float = 50.0,
    rope_length: float = 0.50,
    rope_density: float = 500.0,
    turn_hz: float = 3.1,
    render_fps: int = 25,
    overlay_fn=None,
    turner_onnx=None,
    hop_onnx=None,
    seed: int = 0,
    render: bool = True,
    rope_contacts: str | None = None,
    physics_observer=None,
    model_setup=None,
    legacy_rope_offset: bool = True,
    connect_timeconst: float = 0.04,
    rope_floor_timeconst: float | None = None,
    physics_dt: float = .001,
    hop_start_delay: float = 8.,
    rope_joint_type: str = "hinge",
    rope_radius: float = .003,
    rope_pass_time_fn=None,
    jumper_y: float = -.10,
    rope_phase_bias: float = 0.,
    settle_seconds: float = 1.,
    rope_initial_phase: float = 0.,
    rope_velocity_limit: float = 40.,
    fixed_turn_rate: bool = False,
    hop_feedback: bool = True,
    max_turn_hz: float | None = None,
    asset_colors: bool = False,
    presentation: str = "lab",
    ducks=None,
    policy_observer=None,
    spawn_jitter: float = .02,
):
    """Return (legacy timing counters, frames), NOT physical skip success.

    ``render=False`` runs without a GL context. ``rope_contacts=None`` keeps
    the historical contactless dynamics; full enables floor + jumper contacts
    from initialization. ``physics_observer`` receives every physics step,
    including the one-second warmup; ``model_setup`` runs before any stepping.
    """
    assert turner_onnx and hop_onnx, "needs the turner + hop policies"
    if not 0 <= hop_start_delay <= 8:
        raise ValueError("hop_start_delay must be between 0 and the legacy 8 second gate")
    if settle_seconds < 0:
        raise ValueError("settle_seconds must be nonnegative")
    if rope_velocity_limit < 0:
        raise ValueError("rope_velocity_limit must be nonnegative; zero disables clipping")
    if max_turn_hz is not None:
        if max_turn_hz <= 0:
            raise ValueError('max_turn_hz must be positive')
        turn_hz = min(turn_hz, max_turn_hz)
    bank = PolicyBank({
        "stand": str(POL / "alpha_stand.onnx"),
        "walk": str(POL / "alpha_walking.onnx"),
        "turn": str(turner_onnx),
        "jump": str(hop_onnx),
    })
    if physics_dt <= 0 or not np.isclose(.02/physics_dt,round(.02/physics_dt)):
        raise ValueError("physics_dt must divide the 20 ms policy period")
    SUB = round(.02/physics_dt)

    m, d, info = build_classic_world(
        ducks=ducks,
        rope_length=rope_length, rope_density=rope_density,
        timestep=physics_dt, carrier_height=0.20, rope_kind="triple",
        connect_to="handles", turner_sep=0.448,
        rope_contacts=rope_contacts,
        legacy_rope_offset=legacy_rope_offset,
        connect_timeconst=connect_timeconst,
        rope_floor_timeconst=rope_floor_timeconst,
        rope_joint_type=rope_joint_type,
        rope_radius=rope_radius,
        jumper_y=jumper_y,
        rope_initial_phase=rope_initial_phase,
        asset_colors=asset_colors,
        presentation=presentation,
    )
    if model_setup is not None:
        model_setup(m, d, info)
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith("rope/"):
            m.geom_rgba[g] = [1.0, 0.45, 0.05, 1.0]
    ids = info["rope_body_ids"]

    rt = {}
    for nm in ("lavender", "cream"):
        rt[nm] = RopeTurnerDuckRuntime(m, d, bank, prefix=f"{nm}/", name=nm,
                                       turn_frequency=turn_hz,
                                       rope_body_ids=info["rope_body_ids"],
                                       axis_y=0.0, axis_z=0.20)
        # the turner policy runs from t=0 (it stands AND turns; the 69D obs
        # can't feed the 61D stand policy, so no stand-first settle)
        rt[nm].active_policy = "turn"
        rt[nm].set_command(twist=(0, 0, 0))
    rt["sky"] = DuckRuntime(m, d, bank, prefix="sky/", name="sky")
    rt["sky"].active_policy = "stand"
    rt["sky"].set_command(twist=(0, 0, 0))
    # Feedback from the material middle of the real flexible rope. No clock
    # drives the jumper's height command. The training apparatus uses the same
    # lower-crossing phase convention, but acceptance remains this full scene.
    middle_body = ids[len(ids)//2]
    handle_sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'{nm}/handle_tip')
                    for nm in ('lavender', 'cream')]
    if min(handle_sites) < 0:
        raise ValueError('Missing rope handle sites')
    def measured_rope_phase(name='sky'):
        if not hop_feedback:
            return float(np.pi)  # explicit ablation: zero height delta, same physics
        belly = d.xpos[middle_body]
        foot_y = .5*(rt[name].site_pos('left_foot')[1]+rt[name].site_pos('right_foot')[1])
        axis_z = float(np.mean(d.site_xpos[handle_sites, 2]))
        return float(np.arctan2(belly[1]-foot_y, axis_z-belly[2]))
    rt['sky'].hop_phase_source = measured_rope_phase
    for duck in (ducks or [])[3:]:
        rt[duck.name] = DuckRuntime(m, d, bank, prefix=duck.name+'/', name=duck.name)
        rt[duck.name].active_policy = 'stand'
        rt[duck.name].set_command()
        rt[duck.name].hop_phase_source = lambda name=duck.name: measured_rope_phase(name)

    # Seeded spawn jitter exposes the chain's startup sensitivity. Evaluation
    # reports unsuccessful seeds as well as the seed shown in the video.
    if not 0 <= spawn_jitter <= .1:
        raise ValueError('spawn_jitter must be in [0, .1] radians')
    rng = np.random.default_rng(seed)
    # Spawn the ducks at the DEFAULT_POSE (home crouch), NOT the XML's
    # straight-leg qpos0: the policies' joint_pos_rel obs is relative to home,
    # so a straight-leg spawn reads as a ±0.45 rad offset — inside the stand
    # policy's robustness but NOT the specialist turner's (measured: the turner
    # policy fires violent actions on the offset obs and throws the duck).
    for dd in rt.values():
        d.qpos[dd.joint_qpos_idx] = dd.default_pose + rng.uniform(-spawn_jitter, spawn_jitter, 14)
        d.qvel[dd.joint_qvel_idx] = 0.0
    mujoco.mj_forward(m, d)

    # anti-whip: clip the rope's joint velocities in the physics loop (the
    # 3.1 Hz chain occasionally diverges via a joint-velocity explosion —
    # measured; the loop never reaches ±40 rad/s, the whip does)
    _span = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}
    rope_vadr = np.concatenate([
        np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + _span.get(m.jnt_type[j], 1))
        for j in range(m.njnt)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("rope/")])

    # Legacy counters below score TIMING, even when contacts are enabled.
    # They do not prove passage below the feet or a clean landing.

    def feet_z():
        return min(rt["sky"].site_pos("left_foot")[2], rt["sky"].site_pos("right_foot")[2])

    def rope_near_z():
        pts = np.array([d.xpos[b] for b in ids])
        near = pts[np.abs(pts[:, 0]) < 0.15]
        return float(np.min(near[:, 2])) if len(near) else 1.0

    def rope_top_z():
        pts = np.array([d.xpos[b] for b in ids])
        near = pts[np.abs(pts[:, 0]) < 0.15]
        return float(np.max(near[:, 2])) if len(near) else 0.0

    ren = mujoco.Renderer(m, height=540, width=960) if render else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Include the whole overhead loop; the old low, tight framing clipped the
    # rope at the top and visually made its two ends look disconnected.
    camera.lookat = [0, 0, 0.25]
    camera.distance = 1.15
    camera.azimuth = 115
    camera.elevation = -12
    if ducks is not None and len(ducks) > 3:
        camera.lookat = [0, -.08, .25]
        camera.distance = 1.6
        camera.azimuth = -70
        camera.elevation = -25

    # settle: 1 s to absorb the connect's initial transient, then the turners
    # circle from t=0 — in-phase 3.1 Hz circles spin the rope up with NO seed
    # (measured: two-pin probe, r=0.05-0.06, locks at the drive rate)
    for _ in range(round(settle_seconds * 50)):
        if policy_observer is not None:
            policy_observer(m, d, info, rt)
        for dd in rt.values():
            dd.step()
        for _ in range(SUB):
            if rope_velocity_limit:
                d.qvel[rope_vadr] = np.clip(d.qvel[rope_vadr], -rope_velocity_limit, rope_velocity_limit)
            mujoco.mj_step(m, d)
            if physics_observer is not None:
                physics_observer(m, d, info, False)


    frames = []
    metrics = dict(passes=0, skips=0, trips=0)
    last_apex_t = None
    last_feedback_pass = None
    prev_rope_z = 1.0
    pass_enter_t = None
    airborne = False
    rising = False
    prev_feet = 0.0
    air_hist: list[bool] = []
    up_hist: list[bool] = []
    hopping = False
    t = 0.0
    takeoffs: list[float] = []
    duck_rate = turn_hz
    e_filt = 0.0
    # rope rate tracker (belly wraps) for the hop gate
    _prev_th = None
    _wraps = 0
    _cont = 0.0
    rope_rate = 0.0
    rope_fast_t = 0.0
    low_z_t = 0.0
    hop_start_t = None
    hop_retries = 0
    last_stand_t = 0.0
    NPOL = int(seconds * 50)
    for pi in range(NPOL):
        t = pi * 0.02
        if hop_start_delay < 8. and not hopping and hop_start_t is None and hop_retries == 0 and t >= hop_start_delay:
            rt["sky"].active_policy = "jump"
            rt["sky"].set_command()
            hopping = True
            hop_start_t = t
        # rope rotation rate (belly wraps about the drive axis)
        _bp = np.array([d.xpos[b] for b in ids])
        _belly = _bp[np.argmin(_bp[:, 2])]
        _th = math.atan2(_belly[1], -(_belly[2] - 0.20))
        if _prev_th is not None:
            _dth = _th - _prev_th
            if _dth > math.pi: _wraps -= 1
            elif _dth < -math.pi: _wraps += 1
            _new_cont = _th + _wraps * 2 * math.pi
            rope_rate = 0.98 * rope_rate + 0.02 * abs(_new_cont - _cont) / 0.02
            _cont = _new_cont
        _prev_th = _th
        rope_fast_t = rope_fast_t + 0.02 if rope_rate > 15.0 else 0.0
        # the jumper starts hopping once the rope spins fast for a sustained
        # stretch AND sweeps low (the over-conservative late gate measured
        # WORSE — the early window is fine once the rope locks)
        if not hopping and t > 8.0 and rope_fast_t > 1.0 and rope_near_z() < 0.12:
            rt["sky"].active_policy = "jump"
            rt["sky"].set_command()
            hopping = True
            hop_start_t = t
        # watchdog: SUSTAINED low trunk (a real face-plant) → re-stand and
        # retry the hop (the hop's launch crouch dips to z≈0.03 — only a
        # sustained low z is a real fall, so gate on duration)
        sky_z = rt["sky"].trunk_pos()[2]
        low_z_t = low_z_t + 0.02 if sky_z < 0.07 else 0.0
        if (hopping and hop_start_t is not None and t > hop_start_t + 2.0
                and low_z_t > 0.8 and hop_retries < 4):
            rt["sky"].active_policy = "stand"
            rt["sky"].set_command(twist=(0, 0, 0))
            hopping = False
            hop_start_t = None
            last_stand_t = t
            hop_retries += 1
        if (not hopping and hop_start_t is None and t > 8.0
                and t > last_stand_t + 2.5 and rope_near_z() < 0.15):
            rt["sky"].active_policy = "jump"
            rt["sky"].set_command()
            hopping = True
            hop_start_t = t
        if policy_observer is not None:
            policy_observer(m, d, info, rt)
        for dd in rt.values():
            dd.step()
        for _ in range(SUB):
            if rope_velocity_limit:
                d.qvel[rope_vadr] = np.clip(d.qvel[rope_vadr], -rope_velocity_limit, rope_velocity_limit)
            mujoco.mj_step(m, d)
            if physics_observer is not None:
                physics_observer(m, d, info, hopping)

        fz = feet_z()
        if airborne and fz < 0.01:
            airborne = False
        elif not airborne and fz > 0.02:
            airborne = True
            takeoffs.append(t)
            if len(takeoffs) >= 3 and not fixed_turn_rate:
                duck_rate = 1.0 / float(np.median(np.diff(takeoffs[-4:])))
                # the drive RATE tracks the hopper's measured rate (a learned
                # hopper is no metronome — measured in the carrier version)
                for nm in ("lavender", "cream"):
                    rt[nm].turn_frequency += 0.05 * (duck_rate - rt[nm].turn_frequency)
                    if max_turn_hz is not None:
                        rt[nm].turn_frequency = min(rt[nm].turn_frequency, max_turn_hz)
        air = airborne
        air_hist.append(air)
        up_hist.append(rt["sky"].is_upright(0.5))
        if len(air_hist) > 60:
            air_hist.pop(0); up_hist.pop(0)
        rz = rope_near_z()

        if air and fz < prev_feet and rising:
            last_apex_t = t - 0.02
        rising = air and fz >= prev_feet

        feedback_pass = None
        if prev_rope_z > PASS_Z >= rz:
            pass_enter_t = t
        if prev_rope_z <= PASS_Z < rz and pass_enter_t is not None:
            last_pass_t = 0.5 * (pass_enter_t + t)
            if rope_pass_time_fn is None:
                feedback_pass = last_pass_t
            pass_enter_t = None
            mid_idx = int((last_pass_t - t) / 0.02)
            air_mid = air_hist[mid_idx] if -len(air_hist) <= mid_idx < 0 else air
            up_mid = up_hist[mid_idx] if -len(up_hist) <= mid_idx < 0 else True
            if hopping:
                metrics["passes"] += 1
                if air_mid and up_mid:
                    metrics["skips"] += 1
                else:
                    metrics["trips"] += 1
        if rope_pass_time_fn is not None:
            observed_pass = rope_pass_time_fn()
            if observed_pass is not None and observed_pass != last_feedback_pass:
                feedback_pass = observed_pass
                last_feedback_pass = observed_pass
        if feedback_pass is not None and not fixed_turn_rate:
            # timing PLL (classic structure): EMA the pass-vs-apex error and
            # integrate the shared drive clock's phase offset at 25%/cycle —
            # the raw-gain version limit-cycled (measured in the carrier demo)
            if last_apex_t is not None:
                P = 1.0 / rt["lavender"].turn_frequency
                e = (feedback_pass - last_apex_t - rope_phase_bias + P / 2) % P - P / 2
                e_filt = 0.5 * e + 0.5 * e_filt
                for nm in ("lavender", "cream"):
                    rt[nm].phase_offset_s += e_filt * 0.25
                    rt[nm].phase_offset_s = float(np.clip(rt[nm].phase_offset_s, -0.16, 0.16))
                last_apex_t = None
        prev_feet = fz
        prev_rope_z = rz

        if render and pi % max(1, round(50 / render_fps)) == 0:
            sp = rt["sky"].trunk_pos()
            camera.lookat = [0.9 * camera.lookat[0] + 0.1 * sp[0],
                             0.9 * camera.lookat[1] + 0.1 * sp[1], 0.13]
            ren.update_scene(d, camera=camera)
            img = ren.render().copy()
            if overlay_fn is not None:
                img = overlay_fn(img, t, metrics)
            frames.append(img)
    if ren is not None:
        ren.close()
    return metrics, frames
