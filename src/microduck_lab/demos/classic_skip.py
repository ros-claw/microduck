"""Classic overhead-rope skip scene — the money shot.

The elastic-cable rope spins up into a real overhead loop (phase-tracking
swing-up, proven by the Oracle Feasibility Lab) and the jumper hops the belly's
bottom pass. CPU MuJoCo (the elastic cable has no GPU/Warp backend).

The rope drive (mocap carriers) is the DECLARED idealization: the turner ducks
hold the rope with a firm grip while the learned whole-body turner is trained
(the 0904 doc's program). The rope's rotation, the jumper's contacts, and the
trip physics are all real.
"""

from __future__ import annotations

import math
import pathlib

import mujoco
import numpy as np

from ..sim.classic_rope import build_classic_world
from ..sim.runtime import PolicyBank, DuckRuntime
from ..skills.jump import JumpSkill

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
POL = ROOT / "microduck/policies"


def run_classic_skip(
    out,
    seconds: float = 26.0,
    rope_length: float = 0.86,
    rope_density: float = 400.0,
    seed: int = 0,
    render_fps: int = 25,
    overlay_fn=None,
    bank=None,
    hop_onnx=None,          # trained hop policy; None = procedural JumpSkill
):
    """Run + render a classic overhead-rope skip session. Returns metrics dict."""
    if bank is None:
        bank = {"stand": str(POL / "alpha_stand.onnx"),
                "sitstand": str(POL / "alpha_sitstand.onnx")}
        if hop_onnx:
            bank["jump"] = str(hop_onnx)
    bank = PolicyBank(bank)
    SUB = 20   # 1 ms physics × 20 = 50 Hz policy
    rng = np.random.default_rng(seed)

    m, d, info = build_classic_world(rope_length=rope_length, rope_density=rope_density)
    # thicker, high-visibility rope so the skip reads on video
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
    # grippy soles for the jumper (the trained hop needs them — physics notes)
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith("sky/") and "foot_collision" in nm:
            m.geom_friction[g, 0] = 2.0
            m.geom_solref[g] = [0.04, 1.0]

    def belly_state():
        pts = np.array([d.xpos[b] for b in ids])
        belly = pts[np.argmin(pts[:, 2])]
        return math.atan2(belly[1], -(belly[2] - axis_z)), pts

    # settle
    for _ in range(int(3 * 50)):
        for dd in rt.values():
            dd.step()
        for _ in range(SUB):
            mujoco.mj_step(m, d)

    if hop_onnx is not None:
        from ..skills.jump import PolicyJump
        jump = PolicyJump(rt["sky"], duration=1.0)
    else:
        jump = JumpSkill(rt["sky"], crouch_depth=0.55, flight_time=0.01, land_time=0.01)
    prev_th, wraps = belly_state()[0], 0
    cont = 0.0
    phys_i = 0
    pass_times = []
    prev_ttg = None
    prev_cont = 0.0

    ren = mujoco.Renderer(m, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0, 0, 0.13]
    camera.distance = 1.05
    camera.azimuth = 115
    camera.elevation = -12

    frames = []
    metrics = dict(hops=0, passes=0, contacts=0, skips=0)
    NPOL = int(seconds * 50)
    for pi in range(NPOL):
        t = pi * 0.02
        for dd in rt.values():
            dd.step()
        jump.update(0.02)
        for s in range(SUB):
            th, pts = belly_state()
            dth = th - prev_th
            if dth > math.pi:
                wraps -= 1
            elif dth < -math.pi:
                wraps += 1
            prev_th = th
            prev_cont = cont
            cont = th + wraps * 2 * math.pi
            r = min(0.055, 0.012 + phys_i * 0.000002) if abs(cont) < 2 * math.pi else 0.055
            phi = cont + math.pi / 2
            yA = r * math.cos(phi)
            zA = axis_z + r * math.sin(phi)
            d.mocap_pos[ma] = cA + np.array([0, yA, zA - axis_z])
            d.mocap_pos[mb] = cB + np.array([0, yA, zA - axis_z])
            mujoco.mj_step(m, d)
            phys_i += 1
        # bottom pass detection (belly wraps through the bottom)
        ca = cont % (2 * math.pi)
        cb = prev_cont % (2 * math.pi)
        if cb > 1.7 * math.pi and ca < 0.3 * math.pi:
            pass_times.append(t)
            metrics["passes"] += 1
        # rhythm-triggered hop
        if len(pass_times) >= 2 and jump.state == "idle" and rt["sky"].is_upright(0.45):
            P = float(np.median(np.diff(pass_times)))
            next_pass = pass_times[-1] + P
            while next_pass - t < 0.84:
                next_pass += P
            ttg = next_pass - t
            if prev_ttg is not None and prev_ttg > 0.86 >= ttg:
                if jump.trigger():
                    metrics["hops"] += 1
            prev_ttg = ttg
        if pi % 2 == 0:
            ren.update_scene(d, camera=camera)
            img = ren.render().copy()
            if overlay_fn is not None:
                img = overlay_fn(img, t, metrics)
            frames.append(img)
    return metrics, frames
