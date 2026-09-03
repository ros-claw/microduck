"""The "Teach a Duck to Jump" scene — the ROSClaw evolution payoff.

Left: the procedural tuck-hop (the best hand-tuned maneuver, ~2.5 cm).
Right: the trained jump policy (mjlab PPO → ONNX), same physics, same camera.
Real physics both sides; the only difference is the learned skill.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

from ..sim.composer import compose_world, DuckSpec
from ..sim.runtime import PolicyBank, DuckRuntime, SUBSTEPS
from ..sim.renderer import CinematicRenderer
from ..skills.jump import JumpSkill, PolicyJump
from ..video.overlay import overlay
from ..video.writer import VideoWriter

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"
JUMP_ONNX = ROOT / "microduck_rl/output_jump_pd.onnx"


def _run_duck(policy: str, cfg: dict | None = None, jump_onnx=None, seconds=6.0):
    """Run one duck hopping; returns (frames, peaks)."""
    from ..sim.composer import DUCK_COLORS
    w = compose_world(str(ROBOT), [DuckSpec("sky", (0, 0), color=DUCK_COLORS["sky"])],
                      playground=True, grippy_ducks=["sky"] if policy == "policy" else [])
    bank = PolicyBank({"stand": str(POL / "alpha_stand.onnx"),
                       "jump": str(jump_onnx) if jump_onnx else ""})
    d = DuckRuntime(w.model, w.data, bank, prefix="sky/", name="sky")
    for _ in range(2 * 50):
        d.step(); mujoco.mj_step(w.model, w.data, SUBSTEPS)
    if policy == "procedural":
        js = JumpSkill(d)
    else:
        js = PolicyJump(d)
    ren = CinematicRenderer(w.model, 480, 540)
    ren.cam.lookat = [0, 0, 0.15]; ren.cam.distance = 0.55
    ren.cam.azimuth = 90; ren.cam.elevation = -8
    frames = []
    peaks = []
    hops = int(seconds / 1.6)
    for hop in range(hops):
        js.trigger()
        peak = 0.0
        for i in range(int(1.6 * 50)):
            js.update(0.02)
            d.step(); mujoco.mj_step(w.model, w.data, SUBSTEPS)
            l, r = js.feet_z()
            peak = max(peak, min(l, r))
            if i % 2 == 0:
                frames.append(ren.render(w.data))
        peaks.append(peak)
    ren.close()
    return frames, peaks


def scene_jump_evolution(out: pathlib.Path, fps=25):
    """Split-screen: procedural vs trained jump, 3 hops each."""
    print("  [jump scene] procedural (left)...", flush=True)
    left_frames, left_peaks = _run_duck("procedural")
    print(f"    procedural peaks: {[round(p*100,1) for p in left_peaks]} cm", flush=True)
    print("  [jump scene] trained (right)...", flush=True)
    right_frames, right_peaks = _run_duck("policy", jump_onnx=JUMP_ONNX)
    print(f"    trained peaks: {[round(p*100,1) for p in right_peaks]} cm", flush=True)

    wr = VideoWriter(str(out), fps)
    n = min(len(left_frames), len(right_frames))
    for i in range(n):
        combined = np.concatenate([left_frames[i], right_frames[i]], axis=1)
        img = overlay(combined, status="EVOLVING",
                  title="Teach a Duck to Jump — before / after",
                  subtitle="左：手写程序跳  vs  右：RL 训练出的跳跃策略")
        wr.write(img)
    wr.close()
    return left_peaks, right_peaks
