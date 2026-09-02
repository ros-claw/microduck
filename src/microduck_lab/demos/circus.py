"""Microduck Circus — the hero demo.

    You: “你们三个，表演一个跳长绳！”
    Ducks: walk in, turn the rope, jump, trip, practice, evolve, perform.

Scenes are rendered to separate mp4 segments (so each can be re-shot), then
concatenated. Every second of physics you see is the real MuJoCo simulation;
the only declared aid is the coach rope wind-up (see rope_coach.py docstring).
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess

import mujoco
import numpy as np

from ..sim.composer import compose_world, DuckSpec, RopeSpec, DUCK_COLORS
from ..sim.runtime import PolicyBank, DuckRuntime, SUBSTEPS
from ..sim.renderer import CinematicRenderer
from ..skills.navigate import WalkTo, TurnTo
from ..skills.skip import RopeSkipSession, SkipConfig, DT
from ..video.overlay import overlay, title_card, grid_montage
from ..video.writer import VideoWriter

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"
POLICIES = {
    "stand": str(POL / "alpha_stand.onnx"),
    "walk": str(POL / "alpha_walking.onnx"),
    "sitstand": str(POL / "alpha_sitstand.onnx"),
    "roulade": str(POL / "roulade.onnx"),
}

FPS = 25
W, H = 960, 540


def _write_static(writer: VideoWriter, frame: np.ndarray, seconds: float):
    writer.write_n(frame, int(seconds * FPS))


# ------------------------------------------------------------------ scene 2
def scene_intro(out: pathlib.Path):
    """Three ducks walk to their marks, rope in mouths; user sets the goal."""
    ducks = [
        DuckSpec("lavender", (-0.40, 0.16), yaw=0.9, color=DUCK_COLORS["lavender"]),
        DuckSpec("cream", (0.40, 0.16), yaw=math.pi - 0.9, color=DUCK_COLORS["cream"]),
        DuckSpec("sky", (0.0, -0.55), yaw=math.pi / 2, color=DUCK_COLORS["sky"]),
    ]
    w = compose_world(ROBOT, ducks,
                      rope=RopeSpec("lavender", "cream", length=0.55, count=30, density=50),
                      playground=True, rope_height=0.19)
    bank = PolicyBank(POLICIES)
    rt = {d.name: DuckRuntime(w.model, w.data, bank, prefix=f"{d.name}/", name=d.name)
          for d in ducks}
    ren = CinematicRenderer(w.model, W, H)
    ren.set_cam("wide")
    wr = VideoWriter(str(out), FPS)

    nav = {"lavender": WalkTo(rt["lavender"]), "cream": WalkTo(rt["cream"]),
           "sky": WalkTo(rt["sky"])}
    nav["lavender"].go_to(-0.27, 0.0)
    nav["cream"].go_to(0.27, 0.0)
    nav["sky"].go_to(0.0, 0.03)

    steps = int(14 * 50)
    frame_i = 0
    chat_at = int(3.0 * FPS)      # user message appears
    ans_at = None
    for i in range(steps):
        for n in nav.values():
            n.update()
        for d in rt.values():
            d.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
        if i % 2 == 0:  # 25 fps
            img = ren.render(w.data)
            chat = "你们三个，给我表演跳长绳！" if frame_i > chat_at else None
            ans = None
            if all(n.done for n in nav.values()) and ans_at is None:
                ans_at = frame_i + int(0.6 * FPS)
            if ans_at and frame_i >= ans_at:
                ans = "技能检查：walk ✓  sit ✓  spin ✓  rope_skip — 需要练习"
            img = overlay(img, status="ACTING", title="Microduck Circus",
                          chat=chat, chat_answer=ans)
            wr.write(img)
            frame_i += 1
        if all(n.done for n in nav.values()) and ans_at and frame_i > ans_at + int(1.6 * FPS):
            break
    wr.close()
    ren.close()
    return out


# ------------------------------------------------------------- scene helper
def _run_skip_scene(out: pathlib.Path, cfg: SkipConfig, *, seconds: float,
                    status: str, title: str, subtitle: str | None = None,
                    slowmo_at: float | None = None, live_metrics=True):
    """Run one skipping episode with rendering. Returns (metrics, consec_best)."""
    s = RopeSkipSession(str(ROBOT), POLICIES, cfg, playground=True)
    s.settle()
    s.start_rope()
    ren = CinematicRenderer(s.world.model, W, H)
    ren.set_cam("side")
    wr = VideoWriter(str(out), FPS)
    consec = 0
    for i in range(int(seconds * 50)):
        s.step()
        if i % 2 == 0:
            img = ren.render(s.world.data)
            m = s.metrics
            met = [f"rope passes: {m.crossings}",
                   f"skips: {m.successful_skips}",
                   f"trips: {m.trips}",
                   f"best streak: {m.consecutive_best}"] if live_metrics else None
            img = overlay(img, status=status, title=title, subtitle=subtitle,
                          metrics=met)
            wr.write(img)
        consec = max(consec, s.metrics.consecutive_best)
    wr.close()
    ren.close()
    return s.metrics, consec


def scene_first_attempt(out: pathlib.Path):
    """The honest first attempt with untrained timing — trips included."""
    cfg = SkipConfig(trigger_lead_s=0.46, seed=42)   # untrained guess: jumps way too late
    m, _ = _run_skip_scene(out, cfg, seconds=16.0, status="ACTING",
                           title="First attempt — untrained timing",
                           subtitle="第一次尝试（没练过）")
    return m


def scene_practice(out: pathlib.Path, variants: list[tuple[str, SkipConfig]]):
    """Practice montage: N episodes in a grid, real metrics per tile."""
    tiles = []
    labels = []
    for tag, cfg in variants:
        s = RopeSkipSession(str(ROBOT), POLICIES, cfg, playground=False)
        s.settle()
        s.start_rope()
        ren = CinematicRenderer(s.world.model, 320, 180)
        ren.set_cam("side")
        frames = []
        for i in range(int(12 * 50)):
            s.step()
            if i > 5 * 50 and i % 8 == 0:      # capture the action window
                frames.append(ren.render(s.world.data))
        ren.close()
        # keep a middle chunk of ~2.5s at 12.5fps → 31 frames
        tiles.append(frames[:31])
        m = s.metrics
        labels.append(f"{tag}: {m.successful_skips}/{m.crossings} skips")
    # write montage video: each frame = grid of tiles
    wr = VideoWriter(str(out), FPS)
    T = max(len(t) for t in tiles)
    from ..video.overlay import _draw_text
    from PIL import Image
    for f in range(T):
        grid = grid_montage([t[min(f, len(t) - 1)] for t in tiles], cols=4)
        im = Image.fromarray(grid).resize((W, H))
        arr = np.asarray(im).copy()
        arr = overlay(arr, status="PRACTICING", title="Practice — 8 parallel episodes",
                      subtitle="并行练习中（真实物理仿真）")
        wr.write(arr)
    wr.close()
    return labels


def scene_evolution(out: pathlib.Path, practice_log: pathlib.Path,
                    champion: dict):
    """The Darwin report card, from the real practice log."""
    wr = VideoWriter(str(out), FPS)
    base = title_card("DARWIN EVALUATION", "", (W, H))
    rows = []
    if practice_log.exists():
        from ..learning.practice import PracticeLog
        recs = PracticeLog(practice_log).read_all()
        # summarize per skill_version
        by = {}
        for r in recs:
            by.setdefault(r["skill_version"], []).append(r)
        for v, rs in by.items():
            rate = np.mean([r["metrics"]["success"] / max(1, r["metrics"]["crossings"])
                            for r in rs])
            rows.append((v, rate))
    champ_txt = (f"CHAMPION v{champion['champion']}  holdout "
                 f"{champion['holdout_success']:.0%}  best streak {champion['consecutive_best']}")
    for f in range(int(4.5 * FPS)):
        img = base.copy()
        lines = [f"{v:>10s}   {'█' * int(rate * 30):30s} {rate:.0%}"
                 for v, rate in rows[-8:]]
        img = overlay(img, status="EVOLVING", metrics=["evolution of rope_skip:", *lines],
                      subtitle=None)
        wr.write(img)
    # champion card
    card = title_card("CHAMPION PROMOTED", champ_txt, (W, H), color=(30, 26, 10))
    _write_static(wr, card, 1.8)
    wr.close()


def scene_performance(out: pathlib.Path, champion_cfg: SkipConfig):
    """The performance take: champion config, camera work, slow-mo on the best jump."""
    return _run_skip_scene(out, champion_cfg, seconds=16.0, status="CHAMPION",
                           title="Performance — champion skill",
                           subtitle="正式表演")


def scene_celebration(out: pathlib.Path):
    """Rope drop, everyone stands up, spin, roulade finale."""
    ducks = [
        DuckSpec("lavender", (-0.30, 0.10), yaw=math.pi, color=DUCK_COLORS["lavender"]),
        DuckSpec("cream", (0.30, 0.10), yaw=0.0, color=DUCK_COLORS["cream"]),
        DuckSpec("sky", (0.0, -0.12), yaw=-math.pi / 2, color=DUCK_COLORS["sky"]),
    ]
    w = compose_world(ROBOT, ducks, playground=True)
    bank = PolicyBank(POLICIES)
    rt = {d.name: DuckRuntime(w.model, w.data, bank, prefix=f"{d.name}/", name=d.name)
          for d in ducks}
    ren = CinematicRenderer(w.model, W, H)
    ren.set_cam("hero34")
    wr = VideoWriter(str(out), FPS)

    # settle → synchronized spin → turn to face outward → staggered roulades
    outward = {"lavender": math.pi, "cream": 0.0, "sky": -math.pi / 2}
    turns = {n: TurnTo(d) for n, d in rt.items()}
    spin_at = int(2.0 * 50)
    turn_at = int(4.8 * 50)
    roulade_at = {"lavender": int(6.4 * 50), "sky": int(7.1 * 50), "cream": int(7.8 * 50)}
    roulade_dur = int(2.0 * 50)
    turned = False
    for i in range(int(11.5 * 50)):
        if i == turn_at:
            for n, t in turns.items():
                t.turn_to(outward[n])
        for name, d in rt.items():
            if i < spin_at:
                pass
            elif i < turn_at:
                d.active_policy = "walk"
                d.set_command(twist=(0, 0, 1.1))
            elif i < roulade_at[name]:
                turns[name].update()           # face outward first
            elif i < roulade_at[name] + roulade_dur:
                d.active_policy = "roulade"
                d.set_command(twist=(0, 0, 0), head=(0, 0, 0, 0), body=(0, 0, 0, 0, 0, 0))
            else:
                d.active_policy = "stand"
                d.set_command(twist=(0, 0, 0))
            d.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
        if i % 2 == 0:
            img = ren.render(w.data)
            img = overlay(img, status="CHAMPION", title="Celebration", subtitle="谢幕！")
            wr.write(img)
    wr.close()
    ren.close()


def assemble(segments: list[pathlib.Path], out: pathlib.Path, fps=FPS):
    """Concat mp4 segments with the imageio-ffmpeg binary."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True, capture_output=True)
    lst.unlink()
