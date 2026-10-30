"""Render actual recorded integration states; camera/UI never enter physics."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import mujoco
from mujoco.rendering.classic.renderer import Renderer
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json

COLORS = {"RED": (245, 91, 100), "BLUE": (81, 171, 255)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def label(draw, xy, text, size=24, color=(244, 247, 252), bold=False, anchor=None):
    draw.text(
        xy,
        text,
        font=ImageFont.truetype(BOLD if bold else FONT, size),
        fill=color,
        anchor=anchor,
        stroke_width=1,
        stroke_fill=(16, 21, 30),
    )


def project(point, scene, m, width, height):
    camera = scene.camera[0]
    delta = np.array(point) - camera.pos
    depth = float(delta @ camera.forward)
    if depth <= 0:
        return None
    scale = height / (2 * math.tan(math.radians(m.vis.global_.fovy) / 2))
    return (
        width / 2 + float(delta @ np.cross(camera.forward, camera.up)) * scale / depth,
        height / 2 - float(delta @ camera.up) * scale / depth,
    )


def render(plan_path, out, width=1920, height=1080):
    plan_path, out = Path(plan_path), Path(out)
    if out.exists() or out.with_suffix(".frames.jsonl").exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "duckfu.film-plan.v1" or plan["fps"] != 50:
        raise ValueError("Unsupported event-driven film plan")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out), fps=50, codec="libx264", quality=8, macro_block_size=1
    )
    cache = {}
    frame_number = 0
    vertical = height > width
    with open(out.with_suffix(".frames.jsonl"), "w") as provenance:
        for shot_index, shot in enumerate(plan["shots"]):
            directory = Path(shot["run"])
            if digest(directory / "manifest.json") != shot["source_manifest_sha256"]:
                raise ValueError("Changed source manifest")
            key = str(directory)
            if key not in cache:
                manifest = json.loads((directory / "manifest.json").read_text())
                for name in ("states.npy", "scene.mjb"):
                    if digest(directory / name) != manifest["hashes"][name]:
                        raise ValueError("Changed capture")
                m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
                d = mujoco.MjData(m)
                states = np.load(directory / "states.npy", mmap_mode="r")
                events = [
                    json.loads(line)
                    for line in (directory / "runtime_events.jsonl")
                    .read_text()
                    .splitlines()
                ]
                cache[key] = (manifest, m, d, states, events, m.geom_group.copy())
            manifest, m, d, states, events, groups = cache[key]
            renderer = Renderer(m, height=height, width=width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            opt = mujoco.MjvOption()
            m.geom_group[:] = groups
            collision = shot["view"] == "collision"
            opt.geomgroup[3] = int(collision)
            if collision:
                opt.geomgroup[2] = 0
                opt.geomgroup[4] = 0
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
                for g in range(m.ngeom):
                    if (
                        m.geom_group[g] == 3
                        and "_combat_shell" not in m.body(int(m.geom_bodyid[g])).name
                    ):
                        m.geom_group[g] = 4
            contacts = [e for e in events if e["state"] == "VALID_STRIKE"]
            focus = (
                min(
                    contacts,
                    key=lambda e: abs(
                        e["time_s"] - (shot["start_s"] + shot["end_s"]) / 2
                    ),
                )
                if contacts
                else None
            )
            look = None
            distance = None
            for k in range(shot["frames"]):
                source_time = shot["start_s"] + k / 50 * shot["speed"]
                step = min(
                    manifest["steps"] - 1,
                    round(source_time / manifest["physics"]["dt"]),
                )
                mujoco.mj_setState(m, d, states[step], manifest["state_spec"])
                mujoco.mj_forward(m, d)
                positions = {
                    n: d.xpos[m.body(n + "/trunk_base").id].copy()
                    for n in manifest["spawns"]
                }
                eliminated = {
                    e["actor"]
                    for e in events
                    if e["state"] in ("RING_OUT", "DOWN_COUNT")
                    and e["time_s"] <= source_time
                }
                alive = [n for n in positions if n not in eliminated]
                if shot["view"] in ("pair", "collision") and focus:
                    selected = [focus["actor"], focus["target"]]
                else:
                    selected = alive or list(positions)
                points = np.array([positions[n] for n in selected])
                if shot["view"] == "wide":
                    target = np.array([0.0, 0.0, 0.12])
                    target_distance = 2.65 * manifest["config"]["radius_m"] / 0.75
                else:
                    target = points.mean(axis=0)
                    target[2] = max(0.10, float(target[2]) + 0.04)
                    span = float(
                        np.max(
                            np.linalg.norm(
                                points[:, :2] - points[:, :2].mean(axis=0), axis=1
                            )
                        )
                    )
                    target_distance = max(1.08, span * 2.5 + 0.65)
                    if vertical:
                        target_distance = max(1.65, target_distance * 1.3)
                look = target if look is None else look + 0.09 * (target - look)
                distance = (
                    target_distance
                    if distance is None
                    else distance + 0.09 * (target_distance - distance)
                )
                camera.lookat[:] = look
                camera.distance = distance
                camera.azimuth = 112 + math.degrees(manifest["config"]["heading_rad"])
                camera.elevation = -34 if shot["view"] == "wide" else -25
                renderer.update_scene(d, camera=camera, scene_option=opt)
                frame = Image.fromarray(renderer.render()).convert("RGB")
                draw = ImageDraw.Draw(frame)
                scale = width / 1920
                top = round(90 * max(scale, 0.65))
                draw.rectangle((0, 0, width, top), fill=(12, 17, 25))
                label(
                    draw,
                    (round(32 * scale), round(18 * max(scale, 0.65))),
                    "DUCK FU",
                    max(26, round(37 * scale)),
                    bold=True,
                )
                label(
                    draw,
                    (width - 25, 20),
                    "SIMULATION",
                    max(18, round(21 * scale)),
                    color=(169, 181, 197),
                    anchor="ra",
                )
                team_counts = {
                    team: sum(manifest["spawns"][n]["team"] == team for n in alive)
                    for team in COLORS
                }
                match_label = (
                    f"RED {team_counts['RED']}    vs    BLUE {team_counts['BLUE']}"
                )
                label(
                    draw,
                    (width / 2, top / 2),
                    match_label,
                    max(20, round(27 * scale)),
                    bold=True,
                    anchor="mm",
                )
                for n, p in positions.items():
                    if n in eliminated:
                        continue
                    xy = project(p + [0, 0, 0.19], renderer.scene, m, width, height)
                    if xy is None:
                        continue
                    x, y = xy
                    spawn = manifest["spawns"][n]
                    color = COLORS[spawn["team"]]
                    text = spawn["display_name"]
                    font = ImageFont.truetype(BOLD, max(19, round(24 * scale)))
                    box = draw.textbbox((x, y), text, font=font, anchor="ms")
                    draw.rounded_rectangle(
                        (box[0] - 10, box[1] - 6, box[2] + 10, box[3] + 6),
                        radius=6,
                        fill=(18, 25, 35),
                        outline=color,
                        width=2,
                    )
                    draw.text((x, y), text, font=font, fill=color, anchor="ms")
                bottom = height - (160 if vertical else round(115 * max(scale, 0.65)))
                draw.rectangle((0, bottom, width, height), fill=(12, 17, 25))
                cap = shot["caption"]
                cap_size = max(20, round(29 * scale))
                while (
                    ImageFont.truetype(BOLD, cap_size).getlength(cap) > width - 50
                    and cap_size > 13
                ):
                    cap_size -= 1
                label(
                    draw,
                    (width / 2, bottom + 18),
                    cap,
                    cap_size,
                    bold=True,
                    anchor="mt",
                )
                speed = (
                    f"{shot['speed']:g}x SLOW REPLAY"
                    if shot["speed"] < 1
                    else (
                        "CONTINUOUS 1x" if plan["continuous"] else "MATCH 1x / EDITED"
                    )
                )
                clock = f"{source_time:05.2f}s"
                label(
                    draw,
                    (25, height - 22),
                    speed,
                    max(18, round(20 * scale)),
                    color=(255, 205, 105) if shot["speed"] < 1 else (173, 190, 206),
                    anchor="ls",
                )
                label(
                    draw,
                    (width - 25, height - 22),
                    clock,
                    max(18, round(20 * scale)),
                    anchor="rs",
                )
                if plan["edition"] == "technical" and focus and shot["speed"] < 1:
                    impulse = focus["forward_impulse_Ns"]
                    label(
                        draw,
                        (30, top + 25),
                        f"MEASURED RESPONSE / {impulse:.3f} N s",
                        max(20, round(25 * scale)),
                        color=(255, 205, 105),
                    )
                output_time = frame_number / 50
                writer.append_data(np.asarray(frame))
                provenance.write(
                    json.dumps(
                        dict(
                            frame=frame_number,
                            output_time_s=output_time,
                            shot=shot_index,
                            run=key,
                            source_step=step,
                            source_time_s=step * manifest["physics"]["dt"],
                            speed=shot["speed"],
                            view=shot["view"],
                            camera_read_only=True,
                        )
                    )
                    + "\n"
                )
                if frame_number in (100, 300, 700):
                    frame.save(out.with_name(out.stem + f"-frame-{frame_number}.jpg"))
                frame_number += 1
            renderer.close()
    writer.close()
    metadata = dict(
        schema="duckfu.film.v1",
        edition=plan["edition"],
        frames=frame_number,
        fps=50,
        width=width,
        height=height,
        duration_s=frame_number / 50,
        continuous=plan["continuous"],
        speed_from_recorded_states=True,
        synthetic_inbetween_frames=False,
        silent=True,
        renderer_sha256=digest(__file__),
        plan_sha256=digest(plan_path),
        video_sha256=digest(out),
        provenance_sha256=digest(out.with_suffix(".frames.jsonl")),
        film_gate="PENDING_HUMAN_REVIEW",
    )
    write_json(out.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    a = p.parse_args()
    render(a.plan, a.out, a.width, a.height)


if __name__ == "__main__":
    main()
