"""Read-only, fixed-camera normal-speed lab footage with frame provenance."""

import argparse
from pathlib import Path
import json
import os
import math

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import mujoco
from mujoco.rendering.classic.renderer import Renderer
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.combat.world import digest


def render(directory, out, width=1280, height=720, collision=False):
    directory, out = Path(directory), Path(out)
    if (
        out.exists()
        or out.with_suffix(".frames.jsonl").exists()
        or out.with_suffix(".json").exists()
    ):
        raise FileExistsError(out)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["capture"]:
        raise ValueError("Full physical capture required")
    for name in ("scene.mjb", "states.npy"):
        if digest(directory / name) != manifest["hashes"][name]:
            raise ValueError(f"Capture differs: {name}")
    m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    d = mujoco.MjData(m)
    states = np.load(directory / "states.npy", mmap_mode="r")
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = int(collision)
    if collision:
        opt.geomgroup[2] = 0
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
        # Display the external hulls, avoiding overlapping native support shells.
        for g in range(m.ngeom):
            if (
                m.geom_group[g] == 3
                and "_combat_shell" not in m.body(int(m.geom_bodyid[g])).name
            ):
                m.geom_group[g] = 4
        opt.geomgroup[4] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.02, 0, 0.13]
    camera.distance, camera.azimuth, camera.elevation = 1.15, 70, -40
    camera.azimuth += math.degrees(manifest["config"]["heading_rad"])
    renderer = Renderer(m, height=height, width=width)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, max(14, width // 70))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out),
        fps=50,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    stride = round(0.02 / manifest["physics"]["dt"])
    frames = 0
    with open(out.with_suffix(".frames.jsonl"), "w") as source:
        for step in range(0, manifest["steps"], stride):
            mujoco.mj_setState(m, d, states[step], manifest["state_spec"])
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=camera, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            cam = renderer.scene.camera[0]
            for i, (name, spawn) in enumerate(manifest["spawns"].items()):
                p = d.xpos[m.body(name + "/trunk_base").id] + [0, 0, 0.21]
                delta = p - cam.pos
                depth = float(np.dot(delta, cam.forward))
                if depth > 0:
                    scale = height / (
                        2 * math.tan(math.radians(m.vis.global_.fovy) / 2)
                    )
                    x = (
                        width / 2
                        + float(np.dot(delta, np.cross(cam.forward, cam.up)))
                        * scale
                        / depth
                    )
                    y = height / 2 - float(np.dot(delta, cam.up)) * scale / depth
                    label = (
                        "A / COACH"
                        if spawn["role"] == "coach_attacker"
                        else "B / " + spawn["role"].upper()
                    )
                    color = (
                        (75, 120, 230)
                        if spawn["role"] == "coach_attacker"
                        else (200, 90, 45)
                    )
                    draw.text(
                        (x, y),
                        label,
                        font=font,
                        fill=color,
                        anchor="ms",
                        stroke_width=1,
                        stroke_fill=(255, 255, 255),
                    )
            draw.rectangle((0, 0, width, 44), fill=(24, 28, 31))
            draw.text(
                (18, 12),
                "DUCK FU  /  DEVELOPMENT LAB  /  FIXED COACH + PUBLIC-STATE DEFENSE",
                font=font,
                fill=(239, 241, 243),
            )
            draw.rectangle((0, height - 38, width, height), fill=(24, 28, 31))
            text = f"CONTINUOUS 1x | {step * manifest['physics']['dt']:05.2f}s | visual-piece convex contacts | motor policies at 50 Hz"
            draw.text((18, height - 28), text, font=font, fill=(225, 229, 231))
            writer.append_data(np.asarray(frame))
            source.write(
                json.dumps(
                    dict(
                        frame=frames,
                        source_step=step,
                        source_time_s=step * manifest["physics"]["dt"],
                        speed=1.0,
                        run=str(directory),
                    )
                )
                + "\n"
            )
            if frames == 170:
                frame.save(out.with_suffix(".jpg"))
            frames += 1
    writer.close()
    renderer.close()
    result = dict(
        schema="duckfu.media.v1",
        frames=frames,
        fps=50,
        width=width,
        height=height,
        duration_s=frames / 50,
        speed=1.0,
        continuous=True,
        no_fx=True,
        silent=True,
        collision_overlay=collision,
        source_manifest_sha256=digest(directory / "manifest.json"),
        renderer_sha256=digest(__file__),
        video_sha256=digest(out),
        provenance_sha256=digest(out.with_suffix(".frames.jsonl")),
    )
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cut", choices=["reference"], default="reference")
    p.add_argument("--collision", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    a = p.parse_args()
    render(a.run, a.out, a.width, a.height, a.collision)
