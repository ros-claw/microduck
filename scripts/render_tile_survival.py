"""Clearly labelled paired runs; cameras read the verified states only."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.arena.episode import STATE, sha
from microduck_lab.arena.verify import verify


def frames(audit):
    t = 0.0
    while t < audit["duration"] - 0.00001:
        speed = 0.5 if audit["brain"] == "hold" and 5.8 <= t < 7.2 else 1.0
        yield t, speed
        t += 0.02 * speed


def render(response, hold, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    bold = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30
    )
    with imageio.get_writer(
        out, fps=50, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for number, path in enumerate((Path(response), Path(hold))):
            verify(path)
            audit = json.loads((path / "audit.json").read_text())
            z = dict(np.load(path / "trajectory.npz", allow_pickle=False))
            m = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
            d = mujoco.MjData(m)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, -0.25]
            camera.distance = 2.7
            camera.azimuth = 135
            camera.elevation = -28
            side = mujoco.MjvCamera()
            side.lookat[:] = [0, 0, -0.3]
            side.distance = 2.2
            side.azimuth = 135
            side.elevation = -6
            r = mujoco.Renderer(m, height=720, width=1280)
            sr = mujoco.Renderer(m, height=270, width=480)
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 0
            count = 0
            for t, speed in frames(audit):
                i = min(round(t / audit["dt"]), len(z["states"]) - 1)
                mujoco.mj_setState(m, d, z["states"][i], STATE)
                mujoco.mj_forward(m, d)
                states = ["LOCKED"] * 9
                alive = True
                for event in audit["events"]:
                    if event["time"] <= t:
                        if "tile" in event:
                            states[event["tile"]] = event["state"]
                        if event["state"] == "ELIMINATED":
                            alive = False
                for tile, state in enumerate(states):
                    m.geom_rgba[m.geom(f"tile_{tile}_geom").id] = (
                        [0.9, 0.65, 0.12, 1]
                        if state == "WARNING"
                        else [0.85, 0.2, 0.18, 1]
                        if state not in ("LOCKED", "WARNING")
                        else [0.23, 0.48, 0.56, 1]
                    )
                r.update_scene(d, camera=camera, scene_option=opt)
                image = Image.fromarray(r.render())
                sr.update_scene(d, camera=side, scene_option=opt)
                image.paste(Image.fromarray(sr.render()), (788, 390))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 96), fill=(15, 21, 29))
                title = (
                    "RUN A / RESPOND TO CURRENT WARNINGS"
                    if number == 0
                    else "RUN B / HOLD POSITION / SAME SEED"
                )
                draw.text((26, 12), title, font=bold, fill="white")
                desc = (
                    "Learned walking + visible-state feedback; no future schedule"
                    if number == 0
                    else "Control run: release removes real support; referee checks the fall"
                )
                draw.text((26, 55), desc, font=font, fill=(155, 218, 235))
                draw.rectangle((788, 360, 1268, 390), fill=(15, 21, 29))
                draw.text(
                    (800, 362),
                    "Side view / same physical state",
                    font=font,
                    fill="white",
                )
                draw.rectangle((0, 673, 1280, 720), fill=(15, 21, 29))
                status = "ALIVE" if alive else "ELIMINATED"
                draw.text(
                    (26, 682),
                    f"SIMULATION | {speed:g}x | t={t:05.2f}s | {status} | active tiles={sum(s in ('LOCKED', 'WARNING') for s in states)}",
                    font=font,
                    fill="white" if alive else (255, 150, 140),
                )
                writer.append_data(np.asarray(image))
                count += 1
                if number == 0 and abs(t - 8) < 0.01:
                    image.save(out.with_suffix(".jpg"))
            r.close()
            sr.close()
            records.append(
                dict(
                    run=str(path),
                    brain=audit["brain"],
                    seed=audit["seed"],
                    source_hashes=audit["hashes"],
                    outcome=audit["outcome"],
                    video_frames=count,
                )
            )
    manifest = dict(
        schema="microduck.duckverse.paired-film.v1",
        video_sha256=sha(out),
        fps=50,
        duration_s=sum(r["video_frames"] for r in records) / 50,
        runs=records,
        editing="Two different runs explicitly labelled A/B; hold release at 0.5x uses recorded states, no interpolation",
        physics_edited=False,
        audio="none",
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--response", required=True)
    p.add_argument("--hold", required=True)
    p.add_argument("--out", default="out/duckverse_dg02_en.mp4")
    a = p.parse_args()
    render(a.response, a.hold, a.out)
