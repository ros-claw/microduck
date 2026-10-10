"""Read-only single-shot greybox replay, with visible warning/release colours."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.arena.episode import STATE, sha
from replay_last_duck_standing import replay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True)
p.add_argument("--out", default="out/duckverse_dg01_en.mp4")
a = p.parse_args()
path = Path(a.run)
replay(path)
audit = json.loads((path / "audit.json").read_text())
z = dict(np.load(path / "trajectory.npz", allow_pickle=False))
m = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
d = mujoco.MjData(m)
c = mujoco.MjvCamera()
c.lookat[:] = [0, 0, -0.25]
c.distance = 2.7
c.azimuth = 135
c.elevation = -28
side = mujoco.MjvCamera()
side.lookat[:] = [0, 0, -0.3]
side.distance = 2.2
side.azimuth = 135
side.elevation = -6
side_renderer = mujoco.Renderer(m, height=270, width=480)
r = mujoco.Renderer(m, height=720, width=1280)
opt = mujoco.MjvOption()
opt.geomgroup[3] = 0
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 31)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
frames = []
with imageio.get_writer(
    out, fps=50, codec="libx264", quality=8, macro_block_size=1
) as writer:
    for frame in range(round(audit["duration"] * 50)):
        i = round(frame * 0.02 / audit["dt"])
        t = float(z["time"][i])
        mujoco.mj_setState(m, d, z["states"][i], STATE)
        mujoco.mj_forward(m, d)
        for tile in (1, 4):
            status = "LOCKED"
            for event in audit["events"]:
                if event["tile"] == tile and event["time"] <= t:
                    status = event["state"]
            m.geom_rgba[m.geom(f"tile_{tile}_geom").id] = (
                [0.9, 0.65, 0.12, 1]
                if status == "WARNING"
                else [0.85, 0.2, 0.18, 1]
                if status == "RELEASED"
                else [0.23, 0.48, 0.56, 1]
            )
        r.update_scene(d, camera=c, scene_option=opt)
        image = Image.fromarray(r.render())
        side_renderer.update_scene(d, camera=side, scene_option=opt)
        image.paste(Image.fromarray(side_renderer.render()), (788, 390))
        draw = ImageDraw.Draw(image)
        draw.rectangle((788, 360, 1268, 390), fill=(15, 21, 29))
        draw.text(
            (800, 362), "Side view / same physical state", font=font, fill="white"
        )
        draw.rectangle((0, 0, 1280, 94), fill=(15, 21, 29))
        draw.text(
            (26, 12), "LAST DUCK STANDING / PHYSICS GREYBOX", font=bold, fill="white"
        )
        desc = (
            "Learned motor policy walks across a 12 mm seam"
            if 1 <= t < 7
            else "Stand on the neighbouring tile"
            if 7 <= t < 8
            else "Occupied tile warning / release at 10 s"
            if 8 <= t < 10
            else "Weld released: gravity + real contacts"
            if t >= 10
            else "One duck / nine supported rigid-body tiles"
        )
        draw.text((26, 55), desc, font=font, fill=(155, 218, 235))
        draw.rectangle((0, 673, 1280, 720), fill=(15, 21, 29))
        draw.text(
            (26, 682),
            f"SIMULATION | 1x | t={t:05.2f}s | scripted calibration, no survival AI",
            font=font,
            fill="white",
        )
        array = np.asarray(image)
        writer.append_data(array)
        if frame == 440:
            image.save(out.with_suffix(".jpg"))
r.close()
side_renderer.close()
manifest = dict(
    source_hashes=audit["hashes"],
    video_sha256=sha(out),
    duration_s=audit["duration"],
    fps=50,
    camera="continuous fixed wide view with synchronized fixed side inset",
    speed=1,
    physics_edited=False,
    audio="none",
)
out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(out)
