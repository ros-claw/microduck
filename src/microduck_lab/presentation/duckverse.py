"""Event-based cameras over recorded states; no actor or physics edits."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
from mujoco.rendering.classic.renderer import Renderer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.arena.episode import STATE, sha

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = (12, 20, 32)
COLORS = {
    "lavender": (204, 176, 255),
    "cream": (255, 218, 134),
    "sky": (131, 221, 255),
    "graphite": (193, 207, 219),
}


class Film:
    def __init__(self, path, width=1280, height=720):
        self.path = Path(path)
        self.audit = json.loads((self.path / "audit.json").read_text())
        for filename, digest in self.audit["hashes"].items():
            if sha(self.path / filename) != digest:
                raise ValueError("Evidence digest mismatch: " + filename)
        with np.load(self.path / "trajectory.npz", allow_pickle=False) as z:
            self.states = z["states"]
        self.m = mujoco.MjModel.from_binary_path(str(self.path / "scene.mjb"))
        self.d = mujoco.MjData(self.m)
        self.w, self.h = width, height
        self.m.vis.global_.offwidth = max(width, 1280)
        self.m.vis.global_.offheight = max(height, 1280)
        self.m.vis.headlight.ambient[:] = 0.36
        self.m.vis.headlight.diffuse[:] = 0.8
        self.m.vis.headlight.specular[:] = 0.25
        self.renderer = Renderer(self.m, height=height, width=width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0
        self.names = list(self.audit["spawns"])
        self.bids = {n: self.m.body(n + "/trunk_base").id for n in self.names}
        self.tiles = {
            int(self.m.body(i).name.split("_")[1]): self.m.geom(
                self.m.body(i).name + "_geom"
            ).id
            for i in range(self.m.nbody)
            if self.m.body(i).name.startswith("tile_")
        }
        self.first_contact = next(
            (e["time"] for e in self.audit["events"] if e["state"] == "BODY_CONTACT"),
            None,
        )
        self.losses = [e for e in self.audit["events"] if e["state"] == "ELIMINATED"]
        self.records = []

    def close(self):
        self.renderer.close()

    def status(self, t):
        tiles = {i: "LOCKED" for i in self.tiles}
        alive = set(self.names)
        for e in self.audit["events"]:
            if e["time"] > t:
                continue
            if "tile" in e:
                tiles[e["tile"]] = e["state"]
            if e["state"] == "ELIMINATED":
                alive.discard(e["body_id"])
        return tiles, alive

    def shot(self, t, mode):
        if mode == "reference":
            return "wide", None
        if mode == "vertical":
            return "duel" if t > 25 else "follow", "lavender"
        contact = self.first_contact
        if contact and contact - 0.7 <= t < contact + 0.9:
            return "contact", None
        if any(e["time"] - 0.8 <= t < e["time"] + 0.35 for e in self.losses):
            return "fall", None
        if t < 3:
            return "wide", None
        if t < 8:
            return "follow", "cream"
        if 16.4 <= t < 18.4:
            return "follow", "lavender"
        if t < 13:
            return "follow", "lavender"
        if t < 18:
            return "wide", None
        if t < 23:
            return "follow", "graphite"
        if t < 25:
            return "wide", None
        if t > 32.9:
            return "winner", self.audit["outcome"].get("winner")
        return "duel", None

    def frame(self, t, speed=1, mode="hero", caption=None, overlay=False):
        i = min(round(t / self.audit["config"]["dt"]), len(self.states) - 1)
        mujoco.mj_setState(self.m, self.d, self.states[i], STATE)
        mujoco.mj_forward(self.m, self.d)
        tiles, alive = self.status(t)
        for tile, g in self.tiles.items():
            s = tiles[tile]
            self.m.geom_rgba[g] = (
                [0.93, 0.62, 0.13, 1]
                if s == "WARNING"
                else [0.8, 0.19, 0.19, 1]
                if s != "LOCKED"
                else [0.17, 0.48, 0.55, 1]
            )
        shot, name = self.shot(t, mode)
        c = mujoco.MjvCamera()
        c.lookat[:] = [0, 0, -0.15]
        c.distance = 3.5
        c.azimuth = 132
        c.elevation = -33
        if shot == "follow" and name in self.bids:
            p = self.d.xpos[self.bids[name]]
            c.lookat[:] = p + [0, 0, 0.005]
            c.distance = 1.0 if mode != "vertical" else 1.45
            c.elevation = -22
            c.azimuth = 118
        elif shot == "duck_eye" and name in self.bids:
            head = self.d.xpos[self.m.body(name + "/jaw_soft").id]
            forward = self.d.xmat[self.bids[name]].reshape(3, 3)[:, 0].copy()
            forward[2] = 0
            forward /= np.linalg.norm(forward)
            eye = head + 0.10 * forward + np.array([0, 0, 0.035])
            c.lookat[:] = eye + 0.4 * forward
            c.distance = 0.4
            c.azimuth = np.degrees(np.arctan2(-forward[1], -forward[0]))
            c.elevation = -8
        elif shot == "contact":
            c.lookat[:] = np.mean(
                [self.d.xpos[self.bids[n]] for n in ("cream", "sky")], axis=0
            )
            c.distance = 0.95
            c.elevation = -15
            c.azimuth = 116
        elif shot == "fall":
            near = min(self.losses, key=lambda e: abs(e["time"] - t))
            p = self.d.xpos[self.bids[near["body_id"]]]
            c.lookat[:] = [p[0], p[1], -0.26]
            c.distance = 1.6
            c.elevation = -9
            c.azimuth = 112
        elif shot in ("duel", "winner"):
            poses = [self.d.xpos[self.bids[n]] for n in alive] or [np.array([0, 0, 0])]
            c.lookat[:] = np.mean(poses, axis=0)
            c.distance = 1.25 if shot == "duel" else 0.85
            c.elevation = -20
            c.azimuth = 135
            if mode == "vertical":
                c.distance = 1.85
        self.opt.geomgroup[3] = int(overlay)
        self.renderer.update_scene(self.d, camera=c, scene_option=self.opt)
        pixels = self.renderer.render()
        empty = np.max(pixels, axis=2) < 4
        yy = np.linspace(0, 1, self.h)[:, None, None]
        background = np.broadcast_to(
            np.array([18, 29, 46])[None, None, :] * (1 - yy)
            + np.array([6, 13, 24])[None, None, :] * yy,
            pixels.shape,
        ).astype(np.uint8)
        pixels[empty] = background[empty]
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        scale = self.w / 1280
        if mode == "vertical":
            scale = 0.7
        f = ImageFont.truetype(FONT, round(22 * scale))
        b = ImageFont.truetype(BOLD, round(29 * scale))
        small = ImageFont.truetype(FONT, round(17 * scale))
        top = round(104 * scale)
        bottom = round(52 * scale)
        draw.rectangle((0, 0, self.w, top), fill=INK)
        draw.text(
            (round(24 * scale), round(12 * scale)),
            "LAST DUCK STANDING",
            font=b,
            fill="white",
        )
        subtitle = caption or (
            "4 robots. Shrinking floor. One supported survivor."
            if t < 4
            else "Visible warning > choose a neighbouring tile > walk"
            if t < 25
            else "Real body contacts / no pose animation"
            if shot == "contact"
            else "Support disappears. Gravity decides."
            if shot == "fall"
            else "Winner: " + str(self.audit["outcome"].get("winner")).upper()
            if t > 32.9
            else "The final tiles"
        )
        if mode == "vertical" and len(subtitle) > 42:
            subtitle = "Real contacts. Gravity decides."
        draw.text(
            (round(24 * scale), round(52 * scale)),
            subtitle,
            font=f,
            fill=(148, 220, 234),
        )
        if mode != "vertical":
            x = self.w - 470
            for n in ("lavender", "cream", "sky", "graphite"):
                draw.text(
                    (x, 15),
                    n.upper(),
                    font=small,
                    fill=COLORS[n] if n in alive else (75, 87, 99),
                )
                draw.text(
                    (x, 42),
                    "IN" if n in alive else "OUT",
                    font=small,
                    fill=COLORS[n] if n in alive else (75, 87, 99),
                )
                x += 116
        draw.rectangle((0, self.h - bottom, self.w, self.h), fill=INK)
        draw.text(
            (round(24 * scale), self.h - bottom + round(13 * scale)),
            f"SIMULATION  |  {speed:g}x  |  t={t:05.2f}s  |  seed {self.audit['seed']}",
            font=f,
            fill="white",
        )
        if speed < 1:
            box = (
                self.w - round(260 * scale),
                self.h - bottom - round(62 * scale),
                self.w - round(22 * scale),
                self.h - bottom - round(16 * scale),
            )
            draw.rounded_rectangle(box, radius=8, fill=(29, 46, 60))
            draw.text(
                (box[0] + 12, box[1] + 8),
                "CONTACT / SLOW" if shot == "contact" else "DETAIL / SLOW",
                font=f,
                fill=(255, 218, 134),
            )
        self.records.append(
            {
                "frame": len(self.records),
                "state_index": i,
                "sim_time": i * self.audit["config"]["dt"],
                "speed": speed,
                "shot": shot,
                "collision_overlay": overlay,
            }
        )
        return image


def timeline(audit, mode):
    if mode == "vertical":
        start, end = 20, min(audit["duration"], 34.8)
    else:
        start, end = 0, audit["duration"]
    t = start
    contact = next(
        (e["time"] for e in audit["events"] if e["state"] == "BODY_CONTACT"), -10
    )
    losses = [e["time"] for e in audit["events"] if e["state"] == "ELIMINATED"]
    while t < end:
        speed = 1.0
        if mode == "hero":
            if (
                16.8 <= t < 18.0
                or contact - 0.25 <= t < contact + 0.5
                or any(v - 0.75 <= t < v - 0.25 for v in losses)
            ):
                speed = 0.25
        if mode == "vertical" and contact - 0.25 <= t < contact + 0.5:
            speed = 0.25
        yield t, speed
        t += speed / 50


def render(path, out, mode="hero", preview=None):
    f = Film(
        path, 720 if mode == "vertical" else 1280, 1280 if mode == "vertical" else 720
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if preview is not None:
            f.frame(preview, mode=mode).save(out)
            return
        with imageio.get_writer(
            out,
            fps=50,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for t, speed in timeline(f.audit, mode):
                im = f.frame(t, speed, mode)
                writer.append_data(np.asarray(im))
            if mode == "hero":
                card = Image.new("RGB", (1280, 720), INK)
                draw = ImageDraw.Draw(card)
                draw.text(
                    (88, 220),
                    "LAST DUCK STANDING",
                    font=ImageFont.truetype(BOLD, 46),
                    fill="white",
                )
                draw.text(
                    (88, 300),
                    "WINNER / " + f.audit["outcome"]["winner"].upper(),
                    font=ImageFont.truetype(BOLD, 32),
                    fill=(148, 220, 234),
                )
                draw.text(
                    (88, 385),
                    "ROSClaw x Microduck / reproducible simulation",
                    font=ImageFont.truetype(FONT, 27),
                    fill="white",
                )
                draw.text(
                    (88, 438),
                    "github.com/ros-claw/microduck",
                    font=ImageFont.truetype(FONT, 27),
                    fill=(172, 190, 212),
                )
                for _ in range(100):
                    writer.append_data(np.asarray(card))
                    f.records.append(
                        {
                            "frame": len(f.records),
                            "kind": "end_card",
                            "state_index": None,
                        }
                    )

        frames = out.with_suffix(".frames.jsonl.gz")
        import gzip

        with gzip.open(frames, "wt") as fp:
            for record in f.records:
                fp.write(json.dumps(record) + "\n")
        manifest = {
            "schema": "microduck.duckverse.film.v1",
            "video": out.name,
            "video_sha256": sha(out),
            "fps": 50,
            "duration_s": len(f.records) / 50,
            "mode": mode,
            "seed": f.audit["seed"],
            "source_audit_sha256": sha(f.path / "audit.json"),
            "source_hashes": f.audit["hashes"],
            "frame_map": frames.name,
            "frame_map_sha256": sha(frames),
            "physics_edited": False,
            "audio": "none",
            "evidence_domain": "simulation",
            "editing": "Chronological single run; recorded 4 kHz states; camera/material/HUD only; no pose interpolation",
        }
        out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest))
    finally:
        f.close()
