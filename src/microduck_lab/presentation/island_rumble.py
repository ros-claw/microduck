"""Read-only greybox film. Labels identify bodies; cracks visualize recorded damage."""

import json
import gzip
from pathlib import Path
import math
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .duckverse import Film, FONT, BOLD, COLORS
from ..arena.episode import STATE, sha


class RumbleFilm(Film):
    def __init__(self, path, width=1280, height=720):
        super().__init__(path, width, height)
        self.island = (
            self.audit["config"]["grid"] * (self.audit["config"]["grid"] // 2)
            + self.audit["config"]["grid"] // 2
        )
        self.m.vis.headlight.ambient[:] = 0.45
        self.m.vis.headlight.diffuse[:] = 0.65
        self.font = ImageFont.truetype(BOLD, 19)
        self.big = ImageFont.truetype(BOLD, 30)
        self.small = ImageFont.truetype(FONT, 20)

    def project(self, p):
        cam = self.renderer.scene.camera[0]
        delta = np.asarray(p) - cam.pos
        forward = np.asarray(cam.forward)
        up = np.asarray(cam.up)
        right = np.cross(forward, up)
        z = float(np.dot(delta, forward))
        if z <= 0:
            return None
        scale = self.h / (2 * math.tan(math.radians(self.m.vis.global_.fovy) / 2))
        return self.w / 2 + np.dot(delta, right) / z * scale, self.h / 2 - np.dot(
            delta, up
        ) / z * scale

    def frame(self, t, speed=1, shot="wide", caption=None):
        i = min(round(t / self.audit["config"]["dt"]), len(self.states) - 1)
        mujoco.mj_setState(self.m, self.d, self.states[i], STATE)
        mujoco.mj_forward(self.m, self.d)
        damages = {tile: 0 for tile in self.tiles}
        alive = set(self.names)
        terminal = None
        scores = dict.fromkeys(self.names, 0.0)
        for e in self.audit["events"]:
            if e["time"] > t:
                continue
            if "damage" in e:
                damages[e["tile"]] = e["damage"]
            if e["state"] == "ELIMINATED":
                alive.discard(e["body_id"])
            if e["state"] == "SCOREBOARD" and terminal is None:
                scores = e["scores"]
            if e["state"] == "TERMINAL":
                terminal = e
                scores = e.get("loaded_crown_scores_s", scores)
        for tile, g in self.tiles.items():
            v = min(1, damages[tile])
            self.m.geom_rgba[g] = (
                [0.92, 0.67, 0.12, 1]
                if tile == self.island
                else [0.35 + 0.45 * v, 0.47 - 0.29 * v, 0.52 - 0.38 * v, 1]
            )
        c = mujoco.MjvCamera()
        c.lookat[:] = [0, 0, -0.03]
        c.distance = 2.05 if self.w > self.h else 2.9
        c.azimuth = 130
        c.elevation = -42
        if shot == "contact":
            ps = [self.d.xpos[self.bids[n]] for n in alive]
            c.lookat[:] = np.mean(ps, axis=0) if ps else [0, 0, 0]
            c.distance = 1.12 if self.w > self.h else 1.65
            c.elevation = -23
            c.azimuth = 100
        elif shot == "fall":
            near = min(self.losses, key=lambda e: abs(e["time"] - t))
            p = self.d.xpos[self.bids[near["body_id"]]]
            c.lookat[:] = [p[0], p[1], float(np.clip(p[2] - 0.06, -0.45, 0.02))]
            c.distance = 1.7
            c.elevation = -18
            c.azimuth = math.degrees(math.atan2(p[1], p[0]))
        elif shot == "top":
            c.elevation = -78
            c.distance = 2.0 if self.w > self.h else 2.8
        self.renderer.update_scene(self.d, camera=c, scene_option=self.opt)
        # Render-only coloured badges; no model physics, mass or controls changed.
        for name, bid in self.bids.items():
            scene = self.renderer.scene
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.012, 0.012]),
                self.d.xpos[bid] + [0, 0, 0.15],
                np.eye(3).ravel(),
                np.array([v / 255 for v in COLORS[name]] + [1.0]),
            )
            scene.ngeom += 1

        # Scene-only marks participate in depth testing, so they never draw
        # through feet. These geoms do not exist in the simulation model.
        def line(p0, p1, colour, width):
            scene = self.renderer.scene
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array(colour),
            )
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array(p0), np.array(p1)
            )
            scene.ngeom += 1

        for tile, v in damages.items():
            if v < 0.1 or tile == self.island:
                continue
            bid = self.m.body(f"tile_{tile}").id
            rot = self.d.xmat[bid].reshape(3, 3)
            size = self.audit["config"]["tile_size"]
            xy = [
                (-0.42, -0.12),
                (-0.12, 0.08),
                (0.05, -0.07),
                (0.20, 0.13),
                (0.43, 0.2),
            ]
            pts = [
                self.d.xpos[bid] + rot @ np.array([x * size, y * size, 0.026])
                for x, y in xy[: min(5, 2 + int(v * 3))]
            ]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                line(p0, p1, [0.25, 0.10, 0.07, 1], 0.0008)
        radius = self.audit["config"].get("claim_radius_m", 0)
        if radius:
            bid = self.m.body(f"tile_{self.island}").id
            rot = self.d.xmat[bid].reshape(3, 3)
            pts = [
                self.d.xpos[bid]
                + rot @ np.array([radius * math.cos(a), radius * math.sin(a), 0.026])
                for a in np.linspace(0, 2 * math.pi, 65)
            ]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                line(p0, p1, [1, 1, 0.94, 1], 0.0015)
        pixels = self.renderer.render()
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        labels = []
        for name, bid in self.bids.items():
            p = self.project(self.d.xpos[bid] + [0, 0, 0.17])
            if p is None or (
                name not in alive
                and not (8 < p[0] < self.w - 8 and 112 < p[1] < self.h - 115)
            ):
                continue
            text = name.upper() + ("  OUT" if name not in alive else "")
            width = draw.textlength(text, font=self.font) + 20
            x = float(np.clip(p[0] - width / 2, 8, self.w - width - 8))
            y = float(np.clip(p[1] - 28, 112, self.h - 90))
            for old in labels:
                if x < old[2] and x + width > old[0] and abs(y - old[1]) < 30:
                    y = old[1] + 34
            labels.append((x, y, x + width))
            draw.line([p, (x + width / 2, y + 27)], fill=COLORS[name], width=2)
            draw.rounded_rectangle(
                (x, y, x + width, y + 28),
                radius=6,
                fill=(15, 21, 30),
                outline=COLORS[name],
                width=2,
            )
            draw.text((x + 10, y + 2), text, font=self.font, fill=COLORS[name])
        draw.rectangle((0, 0, self.w, 96), fill=(12, 19, 29))
        draw.text((22, 10), "ISLAND RUMBLE", font=self.big, fill="white")
        cumulative = self.audit["config"].get("score_mode") == "cumulative"
        hold = self.audit["config"]["claim_s"]
        if caption is None:
            phase = self.audit["config"]["final_at_s"]
            caption = (
                f"CONTROL THE CIRCLE. FIRST TO {hold:g}s WINS."
                if cumulative
                else "STEP ON IT. CRACK IT. CLAIM THE GOLD ISLAND."
                if t < phase
                else f"HOLD THE WHITE CIRCLE FOR {hold:g}s — rivals can contest it"
                if radius
                else f"FINAL ISLAND: hold it alone, upright, for {hold:g} seconds"
            )
        if terminal:
            caption = (
                ("WINNER: " + terminal["winner"].upper())
                if terminal.get("winner")
                else "DRAW — no supported winner"
            )
        if self.w < self.h and len(caption) > 45:
            words = caption.split()
            line = ""
            lines = []
            for word in words:
                if len(line + " " + word) > 43:
                    lines.append(line)
                    line = word
                else:
                    line = (line + " " + word).strip()
            lines.append(line)
            draw.rectangle((0, 0, self.w, 110), fill=(12, 19, 29))
            draw.text((22, 8), "ISLAND RUMBLE", font=self.big, fill="white")
            for j, line in enumerate(lines):
                draw.text(
                    (22, 46 + j * 25), line, font=self.small, fill=(255, 217, 133)
                )
        else:
            draw.text((22, 55), caption, font=self.small, fill=(255, 217, 133))
        if cumulative:
            columns = 4 if self.w > self.h else 2
            cell = (self.w - 32) / columns
            for j, name in enumerate(self.names):
                x = 16 + (j % columns) * cell
                y = self.h - 102 - (j // columns) * 56
                score = min(hold, scores.get(name, 0))
                draw.rounded_rectangle(
                    (x, y, x + cell - 12, y + 50), radius=6, fill=(12, 19, 29)
                )
                draw.text(
                    (x + 9, y + 3),
                    f"{name.upper()} OUT"
                    if name not in alive
                    else f"{name.upper()} {score:.2f}/{hold:g}s",
                    font=self.font,
                    fill=COLORS[name],
                )
                draw.rectangle(
                    (x + 9, y + 32, x + cell - 22, y + 39), fill=(60, 66, 76)
                )
                if score > 0:
                    draw.rectangle(
                        (x + 9, y + 32, x + 9 + (cell - 31) * score / hold, y + 39),
                        fill=COLORS[name],
                    )
        draw.rectangle((0, self.h - 42, self.w, self.h), fill=(12, 19, 29))
        draw.text(
            (18, self.h - 32),
            f"PHYSICS PROTOTYPE | {speed:g}x | t={t:05.2f}s | seed {self.audit['seed']}",
            font=self.small,
            fill=(202, 214, 224),
        )
        self.records.append(
            dict(
                frame=len(self.records),
                state_index=i,
                sim_time_s=i * self.audit["config"]["dt"],
                speed=speed,
                shot=shot,
            )
        )
        return np.asarray(image)


def render(run, out, mode="prototype", preview=None):
    width, height = (720, 1280) if mode == "vertical" else (1280, 720)
    f = RumbleFilm(run, width, height)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if preview is not None:
        Image.fromarray(
            f.frame(preview, shot="contact" if preview > 5 else "wide")
        ).save(out)
        f.close()
        return
    contact = f.first_contact
    end = (len(f.states) - 1) * f.audit["config"]["dt"]
    loss = min((e["time"] for e in f.losses), default=end + 10)
    final = next(
        (e["time"] for e in f.audit["events"] if e["state"] == "TERMINAL"), end - 1
    )
    candidates = [
        e
        for e in f.audit["events"]
        if e["state"] == "BODY_CONTACT" and loss + 0.3 < e["time"] < final - 0.3
    ]
    intense = (
        max(candidates, key=lambda e: max(x["force_N"] for x in e["contacts"]))["time"]
        if candidates
        else contact
    )
    cinematic = mode in ("hero", "vertical")
    times = []
    # A clearly labelled glimpse of the same match, then an explicit rewind.
    if cinematic and intense:
        for t in np.arange(max(0, intense - 0.15), min(end, intense + 0.65), 0.25 / 50):
            times.append(
                (
                    float(t),
                    0.25,
                    "contact",
                    "LATER IN THIS MATCH — control the circle. First to 1.5s wins.",
                    "hook",
                )
            )
    t = 0.0
    while t < end:
        slow = bool(contact is not None and contact - 0.12 <= t <= contact + 0.50)
        slow |= loss - 1.0 <= t <= loss - 0.3
        if cinematic and intense:
            slow |= intense - 0.15 <= t <= intense + 0.65
        speed = (
            0.25
            if slow
            else 0.5
            if cinematic and final - 0.4 <= t <= final + 0.3
            else 1.0
        )
        shot = (
            "contact"
            if contact and contact - 0.3 <= t < contact + 1.1
            else "fall"
            if loss - 1.3 <= t < loss + 0.3
            else "contact"
            if cinematic and t > loss + 0.3
            else "top"
            if t < 1.2
            else "wide"
        )
        caption = (
            "BACK TO THE START — stand in the circle to score"
            if cinematic and t < 1.2
            else None
        )
        if mode == "reference":
            speed, shot = 1.0, "wide"
        times.append((t, speed, shot, caption, "match"))
        t += speed / 50
    if cinematic:
        for _ in range(100):
            times.append((end, 0.0, "contact", None, "paused result"))
    writer = imageio.get_writer(
        str(out), fps=50, codec="libx264", quality=8, macro_block_size=2
    )
    try:
        for t, speed, shot, caption, segment in times:
            writer.append_data(f.frame(t, speed, shot, caption))
            f.records[-1]["segment"] = segment
    finally:
        writer.close()
        f.close()
    with gzip.open(out.with_suffix(".frames.jsonl.gz"), "wt") as log:
        for row in f.records:
            log.write(json.dumps(row) + "\n")
    manifest = dict(
        source=str(Path(run).resolve()),
        source_audit_sha256=sha(Path(run) / "audit.json"),
        source_trajectory_sha256=f.audit["hashes"]["trajectory.npz"],
        seed=f.audit["seed"],
        frames=len(times),
        fps=50,
        duration_s=len(times) / 50,
        video_sha256=sha(out),
        read_only=True,
        renderer_source_sha256=sha(Path(__file__)),
        chronological=not cinematic,
        editing="labelled same-match hook, explicit back-to-start, continuous match with event slow motion, paused result"
        if cinematic
        else "continuous same-match action",
        labels="body-anchored name labels + render-only coloured badges; native materials unchanged",
        audio="none (greybox)",
        limitations="conservative collision proxies; simulation, not real hardware",
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
