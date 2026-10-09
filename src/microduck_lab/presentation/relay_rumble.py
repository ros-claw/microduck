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
from .duckverse import Film, COLORS
from ..arena.relay_rumble import CHARACTERS

from ..arena.episode import STATE, sha

FONT = BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
COLORS = {**COLORS, "graphite": (255, 155, 141)}


class RelayFilm(Film):
    def __init__(self, path, width=1280, height=720):
        super().__init__(path, width, height)
        self.island = (
            self.audit["config"]["grid"] * (self.audit["config"]["grid"] // 2)
            + self.audit["config"]["grid"] // 2
        )
        self.m.vis.headlight.ambient[:] = 0.45
        self.m.vis.headlight.diffuse[:] = 0.65
        self.font = ImageFont.truetype(BOLD, 18)
        self.big = ImageFont.truetype(BOLD, 30)
        self.intent_rows = []
        with gzip.open(self.path / "decisions.jsonl.gz", "rt") as log:
            for line in log:
                self.intent_rows.append(json.loads(line))
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
        scores = dict.fromkeys(self.names, 0)
        progress = dict.fromkeys(self.names, 0.0)
        goal = 12
        captures = 0
        for e in self.audit["events"]:
            if e["time"] > t:
                continue
            if "damage" in e:
                damages[e["tile"]] = e["damage"]
            if e["state"] == "ELIMINATED":
                alive.discard(e["body_id"])
            if e["state"] == "SCOREBOARD" and terminal is None:
                scores = e["scores"]
                progress = e.get("progress", progress)
                goal = e.get("tile", goal)
            if e["state"] == "BEACON_CAPTURE":
                scores = e["scores"]
                captures += 1
            if e["state"] == "BEACON_MOVED":
                goal = e["tile"]
            if e["state"] == "TERMINAL":
                terminal = e
                scores = e.get("loaded_crown_scores_s", scores)
        self.island = goal
        stations = {12, 2, 14, 22, 10}
        for tile, g in self.tiles.items():
            v = min(1, damages[tile])
            self.m.geom_rgba[g] = (
                [0.92, 0.67, 0.12, 1]
                if tile == self.island
                else [0.22, 0.56, 0.57, 1]
                if tile in stations
                else [0.35 + 0.45 * v, 0.47 - 0.29 * v, 0.52 - 0.38 * v, 1]
            )
        c = mujoco.MjvCamera()
        c.lookat[:] = [0, 0, -0.03]
        c.distance = 3.65 if self.w > self.h else 4.8
        c.azimuth = 130
        c.elevation = -42
        if shot == "contact":
            contact_events = [
                e for e in self.audit["events"] if e["state"] == "BODY_CONTACT"
            ]
            event = (
                min(contact_events, key=lambda e: abs(e["time"] - t))
                if contact_events
                else None
            )
            touch = (
                max(event["contacts"], key=lambda c: c["force_N"]) if event else None
            )
            pair = (touch["a"], touch["b"]) if touch else None
            ps = (
                [self.d.xpos[self.bids[n]] for n in pair]
                if pair
                else [self.d.xpos[self.bids[n]] for n in alive]
            )
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
        elif shot == "follow":
            gpos = self.d.xpos[self.m.body(f"tile_{goal}").id]
            leader = (
                min(
                    alive,
                    key=lambda n: np.linalg.norm(
                        self.d.xpos[self.bids[n]][:2] - gpos[:2]
                    ),
                )
                if alive
                else self.names[0]
            )
            p = self.d.xpos[self.bids[leader]]
            c.lookat[:] = p + [0, 0, 0.02]
            c.distance = 1.35
            c.azimuth = 125
            c.elevation = -28
        elif shot == "top":
            c.elevation = -78
            c.distance = 3.5 if self.w > self.h else 4.7
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
            text = CHARACTERS[name]["name"] + (" 淘汰" if name not in alive else "")
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
        draw.text((22, 10), "鸭鸭抢岛 · 追逐争夺战", font=self.big, fill="white")
        cumulative = True
        hold = self.audit["config"]["points_to_win"]
        goal_names = {12: "中心岛", 2: "西岛", 14: "北岛", 22: "东岛", 10: "南岛"}
        if caption is None:
            caption = f"第{captures + 1}轮 · 争夺{goal_names[goal]} · 独占承重{self.audit['config']['claim_s']:g}秒得1分，先夺{hold}分获胜"
            recent = [
                e
                for e in self.audit["events"]
                if e["state"] == "BEACON_CAPTURE" and 0 <= t - e["time"] < 0.8
            ]
            if recent:
                e = recent[-1]
                caption = f"{CHARACTERS[e['body_id']]['name']}夺下{goal_names[e['tile']]}！目标转移：{goal_names[goal]}"
        if terminal:
            caption = (
                ("本局获胜：" + CHARACTERS[terminal["winner"]]["name"])
                if terminal.get("winner")
                else "本局平局 · 不强行指定赢家"
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
            draw.text((22, 8), "鸭鸭抢岛 · 追逐争夺战", font=self.big, fill="white")
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
                    (x, y, x + cell - 12, y + 58), radius=6, fill=(12, 19, 29)
                )
                draw.text(
                    (x + 9, y + 3),
                    f"{CHARACTERS[name]['name']} · 淘汰"
                    if name not in alive
                    else f"{CHARACTERS[name]['name']}  {int(score)}/{hold:g}分 · {CHARACTERS[name]['trait']}",
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
            f"开发预览 · 真实接触仿真 | {speed:g}倍速 | 比赛 {t:05.2f}秒 | 同一局连续录像",
            font=self.small,
            fill=(202, 214, 224),
        )
        mapx, mapy, unit = self.w - 154, 118, 22
        draw.rounded_rectangle(
            (mapx - 12, mapy - 10, mapx + 122, mapy + 131), radius=8, fill=(12, 19, 29)
        )
        for tile in self.tiles:
            px = mapx + (tile // 5) * unit
            py = mapy + (4 - tile % 5) * unit
            draw.rectangle(
                (px, py, px + 18, py + 18),
                fill=(255, 199, 65)
                if tile == goal
                else (45, 95, 101)
                if tile in stations
                else (87, 71, 69)
                if damages[tile] > 0.8
                else (60, 68, 79),
            )
        pitch = self.audit["config"]["tile_size"] + self.audit["config"]["gap"]
        for name in alive:
            p = self.d.xpos[self.bids[name]]
            px = mapx + (p[0] / pitch + 2) * unit + 9
            py = mapy + (2 - p[1] / pitch) * unit + 9
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=COLORS[name])
        draw.text(
            (mapx - 2, mapy + 111),
            f"目标：{goal_names[goal]}",
            font=self.font,
            fill=(215, 222, 230),
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


def render_relay(run, out, preview=None):
    f = RelayFilm(run)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if preview is not None:
        Image.fromarray(f.frame(preview, shot="wide")).save(out)
        f.close()
        return
    recorded_end = (len(f.states) - 1) * f.audit["config"]["dt"]
    terminal = next(
        (e["time"] for e in f.audit["events"] if e["state"] == "TERMINAL"), recorded_end
    )
    end = min(recorded_end, terminal + 0.002)
    contacts = [
        e
        for e in f.audit["events"]
        if e["state"] == "BODY_CONTACT" and 1 < e["time"] < terminal - 0.15
    ]
    chosen = []
    for e in sorted(
        contacts, key=lambda e: max(c["force_N"] for c in e["contacts"]), reverse=True
    ):
        if all(abs(e["time"] - t) > 4 for t in chosen):
            chosen.append(e["time"])
        if len(chosen) == 3:
            break
    captures = [e["time"] for e in f.audit["events"] if e["state"] == "BEACON_CAPTURE"]
    intervals = [(max(0, t - 0.12), min(end, t + 0.40), "contact") for t in chosen]
    intervals += [
        (max(0, e["time"] - 0.8), min(end, e["time"] - 0.25), "fall")
        for e in [e for e in f.losses if e["time"] < terminal][:2]
    ]
    times = []
    t = 0.0
    while t < end:
        shot = "top" if t < 1.5 else "wide"
        speed = 1.0
        for a, b, view in intervals:
            if a <= t <= b:
                speed = 0.25
                shot = view
                break
        if speed == 1 and any(c - 0.2 < t < c + 0.45 for c in captures):
            speed = 0.5
            shot = "follow"
        if shot == "wide" and int(t // 4) % 3 == 1:
            shot = "follow"
        caption = (
            "张三：抢点  /  二呆：绕行  /  老六：截路  /  卷王：追分"
            if t < 2.5
            else None
        )
        times.append((t, speed, shot, caption))
        t += speed / 50
    for _ in range(75):
        times.append((end, 0, "wide", None))
    with imageio.get_writer(
        str(out), fps=50, codec="libx264", quality=8, macro_block_size=2
    ) as w:
        for t, speed, shot, caption in times:
            w.append_data(f.frame(t, speed, shot, caption))
    f.close()
    with gzip.open(out.with_suffix(".frames.jsonl.gz"), "wt") as log:
        for row in f.records:
            log.write(json.dumps(row) + "\n")
    manifest = dict(
        source=str(Path(run).resolve()),
        source_audit_sha256=sha(Path(run) / "audit.json"),
        source_trajectory_sha256=f.audit["hashes"]["trajectory.npz"],
        seed=f.audit["seed"],
        fps=50,
        frames=len(times),
        duration_s=len(times) / 50,
        video_sha256=sha(out),
        renderer_source_sha256=sha(Path(__file__)),
        read_only=True,
        chronological=True,
        language="zh-CN",
        audio="none",
        status="unaccepted development preview",
        editing="continuous single match; event-driven slow motion; paused final state",
        labels="Chinese character names; render-only colour badges; original robot materials",
    )
    out.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest
