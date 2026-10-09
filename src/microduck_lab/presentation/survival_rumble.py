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


class SurvivalFilm(Film):
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
        self.small = ImageFont.truetype(FONT, 20)
        self.speaker_angles = {}

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
        finale = None
        tile_states = {}
        for e in self.audit["events"]:
            if e["time"] > t:
                continue
            if e["state"] == "FINALE_WARNING":
                finale = e
                goal = 12
            if "tile" in e and e["state"] in (
                "FINAL_WARNING",
                "RELEASED",
                "FALLING",
                "LOST",
            ):
                tile_states[e["tile"]] = e["state"]
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
        if finale:
            goal = 12
        self.island = goal
        stations = {12, 2, 14, 22, 10}
        for tile, g in self.tiles.items():
            v = min(1, damages[tile])
            self.m.geom_rgba[g] = (
                [0.92, 0.67, 0.12, 1]
                if tile == self.island
                else [0.48, 0.18, 0.13, 1]
                if tile_states.get(tile)
                in ("FINAL_WARNING", "RELEASED", "FALLING", "LOST")
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
            c.distance = 1.4
            # After the body passes below the deck, view from below its rim.
            # A camera above the deck otherwise films an occluding floor slab.
            c.elevation = 6 if p[2] < -0.12 else -30
            if p[2] >= -0.12:
                c.lookat[2] = 0.04
            c.azimuth = math.degrees(math.atan2(p[1], p[0]))
        elif shot.startswith("character_"):
            name = shot.removeprefix("character_")
            c.lookat[:] = self.d.xpos[self.bids[name]] + [0, 0, 0.02]
            c.distance = 1.2
            if name not in self.speaker_angles:
                others = [self.d.xpos[self.bids[n]][:2] for n in alive if n != name]
                away = self.d.xpos[self.bids[name]][:2] - (
                    np.mean(others, axis=0) if others else np.zeros(2)
                )
                self.speaker_angles[name] = (
                    math.degrees(math.atan2(-away[1], -away[0]))
                    if np.linalg.norm(away) > 0.025
                    else 128
                )
            c.azimuth = self.speaker_angles[name]
            c.elevation = -32
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
        elif shot == "duck_eye":
            leader = "lavender"
            head = self.d.xpos[self.m.body(leader + "/jaw_soft").id]
            forward = self.d.xmat[self.bids[leader]].reshape(3, 3)[:, 0].copy()
            forward[2] = 0
            forward /= max(float(np.linalg.norm(forward)), 1e-9)
            eye = head + 0.10 * forward + np.array([0, 0, 0.035])
            c.lookat[:] = (
                eye
                + 0.4 * math.cos(math.radians(8)) * forward
                + np.array([0, 0, -0.4 * math.sin(math.radians(8))])
            )
            c.distance = 0.4
            c.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
            c.elevation = -8
        elif shot in ("duel", "winner"):
            ps = [self.d.xpos[self.bids[n]] for n in alive]
            c.lookat[:] = np.mean(ps, axis=0) if ps else [0, 0, 0]
            spread = max((np.linalg.norm(p[:2] - c.lookat[:2]) for p in ps), default=0)
            c.distance = max(1.0, 1.0 + spread * 2.4)
            c.elevation = -25
            c.azimuth = 128
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
        empty = np.max(pixels, axis=2) < 4
        yy = np.linspace(0, 1, self.h)[:, None, None]
        background = np.broadcast_to(
            np.array([20, 32, 51])[None, None, :] * (1 - yy)
            + np.array([7, 13, 23])[None, None, :] * yy,
            pixels.shape,
        ).astype(np.uint8)
        pixels[empty] = background[empty]
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        labels = []
        for name, bid in self.bids.items():
            ray = self.d.xpos[bid] - self.renderer.scene.camera[0].pos
            ray /= max(float(np.linalg.norm(ray)), 1e-9)
            hit = np.array([-1], dtype=np.int32)
            mujoco.mj_ray(
                self.m,
                self.d,
                self.renderer.scene.camera[0].pos,
                ray,
                self.opt.geomgroup,
                1,
                -1,
                hit,
            )
            if hit[0] >= 0 and not self.m.body(
                int(self.m.geom_bodyid[int(hit[0])])
            ).name.startswith(name + "/"):
                continue
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
        draw.text((22, 10), "鸭鸭生存战 · 最后一鸭", font=self.big, fill="white")
        if terminal and terminal.get("winner"):
            n = terminal["winner"]
            draw.rounded_rectangle(
                (24, 114, 345, 173),
                radius=10,
                fill=(12, 19, 29),
                outline=COLORS[n],
                width=2,
            )
            draw.text(
                (40, 125),
                "唯一幸存者 · " + CHARACTERS[n]["name"],
                font=self.big,
                fill=COLORS[n],
            )
        draw.text(
            (self.w - 240, 22),
            "ROSClaw × Microduck",
            font=self.font,
            fill=(177, 195, 215),
        )
        cumulative = True
        hold = max(5, max(scores.values(), default=0))
        goal_names = {12: "中心岛", 2: "西岛", 14: "北岛", 22: "东岛", 10: "南岛"}
        if caption is None:
            caption = f"剩余{len(alive)}鸭 · 第{captures + 1}轮争夺{goal_names[goal]} · 抢点不判胜，最后一鸭存活才赢"
            if finale:
                deadline = finale["inner_release_s"]
                caption = (
                    f"剩余{len(alive)}鸭 · 回到中心！最后安全台倒计时 {max(0, deadline - t):04.1f}秒"
                    if t < deadline
                    else f"剩余{len(alive)}鸭 · 电机驱动摇摆台 · 唯一站稳的幸存者获胜"
                )
            recent = [
                e
                for e in self.audit["events"]
                if e["state"] == "BEACON_CAPTURE" and 0 <= t - e["time"] < 0.8
            ]
            if recent:
                e = recent[-1]
                caption = f"{CHARACTERS[e['body_id']]['name']}夺下{goal_names[e['tile']]}！目标转移：{goal_names[goal]}"
        if shot == "duck_eye":
            caption = "张三视角 · 四鸭争夺中心岛"
        if terminal:
            caption = (
                (
                    "本局获胜："
                    + CHARACTERS[terminal["winner"]]["name"]
                    + " · 其余三鸭已真实跌落淘汰"
                )
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
            draw.text((22, 8), "鸭鸭生存战 · 最后一鸭", font=self.big, fill="white")
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
                    else f"{CHARACTERS[name]['name']} · 存活 | 抢岛{int(score)}次",
                    font=self.font,
                    fill=COLORS[name],
                )
                draw.text(
                    (x + 9, y + 32),
                    CHARACTERS[name]["trait"] if name in alive else "已跌落 · 本局出局",
                    font=self.font,
                    fill=(177, 190, 202) if name in alive else (113, 124, 138),
                )
        draw.rectangle((0, self.h - 42, self.w, self.h), fill=(12, 19, 29))
        draw.text(
            (18, self.h - 32),
            f"开发预览 · 真实接触仿真 | {str(speed) + '倍速' if speed else '终局定格'} | 比赛 {t:05.2f}秒 | 同一局连续录像",
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
                else (18, 27, 39)
                if tile_states.get(tile) == "LOST"
                else (195, 65, 45)
                if tile_states.get(tile)
                in ("FINAL_WARNING", "RELEASED", "FALLING", "LOST")
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


def plan_survival(audit):
    """Single chronological match; only observed contacts/falls get slow motion."""
    end = (audit["steps"] - 1) * audit["config"]["dt"]
    terminal = next(
        (e["time"] for e in audit["events"] if e["state"] == "TERMINAL"), end
    )
    contacts = [
        e
        for e in audit["events"]
        if e["state"] == "BODY_CONTACT" and 2 < e["time"] < terminal - 0.2
    ]
    chosen = []
    for e in sorted(
        contacts, key=lambda e: max(c["force_N"] for c in e["contacts"]), reverse=True
    ):
        if all(abs(e["time"] - t) > 5 for t in chosen):
            chosen.append(e["time"])
        if len(chosen) == 3:
            break
    intervals = [(max(0, t - 0.18), min(end, t + 0.65), "contact") for t in chosen]
    losses = [e for e in audit["events"] if e["state"] == "ELIMINATED"]
    intervals += [
        (max(0, e["time"] - 1.0), min(end, e["time"] + 0.35), "fall") for e in losses
    ]
    captures = [e["time"] for e in audit["events"] if e["state"] == "BEACON_CAPTURE"]
    finale = next(
        (e["time"] for e in audit["events"] if e["state"] == "FINALE_WARNING"), end
    )
    # Reserve up to 22 seconds for event detail and final winner speech.
    normal = max(0.85, end / 88)
    rows = []
    t = 0.0
    while t < end:
        speed = normal
        shot = "top" if t < 1.8 else "wide"
        for a, b, view in intervals:
            if a <= t <= b:
                speed = 0.30
                shot = view
                break
        if speed == normal and any(c - 0.1 < t < c + 0.45 for c in captures):
            speed = 0.55
            shot = "follow"
        if t > finale and shot == "wide":
            alive = 4 - sum(e["time"] <= t for e in losses)
            if alive <= 2:
                shot = "duel"
        if shot == "wide" and 4.5 < t < 6.3:
            shot = "follow"
        if t >= terminal:
            shot = "winner"
            speed = 0.6
        rows.append(dict(t=t, speed=speed, shot=shot))
        t += speed / 50
    rows.extend(dict(t=end, speed=0, shot="winner") for _ in range(250))
    if len(rows) > 6000:
        raise ValueError("Film exceeds two minutes")
    return rows


def render_survival(run, out, preview=None, speech=None, plan_only=False):
    run = Path(run)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = json.loads((run / "audit.json").read_text())
    times = plan_survival(audit)
    if plan_only:
        out.write_text(
            json.dumps(
                dict(fps=50, source_audit_sha256=sha(run / "audit.json"), frames=times),
                ensure_ascii=False,
            )
        )
        return dict(frames=len(times), duration_s=len(times) / 50)
    f = SurvivalFilm(run)
    if preview is not None:
        Image.fromarray(
            f.frame(
                preview, shot="duel" if preview > audit["finale_started_s"] else "wide"
            )
        ).save(out)
        f.close()
        return dict(preview=str(out))
    cues = json.loads(Path(speech).read_text())["cues"] if speech else []
    with gzip.open(run / "decisions.jsonl.gz", "rt") as stream:
        decisions = [json.loads(line) for line in stream]
    subtitle_font = ImageFont.truetype(BOLD, 28)
    with imageio.get_writer(
        str(out), fps=50, codec="libx264", quality=8, macro_block_size=2
    ) as w:
        for j, row in enumerate(times):
            cue = next((c for c in cues if c["start_s"] <= j / 50 < c["end_s"]), None)
            shot = row["shot"]
            if cue and cue["evidence"]["kind"] == "decision":
                shot = "character_" + decisions[cue["evidence"]["index"]]["body_id"]
            pixels = f.frame(row["t"], row["speed"], shot)
            if cue:
                im = Image.fromarray(pixels)
                draw = ImageDraw.Draw(im)
                text = cue["text"]
                width = draw.textlength(text, font=subtitle_font)
                x = (f.w - width) / 2
                y = f.h - 156
                draw.rounded_rectangle(
                    (x - 15, y - 5, x + width + 15, y + 41), 8, fill=(10, 16, 26)
                )
                draw.text((x, y), text, font=subtitle_font, fill=(255, 244, 215))
                pixels = np.asarray(im)
            w.append_data(pixels)
    f.close()
    with gzip.open(out.with_suffix(".frames.jsonl.gz"), "wt") as log:
        for row in f.records:
            log.write(json.dumps(row) + "\n")
    manifest = dict(
        source=str(run.resolve()),
        source_audit_sha256=sha(run / "audit.json"),
        source_trajectory_sha256=audit["hashes"]["trajectory.npz"],
        seed=audit["seed"],
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
        outcome=audit["outcome"],
        final_alive=audit["final_alive"],
        editing="one continuous match; event-aligned close-ups and slow motion; five-second labelled paused final state",
        labels="Chinese identities and subtitles; render-only badges; native asset materials",
    )
    out.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest
