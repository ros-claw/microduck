"""Render actual recorded integration states; camera/UI never enter physics."""

import argparse
import gzip
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
        str(out),
        fps=50,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        ffmpeg_params=[
            "-crf",
            "22",
            "-preset",
            "fast",
            "-movflags",
            "+faststart",
            "-threads",
            "2",
        ],
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
                roles, body_status = {}, {}
                if manifest["config"]["schema"] == "duckfu.match.v2":
                    with gzip.open(
                        directory / "tactical_decisions.jsonl.gz", "rt"
                    ) as stream:
                        for line in stream:
                            row = json.loads(line)
                            roles.setdefault(row["sim_step"], {})[row["actor"]] = (
                                row.get("role", "")
                            )
                            body_status.setdefault(row["sim_step"], {})[
                                row["actor"]
                            ] = row
                cache[key] = (
                    manifest,
                    m,
                    d,
                    states,
                    events,
                    m.geom_group.copy(),
                    m.geom_rgba.copy(),
                    roles,
                    body_status,
                )
            manifest, m, d, states, events, groups, rgba, roles, body_status = cache[
                key
            ]
            renderer = Renderer(m, height=height, width=width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            opt = mujoco.MjvOption()
            m.geom_group[:] = groups
            m.geom_rgba[:] = rgba
            collision = shot["view"] == "collision"
            opt.geomgroup[3] = int(collision)
            if collision:
                opt.geomgroup[2] = 0
                opt.geomgroup[4] = 0
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
                for g in range(m.ngeom):
                    body_name = m.body(int(m.geom_bodyid[g])).name
                    owner = body_name.split("/")[0]
                    if "_combat_shell" in body_name and owner in manifest["spawns"]:
                        team_color = COLORS[manifest["spawns"][owner]["team"]]
                        m.geom_rgba[g] = [*(v / 255 for v in team_color), 0.45]
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
                event_time = (
                    manifest["duration_s"] if shot["kind"] == "result" else source_time
                )
                eliminated = {
                    e["actor"]
                    for e in events
                    if e["state"] in ("RING_OUT", "DOWN_COUNT")
                    and e["time_s"] <= event_time
                }
                alive = [n for n in positions if n not in eliminated]
                if shot.get("focus_pair"):
                    selected = shot["focus_pair"]
                elif shot.get("focus_actor"):
                    selected = [shot["focus_actor"]]
                elif shot["view"] in ("pair", "collision") and focus:
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
                    target_distance = max(0.88, span * 2.5 + 0.65)
                    if vertical:
                        target_distance = max(1.65, target_distance * 1.3)
                if "camera_distance_m" in shot:
                    target_distance = float(shot["camera_distance_m"])
                if "camera_offset_world_m" in shot:
                    target = target + np.array(shot["camera_offset_world_m"])
                look = target if look is None else look + 0.09 * (target - look)
                distance = (
                    target_distance
                    if distance is None
                    else distance + 0.09 * (target_distance - distance)
                )
                camera.lookat[:] = look
                camera.distance = distance
                camera.azimuth = 112 + math.degrees(manifest["config"]["heading_rad"])
                camera.elevation = shot.get(
                    "camera_elevation_deg", -34 if shot["view"] == "wide" else -25
                )
                renderer.update_scene(d, camera=camera, scene_option=opt)
                frame = Image.fromarray(renderer.render()).convert("RGB")
                draw = ImageDraw.Draw(frame)
                if plan["edition"] in ("hero", "short"):
                    radius = manifest["config"]["radius_m"]
                    ring = [
                        project(
                            [radius * math.cos(a), radius * math.sin(a), 0.0006],
                            renderer.scene,
                            m,
                            width,
                            height,
                        )
                        for a in np.linspace(
                            0,
                            2 * math.pi,
                            9 if manifest["config"]["arena"] == "fixed_octagon" else 97,
                        )
                    ]
                    for j in range(len(ring) - 1):
                        if ring[j] is not None and ring[j + 1] is not None:
                            draw.line(
                                [ring[j], ring[j + 1]],
                                fill=COLORS[
                                    "RED" if j < (len(ring) - 1) / 2 else "BLUE"
                                ],
                                width=max(2, width // 640),
                            )
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
                placed_labels = []
                for n, p in positions.items():
                    if n in eliminated:
                        continue
                    if not shot.get("show_names", True) or (
                        shot.get("focus_labels_only") and n not in selected
                    ):
                        continue
                    xy = project(p + [0, 0, 0.19], renderer.scene, m, width, height)
                    if xy is None:
                        continue
                    x, y = xy
                    spawn = manifest["spawns"][n]
                    color = COLORS[spawn["team"]]
                    text = spawn["display_name"]
                    motor_step = step - step % round(
                        1
                        / (manifest["physics"]["motor_hz"] * manifest["physics"]["dt"])
                    )
                    role = roles.get(motor_step, {}).get(n, "")
                    if role and role not in ("DUEL", "V1"):
                        text += " / " + role
                    font = ImageFont.truetype(BOLD, max(19, round(24 * scale)))
                    box = draw.textbbox((x, y), text, font=font, anchor="ms")
                    if plan.get("declutter_labels"):
                        original = (x, y)
                        x = min(
                            width - (box[2] - box[0]) / 2 - 15,
                            max((box[2] - box[0]) / 2 + 15, x),
                        )
                        box = draw.textbbox((x, y), text, font=font, anchor="ms")
                        while any(
                            not (
                                box[2] + 16 < b[0]
                                or box[0] - 16 > b[2]
                                or box[3] + 12 < b[1]
                                or box[1] - 12 > b[3]
                            )
                            for b in placed_labels
                        ):
                            y -= (box[3] - box[1]) + 20
                            box = draw.textbbox((x, y), text, font=font, anchor="ms")
                        if y < top + 45:
                            y = top + 45
                            box = draw.textbbox((x, y), text, font=font, anchor="ms")
                        placed_labels.append(box)
                        if abs(y - original[1]) > 8 or abs(x - original[0]) > 8:
                            draw.line([original, (x, y + 7)], fill=color, width=2)
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
                if shot.get("continuous_segment"):
                    speed = "UNCUT MATCH / 1x"
                if shot["kind"] == "result":
                    speed = "FINAL RESULT / FRAME HOLD"
                elif shot["kind"] == "contact_detail":
                    speed = "ACTUAL SOLVED CONTACT / FRAME HOLD"
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
                if shot.get("measurement"):
                    measured = shot["measurement"]
                    label(
                        draw,
                        (30, top + 25),
                        f"SOLVER NORMAL FORCE / {measured['normal_force_N']:.2f} N",
                        max(20, round(25 * scale)),
                        color=(255, 205, 105),
                    )
                    parts = [
                        m.body(int(m.geom_bodyid[g]))
                        .name.split("/")[-1]
                        .replace("_combat_shell", "")
                        .replace("_", " ")
                        .upper()
                        for g in measured["geometry_ids"]
                    ]
                    label(
                        draw,
                        (30, top + 65),
                        " / ".join(parts),
                        max(18, round(22 * scale)),
                        color=(205, 216, 230),
                    )
                    position = np.array(measured["position_world_m"])
                    tip = position + np.array(measured["normal_world"]) * 0.06
                    point_xy = project(position, renderer.scene, m, width, height)
                    tip_xy = project(tip, renderer.scene, m, width, height)
                    if point_xy and tip_xy:
                        draw.line([point_xy, tip_xy], fill=(255, 205, 105), width=3)
                        x, y = point_xy
                        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 205, 105))
                        label(
                            draw,
                            tip_xy,
                            "CONTACT NORMAL",
                            max(17, round(20 * scale)),
                            color=(255, 205, 105),
                        )
                body_overlay_data = None
                if shot.get("body_overlay"):
                    actor = shot["body_overlay"]
                    control_step = step - step % round(0.02 / manifest["physics"]["dt"])
                    status = body_status.get(control_step, {}).get(actor)
                    if status is None:
                        raise ValueError("Missing physical body status for overlay")
                    policy = status["command"]["policy"]
                    names = {
                        "recovery_candidate": "LEARNED RECOVERY",
                        "stand": "STANDING CONTROL",
                        "walk_candidate": "LOCOMOTION",
                        "dash_candidate": "CONTACT DRIVE",
                        "evade_candidate": "LATERAL STEP",
                        "sitstand": "GET-UP FALLBACK",
                    }
                    label(
                        draw,
                        (30, top + 25),
                        names.get(policy, policy.upper()),
                        max(20, round(25 * scale)),
                        color=(255, 205, 105),
                    )
                    b = status["observation"]
                    body_overlay_data = dict(
                        actor=actor,
                        source_control_step=control_step,
                        policy=policy,
                        upright=b["upright"],
                        foot_support=b["foot_support"],
                    )
                    label(
                        draw,
                        (30, top + 65),
                        f"UPRIGHT {b['upright']:.2f} / FOOT CONTACT {'YES' if b['foot_support'] else 'NO'}",
                        max(18, round(22 * scale)),
                        color=(205, 216, 230),
                    )
                elif (
                    not shot.get("measurement")
                    and plan["edition"] == "technical"
                    and focus
                    and shot["speed"] < 1
                ):
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
                            **(
                                {"body_overlay": body_overlay_data}
                                if body_overlay_data is not None
                                else {}
                            ),
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
        film_gate=plan["film_gate"],
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
