"""English Hero/Technical cuts from a verified, immutable physical capture."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
import argparse, hashlib, json, math, subprocess, tempfile, wave
from pathlib import Path
import mujoco, numpy as np, imageio.v2 as imageio, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.parkour.cinema import CinematicEventTimeline, ShotDirector

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def sound(path, duration, events):
    """Synthetic Foley, explicitly not simulated microphone audio."""
    rate = 48000
    n = math.ceil(duration * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(42)
    signal = 0.012 * np.sin(2 * np.pi * 48 * t) + 0.008 * np.sin(2 * np.pi * 61 * t)

    def tone(at, seconds, freq, amp, noise=0.0):
        start = max(0, int(at * rate))
        length = min(n - start, int(seconds * rate))
        if length <= 0:
            return
        u = np.arange(length) / rate
        env = np.exp(-u / max(0.025, seconds / 4)) * np.minimum(u / 0.006, 1.0)
        signal[start : start + length] += (
            amp
            * env
            * (
                np.sin(
                    2 * np.pi * (freq * u - 0.16 * freq * u * u / max(seconds, 0.01))
                )
                + noise * rng.normal(0, 0.6, length)
            )
        )

    for at, kind in events:
        if kind == "FOOTSTEP":
            tone(at, 0.09, 180, 0.022, 0.6)
        elif kind == "JEV_DECISION":
            tone(at, 0.055, 1600, 0.035)
        elif kind == "HAZARD_TELEGRAPH":
            for dt in (0, 0.19, 0.38):
                tone(at + dt, 0.10, 880, 0.075)
        elif kind in ("BOSS_LAND", "BOSS_GATE_IMPACT"):
            tone(at, 1.0, 56, 0.36, 0.45)
        elif kind in ("IMPACT", "FALL", "BOSS_HIT_PROP"):
            tone(at, 0.5, 85, 0.18, 0.8)
        elif kind == "CRATE_RELEASE":
            tone(at, 0.25, 350, 0.05, 0.8)
            tone(at + 0.32, 0.45, 110, 0.14, 0.7)
        elif kind == "RECOVER":
            tone(at, 0.22, 540, 0.055)
        elif kind == "JUMP_CENTER":
            tone(at, 0.22, 260, 0.09)
        elif kind == "ROLL_CENTER":
            tone(at, 0.45, 190, 0.035, 0.8)
        elif kind == "FINISH":
            for j, f in enumerate([523.25, 659.25, 783.99]):
                tone(at + j * 0.12, 0.5, f, 0.055)
        elif kind == "VICTORY_HOP_LANDED":
            tone(at, 0.12, 220, 0.035, 0.2)
            tone(at + 0.05, 0.25, 1046.5, 0.035)
        elif kind == "BALL_PUSH":
            tone(at, 0.18, 240, 0.08, 0.25)
        elif kind == "PIN_DOWN":
            tone(at, 0.14, 420, 0.08, 0.6)
        elif kind == "BOWLING_STRIKE":
            for j, f in enumerate([659.25, 830.61, 987.77]):
                tone(at + j * 0.08, 0.4, f, 0.07)
    signal = np.tanh(signal) * 0.85
    pcm = np.asarray(
        np.clip(np.stack([signal, signal], axis=1), -1, 1) * 32767, dtype="<i2"
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def render(
    source, output, cut="hero", preview=False, blur=4, style="classic", goal="stunts"
):
    from microduck_lab.parkour.art_direction import industrial

    source = Path(source)
    output = Path(output)
    report = json.loads((source / "audit.json").read_text())
    arcade = report.get("difficulty") == "arcade"
    for name, sha in report["capture"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Capture checksum mismatch: " + name)
    core_passed = (
        report["passed"] if goal == "stunts" else report["component_course_passed"]
    )
    if not core_passed:
        raise ValueError(
            "This cut requires all core physical gates, including real recovery"
        )
    contacts = (
        json.loads((source / "all-contacts.json").read_text())
        if (source / "all-contacts.json").exists()
        else {}
    )
    gate_key = "publication_gate" if goal == "stunts" else "escape_contact_gate"
    if not preview:
        if contacts.get("capture") != report["capture"]:
            raise ValueError("Contact audit must match this exact capture")
        if not contacts.get(gate_key, {}).get("passed", False):
            raise ValueError(
                "Full contact audit blocks publication: " + str(contacts.get(gate_key))
            )
        proof = json.loads((source / "replay.json").read_text())
        if proof.get("capture") != report["capture"]:
            raise ValueError("Replay proof must match this exact capture")
        if not proof["passed"]:
            raise ValueError("Actuator-input replay must pass before publishing")
    if arcade and not preview:
        causal = json.loads((source / "causality.json").read_text())
        if not causal.get("passed") or causal.get("capture") != report["capture"]:
            raise ValueError("Matching independent contact-causality proof required")
    m = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    d = mujoco.MjData(m)
    states = dict(np.load(source / "trajectory.npz"))
    times = states["time"]
    sweeper_detail = None
    highrate = None
    if report.get("detail_focus"):
        sweeper_detail = json.loads((source / "sweeper-clearance.json").read_text())
        if sweeper_detail.get("capture") != report["capture"] or not sweeper_detail.get(
            "passed"
        ):
            raise ValueError("Matching full-rate sweeper proof required")
        highrate_path = source / "sweeper-highrate.npz"
        if hashlib.sha256(highrate_path.read_bytes()).hexdigest() != sweeper_detail.get(
            "highrate_sha256"
        ):
            raise ValueError("Full-rate contact-state checksum mismatch")
        highrate = dict(np.load(highrate_path))
        if (
            sweeper_detail.get("force_samples", 0) <= 0
            or sweeper_detail.get("peak_force_N", 0) <= 0
        ):
            raise ValueError("Contact detail requires a measured rod/duck force")
        if np.min(np.abs(highrate["time"] - sweeper_detail["peak_state_time"])) > 1e-8:
            raise ValueError("Force-bearing state absent from dense capture")
        report = report | {"sweeper_detail": sweeper_detail}
    timeline = CinematicEventTimeline(report, float(times[-1]))
    director = ShotDirector(timeline)
    clips = timeline.hero() if cut == "hero" else timeline.technical()
    fps = 50
    width, height = 1920, 1080
    renderer = mujoco.Renderer(m, height=height, width=width)
    option = mujoco.MjvOption()
    option.geomgroup[4] = 0
    option.geomgroup[5] = 0
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("boss/stripe"):
            m.geom_group[g] = 5
    posts = [g for g in range(m.ngeom) if m.geom(g).name.startswith("post/")]
    # Art direction is limited to non-colliding scenery and lights.
    for g in posts:
        m.geom_pos[g, 1] = np.sign(m.geom_pos[g, 1]) * 0.75
        m.geom_size[g, :2] = 0.016
    m.light_ambient[:] = [0.25, 0.28, 0.34]
    m.light_diffuse[:] = [0.75, 0.78, 0.83]
    trace_times = np.array([row["t"] for row in report["trace"]])
    events = report["events"]
    camera_position = None
    replay_stats = (
        json.loads((source / "replay.json").read_text())
        if (source / "replay.json").exists()
        else {}
    )

    def state(t, dense=False):
        i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
        if i and abs(times[i - 1] - t) < abs(times[i] - t):
            i -= 1
        d.qpos[:] = states["qpos"][i]
        d.qvel[:] = states["qvel"][i]
        d.ctrl[:] = states["ctrl"][i]
        d.eq_active[:] = states["eq_active"][i]
        m.geom_rgba[:] = states["geom_rgba"][i]
        d.time = float(times[i])
        if dense:
            k = int(np.argmin(np.abs(highrate["time"] - t)))
            for name in ("qpos", "qvel", "ctrl", "eq_active"):
                getattr(d, name)[:] = highrate[name][k]
            d.time = float(highrate["time"][k])
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("floor/"):
                m.geom_rgba[g, :3] = [0.12, 0.18, 0.26]
        if style == "industrial":
            for g in range(m.ngeom):
                name = m.geom(g).name
                if name.startswith("floor/"):
                    m.geom_rgba[g, :3] = [0.065, 0.095, 0.14]
                elif name == "boss/geom":
                    m.geom_rgba[g, :3] = [0.27, 0.06, 0.09]
                elif name == "crate/geom":
                    m.geom_rgba[g, :3] = [0.28, 0.13, 0.045]
        for g in posts:
            m.geom_group[g] = 0
        mujoco.mj_forward(m, d)

    def frame(t, clip, reset=False):
        nonlocal camera_position
        dense = clip.title.startswith("CONTACT FRAME")
        state(t, dense=dense)
        cam, kind = director.camera(t, d, 1 / fps, clip.shot, reset)
        renderer.update_scene(d, camera=cam, scene_option=option)
        camera_position = np.asarray(renderer.scene.camera[0].pos).copy()
        pictures = []
        for offset in (
            [-0.006, -0.002, 0.002, 0.006] if blur == 4 and not dense else [0.0]
        ):
            state(t + offset, dense=dense)
            # Camera occlusion management applies to decorative, non-colliding
            # posts only. Robot, floor, rails, gap and all hazards remain visible.
            subjects = [d.body("duck/trunk_base").xpos + np.array([0, 0, 0.08])]
            if kind in ("spawn", "boss", "finish"):
                subjects.append(d.body("boss").xpos.copy())
            for target in subjects:
                ray = target - camera_position
                for g in posts:
                    p = d.geom_xpos[g]
                    u = float(
                        np.dot((p - camera_position)[:2], ray[:2])
                        / max(np.dot(ray[:2], ray[:2]), 1e-9)
                    )
                    nearest = camera_position + np.clip(u, 0, 1) * ray
                    if (
                        0 < u < 1
                        and 0 < nearest[2] < 0.55
                        and np.linalg.norm((p - nearest)[:2]) < 0.12
                    ):
                        m.geom_group[g] = 5
            for g in range(m.ngeom):
                if m.geom(g).name.startswith("boss/stripe"):
                    m.geom_group[g] = 5
            renderer.update_scene(d, camera=cam, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
            camera_position = np.asarray(renderer.scene.camera[0].pos).copy()
            center = d.body("boss").xpos
            R = d.body("boss").xmat.reshape(3, 3)
            # Submillimetre cosmetic seam lines replace the z-fighting discs.
            for axis in range(3):
                points = []
                for theta in np.linspace(0, 2 * np.pi, 97):
                    local = np.zeros(3)
                    local[(axis + 1) % 3] = 0.2502 * np.cos(theta)
                    local[(axis + 2) % 3] = 0.2502 * np.sin(theta)
                    points.append(center + R @ local)
                for a, b in zip(points, points[1:]):
                    g = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.zeros(3),
                        np.eye(3).ravel(),
                        np.array(
                            [1.0, 0.12, 0.20, 1.0]
                            if style == "industrial"
                            else [0.82, 0.62, 1.0, 1.0]
                        ),
                    )
                    mujoco.mjv_connector(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        2.8 if style == "industrial" else 1.5,
                        a,
                        b,
                    )
                    g.emission = 0.2
                    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                    renderer.scene.ngeom += 1
            if style == "industrial":
                industrial(renderer.scene, m, d, report, camera_position)
            if dense:
                # Draw only the contacted enclosure, as non-colliding render lines.
                gi = m.geom(sweeper_detail["peak_force_geom"]).id
                Rg = d.geom_xmat[gi].reshape(3, 3)
                corners = [
                    d.geom_xpos[gi]
                    + Rg
                    @ (
                        m.geom_size[gi]
                        * np.array([1 if k & bit else -1 for bit in (1, 2, 4)])
                    )
                    for k in range(8)
                ]
                for k in range(8):
                    for bit in (1, 2, 4):
                        if k & bit:
                            continue
                        g = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(
                            g,
                            mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3),
                            np.zeros(3),
                            np.eye(3).ravel(),
                            np.array([0.1, 1.0, 0.8, 1.0]),
                        )
                        mujoco.mjv_connector(
                            g,
                            mujoco.mjtGeom.mjGEOM_LINE,
                            2.0,
                            corners[k],
                            corners[k | bit],
                        )
                        g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                        renderer.scene.ngeom += 1
            pictures.append(renderer.render().astype(np.float32))
        im = Image.fromarray(np.uint8(np.clip(np.mean(pictures, axis=0), 0, 255)))
        draw = ImageDraw.Draw(im, "RGBA")
        row = report["trace"][
            min(len(trace_times) - 1, int(np.searchsorted(trace_times, t)))
        ]
        prior = [e for e in events if e["t"] <= t]
        hp = max(0, 3 - sum(e["hp_cost"] for e in prior if e["type"] == "IMPACT"))
        combo = 0
        for e in prior:
            if not arcade and e["type"] in ("IMPACT", "FALL"):
                combo = 0
            elif e["type"] == ("COMBO_VERIFIED" if arcade else "STUNT_SUCCESS"):
                combo += 1
        draw.rounded_rectangle((38, 32, 554, 101), radius=14, fill=(6, 14, 25, 205))
        draw.text(
            (60, 50),
            "MICRODUCK / STRIKE & ESCAPE"
            if arcade
            else "MICRODUCK / REACTIVE CHASE"
            if report.get("predictive")
            else "MICRODUCK / NEON ESCAPE II",
            font=font(25, True),
            fill=(233, 242, 250),
        )
        draw.rounded_rectangle((1475, 32, 1882, 101), radius=14, fill=(6, 14, 25, 205))
        draw.text((1500, 52), "HP", font=font(24, True), fill="white")
        for k in range(3):
            draw.ellipse(
                (1556 + 31 * k, 57, 1573 + 31 * k, 74),
                fill=(40, 238, 177) if k < hp else (69, 76, 92),
            )
        draw.text(
            (1680, 52), f"COMBO x{combo}", font=font(24, True), fill=(64, 225, 241)
        )
        if report["brain"] != "jev":
            draw.text(
                (60, 92),
                "PHYSICAL BASELINE / RULE-BASED TACTICS",
                font=font(20, True),
                fill=(240, 204, 133),
                stroke_width=2,
                stroke_fill=(5, 10, 18),
            )
        if cut == "hero":
            if arcade:
                pins = [e for e in prior if e["type"] == "PIN_DOWN"]
                unlocked = any(e["type"] == "EXIT_UNLOCK" for e in prior)
                if row["pos"][0] > 4.35 and t < timeline.finish:
                    draw.rounded_rectangle(
                        (660, 32, 1340, 110), radius=14, fill=(6, 14, 25, 220)
                    )
                    draw.text(
                        (690, 49),
                        "STRIKE TO UNLOCK"
                        if not unlocked
                        else "STRIKE!  EXIT UNLOCKED",
                        font=font(27, True),
                        fill=(255, 216, 95) if not unlocked else (64, 239, 177),
                    )
                    draw.text(
                        (690, 84),
                        f"TARGETS {len(pins)}/3  |  PUSH → WATCH → ROLL",
                        font=font(17),
                        fill="white",
                    )
            recent = [
                e for e in prior if e["type"] == "JEV_DECISION" and t - e["t"] < 0.65
            ]
            if recent:
                action = recent[-1]["action"].replace("_", " ")
                draw.text(
                    (60, 125),
                    ("JEV  /  " if report["brain"] == "jev" else "RULE CONTROLLER  /  ")
                    + action,
                    font=font(26, True),
                    fill=(57, 235, 222),
                    stroke_width=2,
                    stroke_fill=(5, 10, 18),
                )
            route_events = [
                e for e in prior if e["type"] == "ROUTE_PREVIEW" and t - e["t"] < 0.9
            ]
            if route_events:
                side = "LEFT" if route_events[-1]["target_y"] > 0 else "RIGHT"
                draw.text(
                    (60, 165),
                    "LOCAL PLAN / PASS " + side,
                    font=font(25, True),
                    fill=(65, 230, 210),
                    stroke_width=2,
                    stroke_fill=(5, 10, 18),
                )
            success = [
                e
                for e in prior
                if e["type"] == ("COMBO_VERIFIED" if arcade else "STUNT_SUCCESS")
                and t - e["t"] < 0.7
            ]
            if success:
                e = success[-1]
                label = (
                    e["challenge"]
                    if arcade
                    else {
                        "ROLL_CENTER": "PERFECT ROLL",
                        "JUMP_CENTER": "LONG JUMP",
                        "TAKE_LEFT_ROUTE": "DODGE",
                        "TAKE_RIGHT_ROUTE": "ALIGNED",
                    }[e["skill"]]
                )
                draw.text(
                    (62, 930),
                    f"{label}  /  COMBO x{e['combo']}"
                    if arcade
                    else f"{label}  +{e['bonus']}",
                    font=font(35, True),
                    fill=(56, 238, 199),
                    stroke_width=2,
                    stroke_fill=(5, 10, 18),
                )
            if timeline.fall <= t < timeline.recover:
                draw.text(
                    (62, 930),
                    "GET UP.",
                    font=font(40, True),
                    fill=(255, 212, 92),
                    stroke_width=2,
                    stroke_fill=(5, 10, 18),
                )
            elif timeline.recover <= t < timeline.recover + 0.65:
                draw.text(
                    (62, 930),
                    "BACK IN THE CHASE.",
                    font=font(34, True),
                    fill=(65, 240, 183),
                    stroke_width=2,
                    stroke_fill=(5, 10, 18),
                )
            if t < 0.7:
                draw.text(
                    (785, 440),
                    "RUN.",
                    font=font(110, True),
                    fill="white",
                    stroke_width=2,
                    stroke_fill=(5, 10, 18),
                )
            if clip.title:
                draw.rounded_rectangle(
                    (555, 365, 1370, 590), radius=24, fill=(5, 14, 26, 220)
                )
                draw.text(
                    (652, 395), clip.title, font=font(82, True), fill=(66, 242, 191)
                )
                draw.text(
                    (688, 514),
                    "ROSCLAW / PHYSICAL AI",
                    font=font(28, True),
                    fill="white",
                )
        else:
            draw.rounded_rectangle(
                (35, 818, 1885, 1048), radius=18, fill=(5, 14, 26, 235)
            )
            draw.text((62, 837), clip.title, font=font(31, True), fill=(64, 235, 213))
            draw.text(
                (62, 891),
                f"SEED {report['seed']}    SIM {t:05.2f}s    {clip.speed:g}x replay    {row['stage']}",
                font=font(26),
                fill="white",
            )
            draw.text(
                (62, 935),
                f"Physics: 0.2 ms steps  |  Learned motor policies: 50 Hz  |  Capture: 200 Hz",
                font=font(25),
                fill=(189, 212, 234),
            )
            decisions = [e for e in prior if e["type"] == "JEV_DECISION"]
            last = decisions[-1] if decisions else None
            accepted = [
                x
                for x in report["decisions"]
                if x.get("result")
                and x.get("encounter") == (last or {}).get("encounter")
            ]
            if accepted:
                dec = accepted[-1]["result"]
                probs = "  ".join(
                    f"{k.replace('TAKE_', '').replace('_ROUTE', '')}: {v:.2f}"
                    for k, v in dec["probabilities"].items()
                )
                draw.text(
                    (62, 984),
                    f"Jev {dec['latency_ms']:.0f} ms (wall clock)  |  {probs}",
                    font=font(21),
                    fill=(207, 222, 235),
                )
            if "LOCAL PHYSICS PREVIEW" in clip.title:
                predictions = [
                    p for p in report.get("forecasts", []) if p["kind"] == "sweeper"
                ]
                if predictions:
                    ms = sum(c["wall_ms"] for c in predictions[-1]["candidates"])
                    draw.rectangle((50, 975, 1870, 1035), fill=(5, 14, 26, 255))
                    draw.text(
                        (62, 984),
                        f"Privileged simulator state + local physics model / {ms:.0f} ms planning / not camera perception",
                        font=font(22),
                        fill=(207, 222, 235),
                    )
            if report.get("detail_focus"):
                if dense:
                    detail = f"Enclosing collision box (cyan) | peak normal force {sweeper_detail['peak_force_N']:.1f} N | impulse {sweeper_detail['normal_impulse_Ns']:.3f} N s"
                elif clip.shot == "sweeper_top":
                    detail = "Capsule rod / enclosing body colliders | contact enabled | full-rate force audit verifies a brief brush"
                elif clip.shot == "landing":
                    detail = f"Airborne phase {timeline.flight['end'] - timeline.flight['start']:.3f} s | real gap geometry | feet re-establish support"
                elif clip.shot == "bowling":
                    detail = "Force transfers from duck to ball to pins | measured pin falls trigger the gate motor"
                else:
                    detail = "Jev selects skills; learned policies drive position servos | simulator-state planning, not camera perception"
                draw.rectangle((50, 975, 1870, 1035), fill=(5, 14, 26, 255))
                draw.text((62, 984), detail, font=font(22), fill=(207, 222, 235))
                if dense:
                    draw.rectangle((50, 924, 1870, 973), fill=(5, 14, 26, 255))
                    draw.text(
                        (62, 935),
                        "Exact 5 kHz contact state | 0.2 mm soft-contact margin | no geometric rod/body penetration in this encounter",
                        font=font(22),
                        fill=(189, 212, 234),
                    )
            if clip.title == "REPRODUCIBLE INPUT REPLAY":
                draw.rounded_rectangle(
                    (280, 220, 1640, 690), radius=24, fill=(5, 14, 26, 235)
                )
                lines = [
                    ("ACTUATOR-INPUT REPLAY", 42),
                    (
                        f"{replay_stats.get('frames', 0):,} recorded states reproduced; max position error {replay_stats.get('max_qpos_error', 0):.1e}",
                        28,
                    ),
                    (
                        f"Duck/hazard penetration: max {report['audit']['max_penetration_m'] * 1000:.3f} mm; P99 {report['audit']['p99_penetration_m'] * 1000:.3f} mm",
                        28,
                    ),
                    (
                        f"Ground/self max overlap: {contacts['categories']['duck_floor']['max_penetration_m'] * 1000:.3f} / {contacts['categories'].get('duck_self', {}).get('max_penetration_m', 0) * 1000:.3f} mm",
                        28,
                    ),
                    (
                        "Real void. Finite-force props. "
                        + (
                            "Motor-policy recovery."
                            if goal == "stunts"
                            else "Reactive escape; no staged knockdown."
                        ),
                        26,
                    ),
                    (
                        "Selected development run; see README for unseen-seed outcomes.",
                        25,
                    ),
                    ("Foley is synthesized in post-production.", 25),
                ]
                if arcade:
                    plan_ms = sum(
                        c["wall_ms"]
                        for f in report["forecasts"]
                        for c in f["candidates"]
                    )
                    lines = [
                        ("ACTUATOR-INPUT REPLAY", 42),
                        (
                            f"{replay_stats.get('frames', 0):,} states reproduced; position error {replay_stats.get('max_qpos_error', 0):.1e}",
                            28,
                        ),
                        (
                            "Six verified encounters. Real contact chain unlocks the motorized gate.",
                            26,
                        ),
                        (
                            f"Privileged-state previews: {plan_ms / 1000:.1f}s wall clock; simulation-time replay.",
                            26,
                        ),
                        (
                            "Historical Jev prefix + new physical finish; see README for outcomes."
                            if report.get("continuation")
                            else "Selected frozen-cohort run; failures and latency are reported in the README.",
                            25,
                        ),
                        ("Synthesized Foley. This is a simulator demonstration.", 25),
                    ]
                yy = 252
                for text, size in lines:
                    draw.text(
                        (325, yy),
                        text,
                        font=font(size, size == 42),
                        fill=(55, 239, 210) if size == 42 else "white",
                    )
                    yy += 55
        if clip.speed != 1 and not clip.hold:
            draw.text(
                (1620, 1020),
                f"{clip.speed:g}x SLOW-MO",
                font=font(23, True),
                fill=(235, 241, 250),
                stroke_width=2,
                stroke_fill=(5, 10, 18),
            )
        return np.asarray(im)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if preview:
            picks = [
                (0.35, "spawn"),
                (3.0, "chase"),
                (sum(timeline.skills["ROLL_CENTER"]) / 2, "roll"),
                (timeline.flight["start"] + 0.08, "jump"),
                (timeline.fall + 0.4, "impact")
                if np.isfinite(timeline.fall)
                else (timeline.route + 1.2, "dodge"),
                (timeline.door + 0.2, "finish"),
            ]
            if arcade:
                push = next(e["t"] for e in events if e["type"] == "BALL_PUSH")
                strike = next(e["t"] for e in events if e["type"] == "BOWLING_STRIKE")
                picks = [
                    (sum(timeline.instances["ROLL_CENTER"][0]) / 2, "roll"),
                    (timeline.flight["start"] + 0.08, "jump"),
                    (push + 0.12, "bowling"),
                    (strike + 0.08, "bowling"),
                    (sum(timeline.instances["ROLL_CENTER"][-1]) / 2, "roll"),
                    (timeline.door + 0.2, "finish"),
                ]
            if report.get("detail_focus"):
                hit = sweeper_detail["peak_state_time"]
                picks = [
                    (timeline.flight["start"] + 0.06, "landing"),
                    (hit - 0.10, "sweeper_top"),
                    (hit, "sweeper_side"),
                    (strike + 0.06, "bowling"),
                    (sum(timeline.instances["ROLL_CENTER"][-1]) / 2, "roll"),
                    (timeline.stop - 0.5, "victory"),
                ]
            sheet = Image.new("RGB", (1920, 1080))
            for j, (t, shot) in enumerate(picks):
                im = Image.fromarray(
                    frame(
                        t,
                        clips[0].__class__(
                            t,
                            t,
                            shot=shot,
                            title="CONTACT FRAME / exact 0.2 ms physics state"
                            if report.get("detail_focus") and shot == "sweeper_side"
                            else "",
                        ),
                        True,
                    )
                )
                im.resize((640, 540)).save(output.parent / f"parkour-preview-{j}.jpg")
                sheet.paste(im.resize((640, 360)), ((j % 3) * 640, (j // 3) * 360))
            sheet.crop((0, 0, 1920, 720)).save(output)
            return
        rendered = []
        sound_events = []
        offset = 0.0
        frames = 0
        with tempfile.TemporaryDirectory(prefix="microduck-film-") as tmp:
            silent = Path(tmp) / "silent.mp4"
            audio = Path(tmp) / "foley.wav"
            with imageio.get_writer(
                str(silent),
                fps=fps,
                codec="libx264",
                macro_block_size=1,
                ffmpeg_params=["-crf", "18", "-preset", "medium", "-threads", "4"],
            ) as writer:
                for number, clip in enumerate(clips):
                    count = math.ceil(clip.duration * fps)
                    rendered.append(
                        dict(
                            start=clip.start,
                            stop=clip.stop,
                            speed=clip.speed,
                            shot=clip.shot,
                            title=clip.title,
                            output_start=offset,
                            frames=count,
                        )
                    )
                    if not clip.hold:
                        for e in events:
                            if clip.start <= e["t"] < clip.stop:
                                sound_events.append(
                                    (
                                        offset + (e["t"] - clip.start) / clip.speed,
                                        e["skill"]
                                        if e["type"] == "SKILL_START"
                                        else e["type"],
                                    )
                                )
                    for j in range(count):
                        t = (
                            clip.start
                            if clip.hold
                            else min(clip.stop, clip.start + j / fps * clip.speed)
                        )
                        writer.append_data(
                            frame(
                                t,
                                clip,
                                reset=(j == 0 and (number == 0 or cut == "technical")),
                            )
                        )
                        frames += 1
                    offset += count / fps
                    print(cut, number + 1, len(clips), round(offset, 2), flush=True)
            sound(audio, frames / fps, sound_events)
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(silent),
                    "-i",
                    str(audio),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
            )
        Image.fromarray(
            frame(
                timeline.flight["start"] + 0.06,
                timeline.hero()[0].__class__(0, 0, shot="jump"),
                True,
            )
        ).save(output.with_suffix(".jpg"), quality=94)
        manifest = dict(
            source=str(source),
            source_audit_sha256=hashlib.sha256(
                (source / "audit.json").read_bytes()
            ).hexdigest(),
            video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            cut=cut,
            style=style,
            fps=fps,
            frames=frames,
            duration_s=frames / fps,
            resolution=[width, height],
            brain=report["brain"],
            evaluation_goal=goal,
            core_physical_gates=core_passed,
            full_stunt_gate=report["passed"],
            contact_publication_gate=contacts.get(gate_key),
            full_contact_audit_sha256=hashlib.sha256(
                (source / "all-contacts.json").read_bytes()
            ).hexdigest(),
            input_replay_sha256=hashlib.sha256(
                (source / "replay.json").read_bytes()
            ).hexdigest(),
            contact_causality_sha256=hashlib.sha256(
                (source / "causality.json").read_bytes()
            ).hexdigest()
            if arcade
            else None,
            sweeper_detail_sha256=hashlib.sha256(
                (source / "sweeper-clearance.json").read_bytes()
            ).hexdigest()
            if sweeper_detail
            else None,
            sweeper_highrate_sha256=sweeper_detail.get("highrate_sha256")
            if sweeper_detail
            else None,
            contact_frame="Nearest actual 5 kHz state, collision enclosure outlined in render-only cyan lines"
            if sweeper_detail
            else None,
            continuation=report.get("continuation"),
            victory=report.get("victory"),
            qualification_scope="Selected simulation for the declared evaluation_goal; not a 20 cm / 1.2 s dodge certificate or unseen success-rate claim. Physics preview uses privileged simulator state, not vision.",
            clips=rendered,
            motion_blur="4 nearest actual 200 Hz states at -6/-2/+2/+6 ms; no joint interpolation"
            if blur == 4
            else "off",
            audio="Procedural post-produced Foley, not a simulated microphone",
            decor="Non-colliding posts styled at +/-0.75 m with 16 mm half-width; foreground occlusion management; studio fill lighting. Collidable geometry and native robot materials retained.",
        )
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(output, frames / fps, flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cut", choices=["hero", "technical"], default="hero")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--blur", type=int, choices=[1, 4], default=4)
    p.add_argument("--style", choices=["classic", "industrial"], default="classic")
    p.add_argument("--goal", choices=["stunts", "escape"], default="stunts")
    render(**vars(p.parse_args()))
