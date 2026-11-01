"""Make silent, actual-state V2 review samples; never runs a fight."""

import argparse
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json
from combat_director import shot


SOURCES = {
    "v1": ("v2-film-v1-91003-1", "V1 RULE BASELINE / CONTINUOUS 1x", 15.0, "pair"),
    "full_brake": (
        "v2-film-full_brake-91003-1",
        "V2 FULL-BRAKE ABLATION / CONTINUOUS 1x",
        15.0,
        "pair",
    ),
    "dynamic": (
        "v2-film-dynamic-91003-1",
        "V2 DYNAMIC HANDOFF / CONTINUOUS 1x",
        15.0,
        "pair",
    ),
    "probe": (
        "v2-film-probe-94001-0",
        "STANDING-TARGET COURSE / SHORT PROBE",
        8.0,
        "pair",
    ),
    "drive": (
        "v2-film-drive-94001-0",
        "STANDING-TARGET COURSE / COMMITTED DRIVE",
        8.0,
        "pair",
    ),
    "team_course": (
        "v2-film-team_course-93004-1",
        "PRESSURE COURSE / PLANNED TARGET CHANGES",
        20.0,
        "wide",
    ),
    "team_free": (
        "v2-film-team_free-92001-0",
        "FREE 2v2 / RULE TEAM INTENTS / CONTINUOUS 1x",
        20.0,
        "wide",
    ),
}


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, (folder, caption, length, view) in SOURCES.items():
        path = out / (name + ".plan.json")
        if path.exists():
            raise FileExistsError(path)
        source = root / folder
        m = json.loads((source / "manifest.json").read_text())
        if (
            m["result"]["kind"] == "NO_CONTEST_PHYSICS"
            or m["physical"]["penetration_max_m"] > 0.003
            or m["physical"]["penetrating_p99_m"] > 0.0015
        ):
            raise ValueError(
                "Rejected capture cannot be presented as a qualified sample"
            )
        s = shot(
            source,
            0,
            length,
            1.0,
            view,
            caption,
            "course" if "COURSE" in caption else "match",
        )
        s.update(frames=round((s["end_s"] - s["start_s"]) * 50), output_start_s=0.0)
        plan = dict(
            schema="duckfu.film-plan.v1",
            edition="reference",
            fps=50,
            shots=[s],
            duration_s=s["frames"] / 50,
            continuous=True,
            simulation_only=True,
            source_run=str(source.resolve()),
            source_seed=m["seed"],
            source_result=m["result"],
            slow_motion_uses_actual_states=True,
            film_gate="PENDING_HUMAN_REVIEW",
            audio="silent",
        )
        write_json(path, plan)
    s = shot(
        root / SOURCES["dynamic"][0],
        4.4,
        6.2,
        0.25,
        "pair",
        "0.25x REPLAY / EVASION FOLLOWED BY ACTUAL CONTACT RESPONSE",
        "replay",
    )
    s.update(frames=round((s["end_s"] - s["start_s"]) / 0.25 * 50), output_start_s=0.0)
    plan = dict(
        schema="duckfu.film-plan.v1",
        edition="reference",
        fps=50,
        shots=[s],
        duration_s=s["frames"] / 50,
        continuous=False,
        simulation_only=True,
        slow_motion_uses_actual_states=True,
        film_gate="PENDING_HUMAN_REVIEW",
        audio="silent",
    )
    write_json(out / "slow_detail.plan.json", plan)


def compose(out):
    for name, left, right, title in [
        ("duel_comparison", "v1", "dynamic", "V1 vs V2 / MATCHED INITIAL STATE"),
        (
            "transition_ablation",
            "full_brake",
            "dynamic",
            "FULL BRAKE vs DYNAMIC HANDOFF / MATCHED INITIAL STATE",
        ),
        (
            "attack_comparison",
            "probe",
            "drive",
            "SHORT PROBE vs COMMITTED DRIVE / STANDING-TARGET COURSE",
        ),
    ]:
        dest = out / f"duck_fu_v2_{name}.mp4"
        if dest.exists():
            raise FileExistsError(dest)
        inputs = [out / (panel + ".mp4") for panel in (left, right)]
        for panel in inputs:
            if not panel.exists():
                raise FileNotFoundError(panel)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        banner = Image.new("RGBA", (1920, 1080), (12, 17, 25, 255))
        draw = ImageDraw.Draw(banner)
        draw.rectangle((0, 180, 1919, 899), fill=(0, 0, 0, 0))
        draw.text(
            (960, 70),
            title,
            font=ImageFont.truetype(font, 34),
            fill="white",
            anchor="mm",
        )
        draw.text(
            (960, 127),
            "Two independent runs / same body and seed / continuous 1x / silent",
            font=ImageFont.truetype(font, 24),
            fill=(170, 184, 202),
            anchor="mm",
        )
        banner_path = out / (name + "-banner.png")
        banner.save(banner_path)
        graph = "[0:v][1:v]hstack=inputs=2,pad=1920:1080:0:180:color=0x0c1119[base];[base][2:v]overlay=0:0:shortest=1[v]"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(inputs[0]),
            "-i",
            str(inputs[1]),
            "-loop",
            "1",
            "-i",
            str(banner_path),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(dest),
        ]
        subprocess.run(cmd, check=True)
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(dest))
        decoder = imageio_ffmpeg.read_frames(str(dest))
        stream = next(decoder)
        decoder.close()
        if abs(stream["fps"] - 50) > 0.001 or stream["size"] != (1920, 1080):
            raise ValueError("Composite dimensions/rate differ")
        panels = []
        for panel in inputs:
            meta = json.loads(panel.with_suffix(".json").read_text())
            if meta["frames"] != frames:
                raise ValueError("Composite/frame provenance mismatch")
            panels.append(
                dict(
                    video=str(panel.resolve()),
                    sha256=digest(panel),
                    frames=frames,
                    source_frame_mapping=str(
                        panel.with_suffix(".frames.jsonl").resolve()
                    ),
                    source_frame_mapping_sha256=digest(
                        panel.with_suffix(".frames.jsonl")
                    ),
                )
            )
        if stream.get("audio_codec"):
            raise ValueError("Review clips must be silent")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                str(dest),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        write_json(
            dest.with_suffix(".panels.json"),
            dict(
                schema="duckfu.comparison-film.v1",
                video_sha256=digest(dest),
                frames=frames,
                fps=50,
                layout="horizontal/padded",
                panels=panels,
                command=cmd,
                continuous_each_panel=True,
                synthetic_interpolation=False,
                silent=True,
                human_review="PENDING_HUMAN_REVIEW",
            ),
        )
        print(dest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["prepare", "render", "compose"], required=True)
    p.add_argument("--root", default="artifacts/combat/local")
    p.add_argument("--out", default="out/duckfu-v2-samples")
    p.add_argument("--panel", choices=[*SOURCES, "slow_detail"])
    a = p.parse_args()
    out = Path(a.out)
    if a.mode == "prepare":
        prepare(Path(a.root), out)
    elif a.mode == "compose":
        compose(out)
    else:
        if not a.panel:
            p.error("--render requires --panel")
        from render_combat_film import render

        render(
            out / (a.panel + ".plan.json"),
            out / (a.panel + ".mp4"),
            1280 if a.panel in ("team_course", "team_free", "slow_detail") else 960,
            720,
        )


if __name__ == "__main__":
    main()
