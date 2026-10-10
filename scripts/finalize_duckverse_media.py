#!/usr/bin/env python3
"""Add original synthesized sound design and frame-map-aligned bilingual SRT."""

import argparse
import gzip
import json
import subprocess
import wave
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from microduck_lab.arena.episode import sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out-dir", default="out")
a = p.parse_args()
root = Path(a.out_dir)


def stamp(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def captions(record):
    if record.get("kind") == "end_card":
        return (
            "ROSClaw x Microduck / reproducible simulation",
            "ROSClaw × Microduck：可复现的物理仿真",
        )
    if record.get("kind") == "method_card":
        return (record["title"], "方法与证据：仿真结果，不代表真机验证")
    if "caption" in record:
        en = record["caption"]
        if en.startswith("MATCH"):
            zh = f"独立比赛 / seed {record['seed']} / 连续原始视角"
        elif "POV" in en:
            zh = "视角回放：镜头跟随真实记录的鸭头位姿"
        elif "GEOMETRY" in en:
            zh = "碰撞几何：原生接触模型与保守的头部、躯干包围盒"
        elif "duck-duck" in en:
            zh = "慢动作细节：真实鸭与鸭接触"
        elif "released support" in en:
            zh = "慢动作细节：地板释放约束，失去支撑后真实跌落"
        else:
            zh = "细节回放：根据当前预警，由学习步态驱动关节运动"
        return en, zh
    t = record["sim_time"]
    shot = record["shot"]
    if shot == "contact":
        return (
            "Real duck-duck contacts / recorded-state slow motion",
            "真实鸭与鸭接触：根据记录状态播放慢动作",
        )
    if shot == "fall":
        return (
            "Released floor > gravity > physical elimination",
            "地板释放约束 → 重力下坠 → 真实淘汰",
        )
    if t > 32.82:
        return (
            "Winner requires upright posture and loaded stable support",
            "冠军需要保持直立，并有真实承重的稳定支撑",
        )
    if t < 3:
        return (
            "Four independent robots share one physical world",
            "四只独立控制的机器人，共处同一个物理世界",
        )
    if t < 8:
        return (
            "A visible warning triggers a walk to a legal neighbouring tile",
            "可见预警触发移动：走向合法的相邻地板",
        )
    if t < 16:
        return (
            "The controller receives current state, never the future schedule",
            "控制器只读取当前状态，不读取未来塌陷安排",
        )
    if t < 23:
        return (
            "Existing learned walking policies / 50 Hz torque-limited servos",
            "现有学习步态：50 Hz 更新，关节力矩受限",
        )
    return (
        "A shrinking arena leaves fewer safe choices",
        "竞技场不断缩小，安全选择越来越少",
    )


for name in [
    "duckverse_game_en",
    "duckverse_game_short_en",
    "duckverse_game_reference",
    "duckverse_game_technical_en",
]:
    movie = root / (name + ".mp4")
    manifest_path = movie.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    with gzip.open(root / manifest["frame_map"], "rt") as fp:
        records = [json.loads(line) for line in fp]
    groups = []
    for i, r in enumerate(records):
        pair = captions(r)
        if not groups or groups[-1][2] != pair or i - groups[-1][0] >= 250:
            groups.append([i, i + 1, pair])
        else:
            groups[-1][1] = i + 1
    for lang, col in [("en", 0), ("zh-CN", 1)]:
        target = root / (name + "." + lang + ".srt")
        target.write_text(
            "\n\n".join(
                f"{j + 1}\n{stamp(s / 50)} --> {stamp(e / 50)}\n{pair[col]}"
                for j, (s, e, pair) in enumerate(groups)
            )
            + "\n"
        )
    if name in ["duckverse_game_en", "duckverse_game_short_en"]:
        rate = 48000
        duration = len(records) / 50
        t = np.arange(round(duration * rate)) / rate
        pad = np.zeros_like(t)
        # Original four-note suspended harmony; restrained low percussion.
        for k, freq in enumerate([146.832, 195.998, 220.0, 293.665]):
            pad += (
                0.009
                * np.sin(2 * np.pi * freq * t + 0.4 * k)
                * (0.7 + 0.3 * np.sin(2 * np.pi * 0.09 * t + k))
            )
        for beat in np.arange(0, duration, 0.5):
            start = round(beat * rate)
            end = min(len(t), start + int(0.18 * rate))
            u = np.arange(end - start) / rate
            pad[start:end] += (
                0.07 * np.sin(2 * np.pi * (65 * u - 95 * u * u)) * np.exp(-u * 23)
            )
        fade = np.minimum(1, t / 0.8) * np.minimum(1, (duration - t) / 1.2)
        signal = np.clip(pad * fade, -0.8, 0.8)
        stereo = np.column_stack([signal, signal * 0.98])
        wav = movie.with_suffix(".soundtrack.wav")
        with wave.open(str(wav), "wb") as fp:
            fp.setnchannels(2)
            fp.setsampwidth(2)
            fp.setframerate(rate)
            fp.writeframes((stereo * 32767).astype("<i2").tobytes())
        temporary = movie.with_name(movie.stem + ".mux.mp4")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(movie),
                "-i",
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(movie)
        wav.unlink()
        manifest["audio"] = (
            "Original synthesized harmony/percussion sound design; not recorded physical audio"
        )
    manifest["video_sha256"] = sha(movie)
    manifest["subtitles"] = {
        lang: {
            "file": name + "." + lang + ".srt",
            "sha256": sha(root / (name + "." + lang + ".srt")),
        }
        for lang in ["en", "zh-CN"]
    }
    manifest["presentation_source_sha256"] = {
        path: sha(path)
        for path in [
            "src/microduck_lab/presentation/duckverse.py",
            "scripts/render_duckverse_technical.py",
            "scripts/finalize_duckverse_media.py",
        ]
    }
    manifest["editing"] += "; explicit end/method cards carry no physical-state claim"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(name, manifest["duration_s"], manifest["video_sha256"])
