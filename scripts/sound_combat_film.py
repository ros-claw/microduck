"""Post-produced English commentary/foley, grounded in a physical film plan.

No audio is presented as a simulated microphone. Contact taps use recorded
positive-force episodes; the continuous reference remains silent.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import imageio_ffmpeg
import numpy as np

from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json


def speech(job):
    tts, cache, text, maximum = job
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for rate in ("+12%", "+25%", "+38%"):
        key = hashlib.sha256(("en-GB-RyanNeural" + rate + text).encode()).hexdigest()
        path = cache / (key + ".mp3")
        if not path.exists():
            success = False
            for _ in range(3):
                result = subprocess.run(
                    [
                        tts,
                        "--voice",
                        "en-GB-RyanNeural",
                        "--rate",
                        rate,
                        "--pitch",
                        "+3Hz",
                        "--text",
                        text,
                        "--write-media",
                        str(path),
                    ],
                    capture_output=True,
                    timeout=55,
                )
                if (
                    result.returncode == 0
                    and path.exists()
                    and path.stat().st_size > 1000
                ):
                    success = True
                    break
            if not success:
                raise RuntimeError("Speech synthesis failed for an English cue")
        raw = subprocess.check_output(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-",
            ]
        )
        pcm = np.frombuffer(raw, dtype="<f4").copy()
        active = np.flatnonzero(np.abs(pcm) > 0.003)
        if not len(active):
            raise ValueError("Silent TTS cue")
        pcm = pcm[max(0, active[0] - 2400) : min(len(pcm), active[-1] + 4800)]
        if len(pcm) / 48000 <= maximum:
            return pcm, dict(
                text=text,
                voice="en-GB-RyanNeural",
                rate=rate,
                voice_asset_sha256=digest(path),
                duration_s=len(pcm) / 48000,
            )
    raise ValueError("Spoken cue cannot fit without overlapping the next cue")


def stamp(t):
    ms = round(t * 1000)
    return (
        f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--silent-video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tts", required=True)
    a = p.parse_args()
    plan_path, source, out = Path(a.plan), Path(a.silent_video), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text())
    if plan["continuous"]:
        raise ValueError("Keep the continuous physical reference silent")
    cache = out.parent / "duckfu-voice-cache"
    cache.mkdir(parents=True, exist_ok=True)
    run = Path(plan["source_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    events = [
        json.loads(line)
        for line in (run / "runtime_events.jsonl").read_text().splitlines()
    ]
    duration = plan["duration_s"]
    rate = 48000
    audio = np.zeros(round(duration * rate), dtype=np.float32)
    rng = np.random.default_rng(1030)

    def add(at, pcm, gain=1.0):
        start = max(0, round(at * rate))
        end = min(len(audio), start + len(pcm))
        if end > start:
            audio[start:end] += gain * pcm[: end - start]

    # Original synthesized rhythm; not a licensed music sample or sensor sound.
    t = np.arange(len(audio)) / rate
    audio += (
        0.012 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    )
    for at in np.arange(0, duration, 0.5):
        tick = np.arange(round(0.07 * rate)) / rate
        add(at, 0.035 * np.exp(-tick * 55) * np.sin(2 * np.pi * 90 * tick))
    foley = []
    contact_sources = {}
    for shot in plan["shots"]:
        if shot["speed"] <= 0:
            continue
        shot_run = Path(shot["run"])
        if str(shot_run) not in contact_sources:
            contact_sources[str(shot_run)] = (
                json.loads((shot_run / "manifest.json").read_text()),
                json.loads((shot_run / "contacts_summary.json").read_text())[
                    "episodes"
                ],
            )
        shot_manifest, shot_episodes = contact_sources[str(shot_run)]
        for e in shot_episodes:
            time = e["start_step"] * shot_manifest["physics"]["dt"]
            if (
                e["summed_normal_impulse_Ns"] < 0.015
                or not shot["start_s"] <= time < shot["end_s"]
            ):
                continue
            at = shot["output_start_s"] + (time - shot["start_s"]) / shot["speed"]
            if foley and at - foley[-1]["output_time_s"] < 0.12:
                continue
            u = np.arange(round(0.11 * rate)) / rate
            tap = np.exp(-u * 40) * (
                np.sin(2 * np.pi * 180 * u) + rng.normal(size=len(u)) * 0.10
            )
            amplitude = min(0.17, 0.05 + e["summed_normal_impulse_Ns"] * 0.15)
            add(at, tap, amplitude)
            foley.append(
                dict(
                    output_time_s=at,
                    source_contact_episode=e["event_id"],
                    source_time_s=time,
                    impulse_Ns=e["summed_normal_impulse_Ns"],
                    source_run=str(shot_run),
                    source_contact_summary_sha256=digest(
                        shot_run / "contacts_summary.json"
                    ),
                )
            )
    editorial_cues = []
    if plan["edition"] == "finisher":
        final_shot = next(s for s in plan["shots"] if s["kind"] == "result")
        at = final_shot["output_start_s"] + 0.08
        for offset, length in ((0.0, 0.10), (0.20, 0.18)):
            u = np.arange(round(length * rate)) / rate
            whistle = np.sin(np.pi * u / length) ** 2 * (
                np.sin(2 * np.pi * 2450 * u) + 0.3 * np.sin(2 * np.pi * 2850 * u)
            )
            add(at + offset, whistle, 0.045)
        editorial_cues.append(
            dict(
                kind="synthesized_final_whistle",
                output_time_s=at,
                actual_result=manifest["result"],
                source_manifest_sha256=digest(run / "manifest.json"),
            )
        )
    winner = manifest["result"]["winner"]
    final_en = f"{winner.title()} team wins!" if winner else "The referee calls a draw!"
    final_cn = (
        ("红队获胜！" if winner == "RED" else "蓝队获胜！")
        if winner
        else "裁判宣布平局！"
    )
    cues = []
    if plan.get("commentary"):
        cues = [
            (c["at_s"], c["text"], c.get("zh_CN", c["text"]))
            for c in plan["commentary"]
        ]
    elif plan["edition"] == "hero":
        cues.append(
            (
                0.15,
                "Four Microducks. Two teams. One physics arena!",
                "四只 Microduck，两支队伍，同一个物理擂台！",
            )
        )
        hit = next(e for e in events if e["state"] == "VALID_STRIKE")
        for shot in plan["shots"]:
            if (
                shot["kind"] == "match"
                and shot["speed"] == 1
                and shot["start_s"] <= hit["time_s"] < shot["end_s"]
            ):
                at = shot["output_start_s"] + hit["time_s"] - shot["start_s"]
                actor = manifest["spawns"][hit["actor"]]["display_name"].title()
                chinese_actor = manifest["spawns"][hit["actor"]].get(
                    "chinese_name", actor
                )
                cues.append(
                    (
                        max(6.3, at),
                        f"{actor}! That bump lands!",
                        f"{chinese_actor}，这一下撞到了！",
                    )
                )
                break
        # One additional call follows an actual response in the normal-speed
        # segment. It is not a scripted hit or a claim of a counter sequence.
        extra = None
        for event in events:
            if event["state"] != "VALID_STRIKE" or event["actor"] == hit["actor"]:
                continue
            for shot in plan["shots"]:
                if shot["kind"] != "match" or shot["speed"] != 1:
                    continue
                if shot["start_s"] <= event["time_s"] < shot["end_s"]:
                    at = shot["output_start_s"] + event["time_s"] - shot["start_s"]
                    if 10 <= at <= 11.5:
                        person = manifest["spawns"][event["actor"]]
                        extra = (
                            at,
                            person["display_name"].title() + " lands a bump!",
                            person.get("chinese_name", person["display_name"])
                            + "也撞中了！",
                        )
                        break
            if extra:
                cues.append(extra)
                break
        cues.append(
            (
                14.0,
                "Tactics pick the move. Learned motors control all fourteen joints.",
                "战术决定动作，学习得到的运动策略控制十四个关节。",
            )
        )
        replay = [
            s for s in plan["shots"] if s["kind"] == "replay" and s["speed"] == 0.25
        ]
        texts = [
            ("Now watch the contact in slow motion.", "慢下来，观察真正发生的接触。"),
            (
                "The robots have to regain stable foot support.",
                "机器人必须重新获得稳定的足部支撑。",
            ),
        ]
        for s, (en, cn) in zip(replay, texts):
            cues.append((s["output_start_s"] + 0.1, en, cn))
        back = plan["shots"][-2]
        cues.append(
            (
                back["output_start_s"] + 0.2,
                "Back to full speed. This fight is being decided by physics!",
                "回到正常速度，胜负由物理过程决定！",
            )
        )
    elif plan["edition"] == "short":
        cues = [
            (
                0.1,
                "Four robots. Two teams. Real physical contact!",
                "四只机器人，两支队伍，真实物理接触！",
            ),
            (
                5.2,
                "Slow motion. Watch the balance and the contact.",
                "慢动作，观察平衡和接触细节。",
            ),
        ]
    else:
        lines = [
            (
                "One shared world. Each robot has its own motor history.",
                "同一个共享世界，每只机器人都有独立的运动历史。",
            ),
            (
                "Maneuvers commit to their motion. The opponent sees public state, not a hidden attack phase.",
                "动作按阶段执行，对手只能观察公开状态，不能读取隐藏的攻击阶段。",
            ),
            (
                "These are the actual component convex collision shapes.",
                "这里显示的是实际使用的零件凸碰撞几何。",
            ),
            (
                "Contact impulses are measured by the solver. There is no added root force.",
                "接触冲量来自求解器测量，没有添加根节点外力。",
            ),
            (
                "Tactics choose a skill. Learned motor policies command the fourteen joints.",
                "战术选择技能，学习得到的运动策略驱动十四个关节。",
            ),
            (
                "Ring out and the public down count decide the result. Every body stays in the simulation.",
                "真实出界和公开倒地读秒决定结果，淘汰后的身体仍保留在仿真中。",
            ),
        ]
        for s in plan["shots"]:
            if s["kind"] == "result":
                continue
            chapter = s["caption"][:2]
            if chapter.isdigit() and 1 <= int(chapter) <= len(lines):
                en, cn = lines[int(chapter) - 1]
            elif s["caption"].startswith("RECOVERY"):
                en, cn = (
                    "Stable foot support.",
                    "稳定足部支撑。",
                )
            else:
                continue
            cues.append((s["output_start_s"] + 0.2, en, cn))
    if not plan.get("commentary"):
        cues.append((plan["shots"][-1]["output_start_s"] + 0.1, final_en, final_cn))
    cues.sort()
    jobs = []
    for i, (at, en, cn) in enumerate(cues):
        end = cues[i + 1][0] - 0.25 if i + 1 < len(cues) else duration - 0.05
        jobs.append((str(Path(a.tts).resolve()), cache, en, end - at))
    with ThreadPoolExecutor(max_workers=4) as pool:
        voices = list(pool.map(speech, jobs))
    spoken = []
    for (at, en, cn), (pcm, record) in zip(cues, voices):
        add(at, pcm, 0.65)
        spoken.append(
            dict(
                start_s=at,
                end_s=at + len(pcm) / rate,
                text=en,
                zh_CN=cn,
                **{k: v for k, v in record.items() if k != "text"},
            )
        )
    peak = float(np.max(np.abs(audio)))
    if peak > 0.94:
        audio *= 0.94 / peak
    wav = out.with_suffix(".post-audio.wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    loudness = None
    loudness_filter = []
    if plan["edition"] in ("tactical", "finisher"):
        measured = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-i",
                str(wav),
                "-af",
                "loudnorm=I=-16:TP=-2:LRA=11:print_format=json",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        loudness = json.loads(
            measured.stderr[measured.stderr.rfind("{") : measured.stderr.rfind("}") + 1]
        )
        filt = (
            "loudnorm=I=-16:TP=-2:LRA=11:linear=false"
            f":measured_I={loudness['input_i']}:measured_TP={loudness['input_tp']}"
            f":measured_LRA={loudness['input_lra']}:measured_thresh={loudness['input_thresh']}"
            f":offset={loudness['target_offset']}"
        )
        loudness_filter = ["-af", filt, "-ar", "48000"]
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(source),
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
            "192k",
            *loudness_filter,
            "-movflags",
            "+faststart",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    shutil.copyfile(
        source.with_suffix(".frames.jsonl"), out.with_suffix(".frames.jsonl")
    )
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata.update(
        silent=False,
        video_sha256=digest(out),
        source_silent_video_sha256=digest(source),
        audio_provenance="post-produced TTS, original synthesized bed and recorded-contact-triggered foley; not sensor audio",
    )
    write_json(out.with_suffix(".json"), metadata)
    write_json(
        out.with_suffix(".audio.json"),
        dict(
            cues=spoken,
            contact_foley=foley,
            editorial_cues=editorial_cues,
            audio_builder_sha256=digest(__file__),
            source_plan_sha256=digest(plan_path),
            voice="en-GB-RyanNeural",
            peak_before_limiting=peak,
            limiter_ceiling=0.94,
            loudness_mastering=dict(
                target_LUFS=-16, target_true_peak_dBTP=-2, first_pass=loudness
            )
            if loudness
            else None,
        ),
    )
    for language, key in (("en", "text"), ("zh-CN", "zh_CN")):
        out.with_suffix("." + language + ".srt").write_text(
            "\n\n".join(
                f"{i + 1}\n{stamp(c['start_s'])} --> {stamp(c['end_s'])}\n{c[key]}"
                for i, c in enumerate(spoken)
            )
            + "\n"
        )
    print(
        json.dumps(
            dict(
                output=str(out),
                duration_s=duration,
                spoken_cues=len(spoken),
                foley_events=len(foley),
                post_produced_audio=True,
            )
        )
    )


if __name__ == "__main__":
    main()
