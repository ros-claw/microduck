#!/usr/bin/env python3
"""Generate Chinese sports speech from this match's event/decision evidence.

Uses an isolated edge-tts executable, cached clips, explicit provenance and
non-overlapping speech. No commentary participates in simulation control.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import wave

import imageio_ffmpeg
import numpy as np
from microduck_lab.arena.episode import sha
from microduck_lab.arena.relay_rumble import CHARACTERS


def prepare(run, plan, out, tts):
    run, plan, out = Path(run), Path(plan), Path(out)
    audit = json.loads((run / "audit.json").read_text())
    film = json.loads(plan.read_text())
    if film["source_audit_sha256"] != sha(run / "audit.json"):
        raise ValueError("Plan/audit mismatch")
    times = np.array([r["t"] for r in film["frames"]])
    duration = len(times) / film["fps"]

    def at(t):
        return min(len(times) - 1, int(np.searchsorted(times, t))) / film["fps"]

    def name(n):
        return CHARACTERS[n]["name"]

    candidates = [
        dict(
            start_s=0.12,
            text="Jev战术生存战！四鸭同场，只能留下一只！",
            priority=100,
            evidence=dict(kind="rules", rule="LastDuckAlive"),
        )
    ]
    contacts = []
    for index, e in enumerate(audit["events"]):
        kind = e["state"]
        text = None
        priority = 0
        if kind == "ELIMINATED":
            left = 4 - sum(
                x["state"] == "ELIMINATED" and x["time"] <= e["time"]
                for x in audit["events"]
            )
            text = f"{name(e['body_id'])}跌落淘汰！" + (
                "只剩最后两只！"
                if left == 2
                else "最后的独苗！"
                if left == 1
                else "还剩三只！"
            )
            priority = 95
        elif kind == "TERMINAL" and e.get("winner"):
            text = f"站稳了！{name(e['winner'])}，唯一幸存者！"
            priority = 99
        elif kind == "RECOVERY_COMPLETED":
            text = f"{name(e['body_id'])}重新站稳！回到对抗！"
            priority = 78
        elif kind == "RECOVERY_GRACE":
            text = "地板多留一点恢复时间！碰撞照常继续！"
            priority = 70
        elif kind == "FINAL_WARNING" and e["time"] < 7:
            text = "第一块地板开始倒计时！提前撤离，一次只掉一块！"
            priority = 85
        elif kind == "RELEASED" and e.get("tile") == 12:
            raise ValueError("Permanent core must not release")
        elif kind == "FINALE_WARNING":
            text = "终局警报！外围即将坍塌，快回中心！"
            priority = 90
        elif kind == "FINALE_RING":
            text = "外圈落下！" if e["radius"] == 2 else "只剩中心台！最后的平衡对抗！"
            priority = 75
        elif kind == "BEACON_CAPTURE":
            islands = {i: f"{i}号岛" for i in range(25)}
            islands.update({12: "中心岛", 2: "西岛", 14: "北岛", 22: "东岛", 10: "南岛"})
            text = f"{name(e['body_id'])}拿下{islands[e['tile']]}！下一个点，走！"
            priority = 55
        elif kind == "BODY_CONTACT" and all(abs(e["time"] - t) > 4 for t in contacts):
            strongest = max(e["contacts"], key=lambda c: c["force_N"])
            if any(x["state"] == "ELIMINATED" and x["body_id"] in (strongest["a"], strongest["b"]) and x["time"] <= e["time"]+.5 for x in audit["events"]):
                continue
            pair = f"{name(strongest['a'])}和{name(strongest['b'])}"
            variants = [
                f"{pair}撞上了！",
                f"{pair}顶在一起！",
                f"看这次接触！{pair}贴上了！",
            ]
            text = variants[len(contacts) % 3]
            priority = 65
            contacts.append(e["time"])
        if text:
            candidates.append(
                dict(
                    start_s=(
                        max(
                            at(e["time"]) + 1.8,
                            at(
                                max(
                                    x["time"]
                                    for x in audit["events"]
                                    if x["state"] == "ELIMINATED"
                                )
                            )
                            + 3.6,
                        )
                        if kind == "TERMINAL"
                        else at(e["time"])
                    ),
                    text=text,
                    priority=priority,
                    evidence=dict(
                        kind="event", index=index, state=kind, sim_time_s=e["time"]
                    ),
                )
            )
    # A short retrospective can fit after a crowded play; its past tense is explicit.
    for index, e in enumerate(audit["events"]):
        if e["state"] == "BEACON_CAPTURE":
            islands = {i: f"{i}号岛" for i in range(25)}
            islands.update({12: "中心岛", 2: "西岛", 14: "北岛", 22: "东岛", 10: "南岛"})
            candidates.append(
                dict(
                    start_s=at(e["time"]) + 3.1,
                    text=f"{name(e['body_id'])}已经拿下{islands[e['tile']]}！",
                    priority=54,
                    group=f"capture-{index}",
                    evidence=dict(
                        kind="event",
                        index=index,
                        state=e["state"],
                        sim_time_s=e["time"],
                        retrospective=True,
                    ),
                )
            )
    # Intent is only an attempt: never claim a successful interception from intent.
    with gzip.open(run / "decisions.jsonl.gz", "rt") as f:
        decisions = [json.loads(line) for line in f]
    phrases = {
        "JEV_CAPTURE": "Jev选择抢点！张三向目标推进！",
        "JEV_INTERCEPT": "Jev选择截路！老六尝试卡住对手！",
        "JEV_EVADE": "二呆选择避让，拉开距离！",
        "JEV_RETREAT_CORE": "卷王向中心撤退，争夺安全位置！",
    }
    actors = {"JEV_CAPTURE": "lavender", "JEV_INTERCEPT": "sky", "JEV_EVADE": "cream", "JEV_RETREAT_CORE": "graphite"}
    for intent, text in phrases.items():
        rows = [
            (i, d)
            for i, d in enumerate(decisions)
            if d["intent"] == intent
            and (intent == "ESCAPE_CROWD" or d["body_id"] == actors[intent])
            and d["time"] > 5
            and (
                intent != "ESCAPE_CROWD"
                or (
                    d["input"]["robot"]["z"] > 0.08
                    and d["input"]["robot"]["speed_m_s"] > 0.02
                )
            )
        ]
        # Offer a few real occurrences; scheduling chooses at most one per intent.
        last = -100
        for i, d in rows:
            if d["time"] - last < 2:
                continue
            last = d["time"]
            candidates.append(
                dict(
                    start_s=at(d["time"]),
                    text=text or f"{name(d['body_id'])}尝试脱离拥挤！",
                    priority=72
                    if intent in ("JEV_CAPTURE", "JEV_INTERCEPT")
                    else 60,
                    group=intent,
                    evidence=dict(
                        kind="decision", index=i, intent=intent, sim_time_s=d["time"]
                    ),
                )
            )
            if sum(c.get("group") == intent for c in candidates) >= 16:
                break
    cache = out.parent / "survival-rumble-voice-cache"
    cache.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    occupied = []
    selected = []
    groups = set()
    for c in sorted(candidates, key=lambda x: (-x["priority"], x["start_s"])):
        if c.get("group") in groups:
            continue
        start = c["start_s"]
        estimate = len(c["text"]) / 6.3 + 0.25
        if start + estimate > duration:
            continue
        if any(start < b + 0.65 and start + estimate > a - 0.65 for a, b in occupied):
            continue
        rate = "+18%" if c["priority"] >= 65 else "+10%"
        voice = "zh-CN-YunjianNeural"
        digest = hashlib.sha256((voice + rate + c["text"]).encode()).hexdigest()
        mp3 = cache / (digest + ".mp3")
        if not mp3.exists() or mp3.stat().st_size < 1000:
            for attempt in range(4):
                result = subprocess.run(
                    [
                        str(tts),
                        "--voice",
                        voice,
                        "--rate",
                        rate,
                        "--text",
                        c["text"],
                        "--write-media",
                        str(mp3),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=50,
                )
                if (
                    result.returncode == 0
                    and mp3.exists()
                    and mp3.stat().st_size > 1000
                ):
                    break
                time.sleep(1 + attempt)
            else:
                raise RuntimeError("Speech service failed for " + c["text"])
        raw = subprocess.check_output(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(mp3),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-",
            ]
        )
        pcm = np.frombuffer(raw, dtype="<f4").astype(float)
        # Remove codec/trailing silence, retain breaths and natural pauses.
        active = np.flatnonzero(np.abs(pcm) > 0.003)
        if not len(active):
            raise ValueError("Silent speech")
        pcm = pcm[max(0, active[0] - 2400) : min(len(pcm), active[-1] + 4800)]
        length = len(pcm) / 48000
        if start + length > duration or any(
            start < b + 0.45 and start + length > a - 0.45 for a, b in occupied
        ):
            continue
        wav = cache / (digest + ".wav")
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(48000)
            f.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
        c.update(
            end_s=start + length,
            voice=voice,
            rate=rate,
            wav=str(wav.resolve()),
            wav_sha256=sha(wav),
        )
        selected.append(c)
        occupied.append((start, start + length))
        if c.get("group"):
            groups.add(c["group"])
    selected.sort(key=lambda c: c["start_s"])
    mandatory = [e for e in audit["events"] if e["state"] in ("ELIMINATED", "TERMINAL")]
    covered = {
        c["evidence"].get("index") for c in selected if c["evidence"]["kind"] == "event"
    }
    missing = [
        i for i, e in enumerate(audit["events"]) if e in mandatory and i not in covered
    ]
    result = dict(
        generator_source_sha256=sha(Path(__file__)),
        source_audit_sha256=sha(run / "audit.json"),
        plan_sha256=sha(plan),
        duration_s=duration,
        voice="zh-CN-YunjianNeural",
        synthetic=True,
        cues=selected,
        unvoiced_terminal_events=missing,
    )
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    def stamp(t):
        ms = round(t * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"

    out.with_suffix(".srt").write_text(
        "\n\n".join(
            f"{i + 1}\n{stamp(c['start_s'])} --> {stamp(c['end_s'])}\n{c['text']}"
            for i, c in enumerate(selected)
        )
        + "\n"
    )
    return dict(
        cues=len(selected), duration_s=duration, unvoiced_terminal_events=missing
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tts", default=".tts-venv/bin/edge-tts")
    a = p.parse_args()
    print(json.dumps(prepare(a.run, a.plan, a.out, a.tts), ensure_ascii=False))
