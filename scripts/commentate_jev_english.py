#!/usr/bin/env python3
"""English sports edition of the accepted film, retaining its event evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import imageio_ffmpeg
import numpy as np
from microduck_lab.arena.episode import sha
from microduck_lab.presentation.jev_english import CHARACTERS_EN

TEXT = [
    "Four ducks. One survivor. Jev calls the plays!",
    "The first floor is counting down! Move before it drops!",
    "Dash is back up! Back in the game!",
    "Jev sends Dash after the beacon!",
    "Sly the schemer tries to cut off the chase!",
    "Ace, the relentless challenger, is back up!",
    "Sly and Ace collide! Neither gives way!",
    "Ace meets Dash! Look at that body contact!",
    "Sly and Ace, shoulder to shoulder again!",
    "Ace and Dash lock horns! Watch that contact!",
    "Ace finds his feet! Still in the fight!",
    "He's up again! Ace refuses to stay down!",
    "Ace goes over! He's out! Three ducks remain!",
    "Sly and Dash collide! The pressure is building!",
    "Sly meets Dash again! There's no room to spare!",
    "Another collision! Sly and Dash, on the edge!",
    "Dash falls out! We're down to the final two!",
    "Sly has claimed the centre beacon!",
    "Waddles is out! Only Sly remains!",
    "He's standing firm! Sly is the sole survivor!",
]


def prepare(run, source, plan, out, tts):
    source, out, run = Path(source), Path(out), Path(run)
    original = json.loads(source.read_text())
    assert original["source_audit_sha256"] == sha(run / "audit.json")
    assert len(original["cues"]) == len(TEXT)
    cues = [{**c, "text": text} for c, text in zip(original["cues"], TEXT)]
    cues.append(
        dict(
            start_s=9.65,
            text="Dash the daredevil. Waddles the survivor.",
            priority=80,
            evidence=dict(kind="rules", rule="character personalities"),
        )
    )
    cues.sort(key=lambda c: c["start_s"])
    cache = out.parent / "jev-english-voice-cache"
    cache.mkdir(exist_ok=True, parents=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    voice = "en-GB-RyanNeural"
    for index, cue in enumerate(cues):
        limit = (
            cues[index + 1]["start_s"] - 0.46
            if index + 1 < len(cues)
            else original["duration_s"] - 0.35
        ) - cue["start_s"]
        for rate in ["+12%", "+20%", "+28%", "+34%"]:
            digest = hashlib.sha256((voice + rate + cue["text"]).encode()).hexdigest()
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
                            "--pitch",
                            "+3Hz",
                            "--text",
                            cue["text"],
                            "--write-media",
                            str(mp3),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=55,
                    )
                    if (
                        result.returncode == 0
                        and mp3.exists()
                        and mp3.stat().st_size > 1000
                    ):
                        break
                else:
                    raise RuntimeError(
                        "English speech synthesis failed: " + cue["text"]
                    )
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
            active = np.flatnonzero(np.abs(pcm) > 0.003)
            assert len(active), "Silent commentary"
            pcm = pcm[max(0, active[0] - 2400) : min(len(pcm), active[-1] + 4800)]
            length = len(pcm) / 48000
            if length <= limit:
                break
        else:
            raise ValueError(
                f"English cue will not fit: {cue['text']} ({length:.2f} > {limit:.2f}s)"
            )
        wav = cache / (digest + ".wav")
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(48000)
            f.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
        cue.update(
            end_s=cue["start_s"] + length,
            voice=voice,
            rate=rate,
            wav=str(wav.resolve()),
            wav_sha256=sha(wav),
        )
        print(f"{index + 1}/{len(cues)} {length:.2f}s {cue['text']}", flush=True)
    result = dict(
        generator_source_sha256=sha(Path(__file__)),
        source_audit_sha256=sha(run / "audit.json"),
        plan_sha256=sha(Path(plan)),
        source_chinese_speech_sha256=sha(source),
        duration_s=original["duration_s"],
        language="en",
        voice=voice,
        synthetic=True,
        characters=CHARACTERS_EN,
        cues=cues,
        unvoiced_terminal_events=[],
    )
    out.write_text(json.dumps(result, indent=2) + "\n")

    def stamp(t):
        ms = round(t * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"

    out.with_suffix(".srt").write_text(
        "\n\n".join(
            f"{i + 1}\n{stamp(c['start_s'])} --> {stamp(c['end_s'])}\n{c['text']}"
            for i, c in enumerate(cues)
        )
        + "\n"
    )
    return dict(cues=len(cues), language="en", duration_s=result["duration_s"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tts", default=".tts-venv/bin/edge-tts")
    a = p.parse_args()
    print(json.dumps(prepare(a.run, a.source, a.plan, a.out, a.tts)))
