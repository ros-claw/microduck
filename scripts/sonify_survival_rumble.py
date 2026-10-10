#!/usr/bin/env python3
"""Original synthetic sound cues aligned to recorded events, not measured audio."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import wave
import numpy as np
import imageio_ffmpeg
from microduck_lab.arena.episode import sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True)
p.add_argument("--video", required=True)
p.add_argument("--out", required=True)
p.add_argument("--speech", required=True)
a = p.parse_args()
run = Path(a.run)
video = Path(a.video)
out = Path(a.out)
audit = json.loads((run / "audit.json").read_text())
with gzip.open(video.with_suffix(".frames.jsonl.gz"), "rt") as f:
    frames = [json.loads(line) for line in f]
times = np.array([r["sim_time_s"] for r in frames])
bounds = [0, *list(np.flatnonzero(np.diff(times) < 0) + 1), len(times)]
rate = 48000
samples = np.zeros(round(len(frames) / 50 * rate) + rate, dtype=np.float64)
rng = np.random.default_rng(20261009)
cues = []
last = {}
for e in audit["events"]:
    kind = e["state"]
    if kind not in (
        "LOADED",
        "CRACKING",
        "RELEASED",
        "BODY_CONTACT",
        "ELIMINATED",
        "TERMINAL",
        "BEACON_CAPTURE",
        "BEACON_MOVED",
    ):
        continue
    # A whole ring release is one audible collapse, rather than 24 stacked tones.
    key = (kind, None if kind == "RELEASED" else e.get("tile"))
    if e["time"] - last.get(key, -100) < 0.25:
        continue
    last[key] = e["time"]
    occurrences = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        segment_times = times[lo:hi]
        if (
            not len(segment_times)
            or not segment_times[0] <= e["time"] <= segment_times[-1]
        ):
            continue
        occurrences.append(lo + int(np.searchsorted(segment_times, e["time"])))
    if not occurrences:
        continue
    duration = {
        "LOADED": 0.05,
        "CRACKING": 0.16,
        "RELEASED": 0.5,
        "BODY_CONTACT": 0.12,
        "ELIMINATED": 0.3,
        "TERMINAL": 0.8,
        "BEACON_CAPTURE": 0.4,
        "BEACON_MOVED": 0.25,
    }[kind]
    t = np.arange(int(duration * rate)) / rate
    noise = rng.normal(0, 1, len(t))
    if kind == "TERMINAL":
        signal = (
            sum(np.sin(2 * np.pi * f * t) for f in (523.25, 659.25, 783.99))
            / 3
            * np.exp(-t * 2)
            * 0.16
        )
    elif kind in ("BEACON_CAPTURE", "BEACON_MOVED"):
        base = 659.25 if kind == "BEACON_CAPTURE" else 440.0
        signal = (
            (np.sin(2 * np.pi * base * t) + 0.5 * np.sin(2 * np.pi * base * 1.5 * t))
            * np.exp(-t * 9)
            * 0.14
        )
    elif kind == "RELEASED":
        signal = (
            (np.sin(2 * np.pi * (180 * t - 90 * t * t)) * 0.5 + noise * 0.2)
            * np.exp(-t * 7)
            * 0.2
        )
    elif kind == "BODY_CONTACT":
        signal = (np.sin(2 * np.pi * 120 * t) + noise * 0.3) * np.exp(-t * 38) * 0.14
    elif kind == "ELIMINATED":
        signal = np.sin(2 * np.pi * (440 * t - 360 * t * t)) * np.exp(-t * 10) * 0.13
    else:
        signal = (
            noise
            * np.exp(-t * (70 if kind == "LOADED" else 35))
            * (0.05 if kind == "LOADED" else 0.10)
        )
    for index in occurrences:
        when = index / 50
        start = round(when * rate)
        samples[start : start + len(signal)] += signal
        cues.append(
            dict(kind=kind, sim_time_s=e["time"], video_time_s=when, synthesized=True)
        )
# Quiet original rhythmic bed, distinct from the event-aligned physical Foley.
for beat in np.arange(0, len(frames) / 50, 60 / 108):
    start = round(beat * rate)
    t = np.arange(round(0.15 * rate)) / rate
    tone = np.sin(2 * np.pi * (75 * t - 90 * t * t)) * np.exp(-t * 32) * 0.035
    samples[start : start + len(tone)] += tone
# Synthetic sports voice and Foley are explicitly identified in the manifest.
speech_path = Path(a.speech)
speech = json.loads(speech_path.read_text())
if speech["source_audit_sha256"] != sha(run / "audit.json"):
    raise ValueError("Speech belongs to another match")
voice = np.zeros_like(samples)
duck = np.ones_like(samples)
for cue in speech["cues"]:
    path = Path(cue["wav"])
    if sha(path) != cue["wav_sha256"]:
        raise ValueError("Speech clip changed")
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != rate or f.getnchannels() != 1:
            raise ValueError("Voice format")
        pcm = (
            np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(float)
            / 32768
        )
    pcm *= 0.72 / max(0.01, float(np.max(np.abs(pcm))))
    start = round(cue["start_s"] * rate)
    end = start + len(pcm)
    voice[start:end] += pcm
    # Smooth ducking around speech preserves intelligibility without pumping.
    fade = round(0.12 * rate)
    lo = max(0, start - fade)
    hi = min(len(samples), end + fade)
    envelope = np.ones(hi - lo) * 0.22
    attack = start - lo
    release = hi - end
    if attack:
        envelope[:attack] = np.linspace(1, 0.22, attack)
    if release:
        envelope[-release:] = np.linspace(0.22, 1, release)
    duck[lo:hi] = np.minimum(duck[lo:hi], envelope)
samples = samples * duck + voice
samples = np.clip(samples[: round(len(frames) / 50 * rate)], -0.95, 0.95)
wav = out.with_suffix(".wav")
with wave.open(str(wav), "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(rate)
    f.writeframes((samples * 32767).astype("<i2").tobytes())
# Two-pass loudness mastering makes speech comfortable to hear without clipping.
measurement = subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "info",
        "-i",
        str(wav),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ],
    capture_output=True,
    text=True,
    check=True,
)
start = measurement.stderr.rfind("{")
end = measurement.stderr.find("}", start) + 1
loudness = json.loads(measurement.stderr[start:end])
master_filter = (
    "loudnorm=I=-16:TP=-1.5:LRA=11:linear=false"
    + f":measured_I={loudness['input_i']}:measured_TP={loudness['input_tp']}"
    + f":measured_LRA={loudness['input_lra']}:measured_thresh={loudness['input_thresh']}"
    + f":offset={loudness['target_offset']}"
)
subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-i",
        str(wav),
        "-c:v",
        "copy",
        "-af",
        master_filter,
        "-ar",
        "48000",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ],
    check=True,
)
wav.unlink()
manifest = json.loads(video.with_suffix(".json").read_text())
manifest.update(
    video_sha256=sha(out),
    audio="synthetic Mandarin sports commentary (Yunjian Neural), original event Foley and ducked rhythmic bed; not measured robot sound",
    commentary_source_sha256=sha(speech_path),
    commentary=speech,
    spoken_language="zh-CN",
    audio_mastering="two-pass loudnorm, target -16 LUFS / -1.5 dBTP / 11 LU LRA",
    audio_pre_master_measurement=loudness,
    audio_cues=cues,
    audio_source_sha256=sha(Path(__file__)),
)
out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
# Audio does not change frame mapping.
out.with_suffix(".frames.jsonl.gz").write_bytes(
    video.with_suffix(".frames.jsonl.gz").read_bytes()
)
print(
    json.dumps(
        dict(video=str(out), duration_s=len(frames) / 50, cues=len(cues)), indent=2
    )
)
