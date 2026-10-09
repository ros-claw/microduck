#!/usr/bin/env python3
"""Decode the final film and validate speech/frame provenance against its capture."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from microduck_lab.arena.episode import sha


def check(run, video):
    run, video = Path(run), Path(video)
    audit = json.loads((run / "audit.json").read_text())
    manifest = json.loads(video.with_suffix(".json").read_text())
    assert manifest["source_audit_sha256"] == sha(run / "audit.json")
    assert manifest["source_trajectory_sha256"] == sha(run / "trajectory.npz")
    assert manifest["video_sha256"] == sha(video)
    assert audit["outcome"]["status"] == "WINNER"
    winner = audit["outcome"]["winner"]
    assert audit["final_alive"] == audit["outcome"]["alive"] == [winner]
    assert audit["final_observations"][winner]["upright"]
    assert 12 in audit["final_observations"][winner]["supporting_tiles"]
    assert len([e for e in audit["events"] if e["state"] == "ELIMINATED"]) == 3
    with gzip.open(video.with_suffix(".frames.jsonl.gz"), "rt") as f:
        frames = [json.loads(line) for line in f]
    times = np.array([r["sim_time_s"] for r in frames])
    assert np.all(np.diff(times) >= 0)
    assert all(
        r["state_index"] == round(r["sim_time_s"] / audit["config"]["dt"])
        for r in frames
    )
    assert max(r["state_index"] for r in frames) < audit["steps"]
    speech = manifest["commentary"]
    cues = speech["cues"]
    assert speech["source_audit_sha256"] == sha(run / "audit.json")
    assert not speech["unvoiced_terminal_events"]
    with gzip.open(run / "decisions.jsonl.gz", "rt") as f:
        decisions = [json.loads(line) for line in f]
    last = 0
    for cue in cues:
        assert 0 <= cue["start_s"] < cue["end_s"] <= manifest["duration_s"]
        assert cue["start_s"] >= last
        last = cue["end_s"]
        assert sha(Path(cue["wav"])) == cue["wav_sha256"]
        e = cue["evidence"]
        if e["kind"] == "event":
            row = audit["events"][e["index"]]
            assert row["state"] == e["state"] and row["time"] == e["sim_time_s"]
        elif e["kind"] == "decision":
            row = decisions[e["index"]]
            assert row["intent"] == e["intent"] and row["time"] == e["sim_time_s"]
            middle = round((cue["start_s"] + cue["end_s"]) / 2 * manifest["fps"])
            assert frames[middle]["shot"] == "character_" + row["body_id"]
        else:
            continue
        source_frame = frames[
            min(len(frames) - 1, round(cue["start_s"] * manifest["fps"]))
        ]
        assert source_frame["sim_time_s"] + audit["config"]["dt"] >= e["sim_time_s"]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert count == len(frames) == manifest["frames"]
    assert abs(duration - len(frames) / manifest["fps"]) < 0.04 and duration < 120
    raw = subprocess.check_output(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-",
        ]
    )
    pcm = np.frombuffer(raw, dtype="<f4")
    assert len(pcm) > 0 and np.isfinite(pcm).all()
    rms = float(np.sqrt(np.mean(pcm.astype(float) ** 2)))
    assert 0.02 < rms < 0.3
    return dict(
        decoded=True,
        frames=count,
        duration_s=duration,
        chronological=True,
        speech_cues=len(cues),
        speech_and_frames_provenance_checked=True,
        audio_peak=float(np.max(np.abs(pcm))),
        audio_rms=rms,
        audio_near_full_scale_fraction=float(np.mean(np.abs(pcm) >= 0.99)),
        outcome=audit["outcome"],
        final_alive=audit["final_alive"],
        video_sha256=sha(video),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = check(a.run, a.video)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
