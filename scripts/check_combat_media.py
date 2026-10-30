"""Decode a continuous lab film and verify every frame's recorded provenance."""

import argparse
from pathlib import Path
import json
import subprocess
import imageio_ffmpeg
from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json


def check(video):
    video = Path(video)
    metadata = json.loads(video.with_suffix(".json").read_text())
    index = video.with_suffix(".frames.jsonl")
    if (
        digest(video) != metadata["video_sha256"]
        or digest(index) != metadata["provenance_sha256"]
    ):
        raise ValueError("Media digest differs")
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    run = Path(rows[0]["run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if digest(run / "manifest.json") != metadata["source_manifest_sha256"]:
        raise ValueError("Source manifest differs")
    steps = list(range(0, manifest["steps"], round(0.02 / manifest["physics"]["dt"])))
    if len(rows) != metadata["frames"] or [r["source_step"] for r in rows] != steps:
        raise ValueError("Frame/source mapping differs")
    if any(
        r["frame"] != i
        or r["speed"] != 1.0
        or r["run"] != rows[0]["run"]
        or abs(r["source_time_s"] - r["source_step"] * manifest["physics"]["dt"])
        > 1e-12
        for i, r in enumerate(rows)
    ):
        raise ValueError("Wrong time/speed/provenance")
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if count != metadata["frames"] or abs(seconds - count / 50) > 0.03:
        raise ValueError("Decoded duration/frame count differs")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(video),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return dict(
        video=str(video),
        passed=True,
        decoded_frames=count,
        duration_s=seconds,
        all_frames_mapped=True,
        continuous_1x=True,
        no_fx_declared=metadata["no_fx"],
        human_gate="PENDING_HUMAN_REVIEW",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    result = check(a.video)
    write_json(a.out, result)
    print(json.dumps(result, indent=2))
