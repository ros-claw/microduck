"""Decode a final film and verify every frame against its physical source plan."""

import argparse
import json
import gzip
from pathlib import Path
import imageio_ffmpeg
from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json


def check(video, plan_path):
    video, plan_path = Path(video), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    metadata = json.loads(video.with_suffix(".json").read_text())
    frames_path = video.with_suffix(".frames.jsonl")
    if (
        digest(video) != metadata["video_sha256"]
        or digest(frames_path) != metadata["provenance_sha256"]
    ):
        raise ValueError("Changed video or frame provenance")
    if digest(plan_path) != metadata["plan_sha256"]:
        raise ValueError("Changed film plan")
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    decoder = imageio_ffmpeg.read_frames(str(video))
    stream = next(decoder)
    decoder.close()
    if (
        count != metadata["frames"]
        or count != sum(s["frames"] for s in plan["shots"])
        or abs(stream["fps"] - 50) > 0.001
    ):
        raise ValueError("Decoded frame count/rate differs from actual-state plan")
    if tuple(stream["size"]) != (metadata["width"], metadata["height"]):
        raise ValueError("Wrong film dimensions")
    if plan["continuous"] and (not metadata["silent"] or stream.get("audio_codec")):
        raise ValueError("Continuous reference must remain silent")
    rows = [json.loads(line) for line in frames_path.read_text().splitlines()]
    if len(rows) != count:
        raise ValueError("Missing physical frame provenance")
    offset = 0
    sources, body_status = {}, {}
    for index, shot in enumerate(plan["shots"]):
        root = Path(shot["run"])
        key = str(root)
        if key not in sources:
            manifest = json.loads((root / "manifest.json").read_text())
            if digest(root / "manifest.json") != shot["source_manifest_sha256"]:
                raise ValueError("Changed source manifest")
            for name in ("states.npy", "scene.mjb"):
                if digest(root / name) != manifest["hashes"][name]:
                    raise ValueError("Changed physical capture")
            sources[key] = manifest
        manifest = sources[key]
        if shot.get("body_overlay") and key not in body_status:
            trace = root / "tactical_decisions.jsonl.gz"
            if digest(trace) != manifest["hashes"]["tactical_decisions.jsonl.gz"]:
                raise ValueError("Changed physical body annotation source")
            with gzip.open(trace, "rt") as body_stream:
                body_status[key] = {
                    (r["sim_step"], r["actor"]): r
                    for r in (json.loads(s) for s in body_stream)
                }
        if shot.get("measurement"):
            measured = shot["measurement"]
            log_path = root / "contacts_full.jsonl.gz"
            if (
                digest(log_path) != measured["source_contact_log_sha256"]
                or measured["source_contact_log_sha256"]
                != manifest["hashes"]["contacts_full.jsonl.gz"]
            ):
                raise ValueError("Changed solved-contact measurement source")
            matched = False
            with gzip.open(log_path, "rt") as log_stream:
                for line in log_stream:
                    step, contacts = json.loads(line)
                    if step == measured["source_step"]:
                        matched = measured["raw_contact"] in contacts
                        break
            raw = measured["raw_contact"]
            if (
                not matched
                or measured["normal_force_N"] != raw[3]
                or measured["geometry_ids"] != raw[:2]
                or measured["position_world_m"] != raw[9:12]
                or measured["normal_world"] != raw[12:15]
            ):
                raise ValueError(
                    "Contact annotation differs from the actual solved log"
                )
        for k in range(shot["frames"]):
            row = rows[offset + k]
            time = shot["start_s"] + k / 50 * shot["speed"]
            expected = min(
                manifest["steps"] - 1, round(time / manifest["physics"]["dt"])
            )
            if (
                row["frame"] != offset + k
                or row["shot"] != index
                or row["source_step"] != expected
                or row["speed"] != shot["speed"]
                or not row["camera_read_only"]
            ):
                raise ValueError(
                    "Frame no longer maps to the recorded integration state"
                )
            if (
                shot.get("measurement")
                and row["source_step"] != shot["measurement"]["source_step"]
            ):
                raise ValueError(
                    "Contact annotation is shown on a different physical state"
                )
            if abs(row["source_time_s"] - expected * manifest["physics"]["dt"]) > 1e-10:
                raise ValueError("Incorrect physical source clock")
            if shot.get("body_overlay"):
                control_step = expected - expected % round(
                    0.02 / manifest["physics"]["dt"]
                )
                status = body_status[key][control_step, shot["body_overlay"]]
                expected_body = dict(
                    actor=shot["body_overlay"],
                    source_control_step=control_step,
                    policy=status["command"]["policy"],
                    upright=status["observation"]["upright"],
                    foot_support=status["observation"]["foot_support"],
                )
                if row.get("body_overlay") != expected_body:
                    raise ValueError(
                        "Body overlay differs from the real motor-boundary observation"
                    )
        offset += shot["frames"]
    if (
        metadata["synthetic_inbetween_frames"]
        or not metadata["speed_from_recorded_states"]
    ):
        raise ValueError("Synthetic animation is prohibited")
    return dict(
        schema="duckfu.film-check.v1",
        video=str(video),
        video_sha256=digest(video),
        decoded_frames=count,
        fps=50,
        duration_s=duration,
        width=metadata["width"],
        height=metadata["height"],
        verified_physical_frame_mappings=len(rows),
        continuous=plan["continuous"],
        audio_codec=stream.get("audio_codec"),
        film_gate=metadata["film_gate"],
        physical_sources_pass=all(
            m["physical"]["penetration_max_m"] <= 0.003
            and m["physical"]["penetrating_p99_m"] <= 0.0015
            and m["result"]["kind"] != "NO_CONTEST_PHYSICS"
            for m in sources.values()
        ),
        passed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = check(args.video, args.plan)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
