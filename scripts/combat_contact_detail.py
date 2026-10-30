"""Locate a solved contact from the raw log for an actual-state film hold."""

import argparse
import gzip
import json
from pathlib import Path
import mujoco
from microduck_lab.combat.world import digest
from microduck_lab.combat.episode import write_json


def detail(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    events = [
        json.loads(s) for s in (run / "runtime_events.jsonl").read_text().splitlines()
    ]
    event = next(e for e in events if e["state"] == "VALID_STRIKE")
    episode = json.loads((run / "contacts_summary.json").read_text())["episodes"][
        event["contact_episode_id"]
    ]
    contact_log = run / "contacts_full.jsonl.gz"
    if digest(contact_log) != manifest["hashes"]["contacts_full.jsonl.gz"]:
        raise ValueError("Changed solved-contact log")
    model = mujoco.MjModel.from_binary_path(str(run / "scene.mjb"))
    wanted = {event["actor"], event["target"]}
    peak = None
    with gzip.open(contact_log, "rt") as stream:
        for line in stream:
            step, rows = json.loads(line)
            if step > episode["end_step"]:
                break
            if step < episode["start_step"]:
                continue
            for row in rows:
                owners = {
                    model.body(int(model.geom_bodyid[g])).name.split("/")[0]
                    for g in row[:2]
                }
                if (
                    owners == wanted
                    and row[3] > 0
                    and (peak is None or row[3] > peak[1][3])
                ):
                    peak = (step, row)
    if peak is None:
        raise ValueError("No actual positive-force contact for the response event")
    step, row = peak
    return dict(
        schema="duckfu.contact-detail.v1",
        source_manifest_sha256=digest(run / "manifest.json"),
        source_contact_log_sha256=digest(contact_log),
        source_step=step,
        source_time_s=step * manifest["physics"]["dt"],
        source_contact_episode_id=event["contact_episode_id"],
        actor=event["actor"],
        target=event["target"],
        attack_id=event["attack_id"],
        normal_force_N=row[3],
        position_world_m=row[9:12],
        normal_world=row[12:15],
        geometry_ids=row[:2],
        raw_contact=row,
        note="Peak individual solved normal force within the first measured-response contact episode; frame hold is explicitly marked, not two seconds of sustained force.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = detail(a.run)
    write_json(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
