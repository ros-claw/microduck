"""Include unfinished recoveries; completed-only averages can hide stuck bodies."""

import argparse
import gzip
import json
from pathlib import Path


def summary(run):
    run = Path(run)
    audit = json.loads((run / "audit.json").read_text())
    episodes = []
    pending = {}
    for e in audit["events"]:
        body = e.get("body_id")
        if e["state"] == "RECOVERY_STARTED":
            pending[body] = e["time"]
        elif e["state"] in ("RECOVERY_COMPLETED", "ELIMINATED") and body in pending:
            start = pending.pop(body)
            episodes.append(
                dict(
                    body_id=body,
                    start_s=start,
                    end_s=e["time"],
                    duration_s=e["time"] - start,
                    completed=e["state"] == "RECOVERY_COMPLETED",
                    ended_by=e["state"],
                )
            )
    for body, start in pending.items():
        episodes.append(
            dict(
                body_id=body,
                start_s=start,
                end_s=audit["duration"],
                duration_s=audit["duration"] - start,
                completed=False,
                ended_by="MATCH_END",
            )
        )
    current = {}
    max_stall = {}
    starts = {}
    with gzip.open(run / "decisions.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            body = row["body_id"]
            r = row["input"]["robot"]
            t = row["time"]
            on_deck = any(
                tile["state"] in ("LOCKED", "WARNING")
                and abs(r["x"] - tile["centre_xy"][0])
                <= audit["config"]["tile_size"] / 2
                and abs(r["y"] - tile["centre_xy"][1])
                <= audit["config"]["tile_size"] / 2
                for tile in row["input"]["visible_tiles"]
            )
            stalled = (
                row["intent"].startswith("LOCAL_RECOVERY")
                and r.get("up_cos", 1) < 0.6
                and r["z"] > -0.05
                and r["speed_m_s"] < 0.04
                and on_deck
            )
            if stalled and body not in starts:
                starts[body] = t
            if stalled:
                current[body] = t - starts[body] + 0.02
                max_stall[body] = max(max_stall.get(body, 0), current[body])
            else:
                starts.pop(body, None)
    return dict(
        source=str(run),
        outcome=audit["outcome"],
        episodes=episodes,
        completed=[e["duration_s"] for e in episodes if e["completed"]],
        unfinished=[e for e in episodes if not e["completed"]],
        maximum_recovery_duration_s=max((e["duration_s"] for e in episodes), default=0),
        maximum_supported_low_motion_topple_s=max_stall,
        stall_metric="50 Hz; local recovery, up cosine <0.6, torso z >-0.05m, horizontal speed <0.04m/s, projection over a remaining tile",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = summary(a.run)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
