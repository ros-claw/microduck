"""Archive all body qualification outcomes, censoring, and matched first falls."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--matches", required=True)
p.add_argument("--states", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
root = Path(a.matches)
report = json.loads((root / "report.json").read_text())
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
audits = []
for row in report["rows"]:
    directory = Path(row["directory"])
    events = [
        json.loads(s)
        for s in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    eliminated = {
        e["actor"]: e["time_s"]
        for e in events
        if e["state"] in ("RING_OUT", "DOWN_COUNT")
    }
    body = np.load(directory / "controls.npz")["body"]
    names = list(json.loads((directory / "manifest.json").read_text())["spawns"])
    alive_seconds = down_seconds = 0.0
    for j, name in enumerate(names):
        mask = np.arange(len(body)) / 50 < eliminated.get(name, float("inf"))
        alive_seconds += float(mask.sum() / 50)
        down_seconds += float(((body[:, j, 3] < 0.55) & mask).sum() / 50)
    row.update(
        alive_duck_seconds=alive_seconds,
        down_duck_seconds=down_seconds,
        ringouts=sum(e["state"] == "RING_OUT" for e in events),
        guard_recovered_events=sum(
            e["state"] == "RECOVERED" and "duration_s" in e for e in events
        ),
    )
    for episode in row["recovery_episodes"]:
        start, end = episode["start_s"], episode["observed_until_s"]
        complete = episode["common_complete_after_s"]
        lost = (
            episode["actor"] in eliminated
            and eliminated[episode["actor"]] <= end + 1e-9
        )
        episode["observed_window_s"] = end - start
        episode["status"] = (
            "recovered"
            if complete is not None
            else "eliminated"
            if lost
            else "censored"
            if end - start < 10
            else "unrecovered"
        )
        episode["censored_before_3s"] = (
            complete is None and not lost and end - start < 3
        )
    audits.append(
        dict(
            directory=str(directory),
            files={
                name: json.loads((directory / name).read_text())
                for name in (
                    "manifest.json",
                    "contacts_summary.json",
                    "metrics.json",
                    "referee.json",
                    "model_inventory.json",
                )
            },
            runtime_events=events,
        )
    )
summary = []
for mode in sorted({r["mode"] for r in report["rows"]}):
    batch = [r for r in report["rows"] if r["mode"] == mode]
    valid = [r for r in batch if r["physical_pass"]]
    episodes = [e for r in valid for e in r["recovery_episodes"]]
    times = [
        e["common_complete_after_s"]
        for e in episodes
        if e["common_complete_after_s"] is not None
    ]
    summary.append(
        dict(
            mode=mode,
            games=len(batch),
            physical_pass=len(valid),
            recovery_starts=len(episodes),
            completed=len(times),
            within_3s=sum(e["within_3s"] for e in episodes),
            statuses=dict(Counter(e["status"] for e in episodes)),
            censored_before_3s=sum(e["censored_before_3s"] for e in episodes),
            completed_only_median_s=float(np.median(times)) if times else None,
            down_events=sum(r["event_counts"]["DOWN"] for r in valid),
            ringouts=sum(r["ringouts"] for r in valid),
            alive_duck_seconds=sum(r["alive_duck_seconds"] for r in valid),
            down_duck_seconds=sum(r["down_duck_seconds"] for r in valid),
            outcomes=dict(Counter(r["result"]["kind"] for r in valid)),
        )
    )
pairs = []
for key in sorted({(r["seed"], r["swap"]) for r in report["rows"]}):
    batch = [r for r in report["rows"] if (r["seed"], r["swap"]) == key]
    starts = [
        min(r["recovery_episodes"], key=lambda e: e["start_s"])
        if r["recovery_episodes"]
        else None
        for r in batch
    ]
    same_start = (
        all(e is not None for e in starts)
        and len({(e["actor"], e["start_s"]) for e in starts}) == 1
    )
    tapes = [np.load(Path(r["directory"]) / "controls.npz") for r in batch]
    prefix = round(starts[0]["start_s"] * 50) if same_start else 0
    pairs.append(
        dict(
            seed=key[0],
            swap=key[1],
            identical_initial_state=len({r["initial_state_sha256"] for r in batch})
            == 1,
            same_first_stumble=same_start,
            identical_pre_stumble_controls=same_start
            and all(
                np.array_equal(tapes[0]["ctrl"][:prefix], t["ctrl"][:prefix])
                for t in tapes[1:]
            ),
            identical_first_body_snapshot=same_start
            and all(
                np.array_equal(tapes[0]["body"][prefix], t["body"][prefix])
                for t in tapes[1:]
            ),
            first_stumble_actor=starts[0]["actor"] if same_start else None,
            first_stumble_s=starts[0]["start_s"] if same_start else None,
            latency_s={
                r["mode"]: e["common_complete_after_s"] if e else None
                for r, e in zip(batch, starts, strict=True)
            },
        )
    )
states = json.loads(Path(a.states).read_text())
subgroups = []
for threshold in (0.8, 0.0):
    for method in ("guard", "stand_first", "stabilize_candidate"):
        batch = [
            r
            for r in states["rows"]
            if r["method"] == method and r["source"]["upright"] < threshold
        ]
        times = [
            r["complete_s"]
            for r in batch
            if r["physical_pass"] and r["complete_s"] is not None
        ]
        subgroups.append(
            dict(
                initial_upright_lt=threshold,
                method=method,
                trials=len(batch),
                within_3s=sum(r["physical_pass"] and r["within_3s"] for r in batch),
                recovery_to_walk=sum(
                    r["physical_pass"] and r["recovery_to_walk_pass"] for r in batch
                ),
                completed_only_median_s=float(np.median(times)) if times else None,
            )
        )
selection = json.loads(Path("training/combat/recovery-v2/selection.json").read_text())
result = dict(
    schema="duckfu.body-recovery-final.v1",
    selection=selection,
    state_qualification=states["summary"],
    state_subgroups=subgroups,
    snapshot_source_trajectories=len(
        {r["source"]["source_run"] for r in states["rows"]}
    ),
    snapshot_source_recovery_events=len(
        {(r["source"]["source_run"], r["source"]["episode"]) for r in states["rows"]}
    ),
    state_qualification_sha256=digest(a.states),
    match_summary=summary,
    paired_initial_and_first_fall=pairs,
    physics=dict(
        games=len(report["rows"]),
        passed=sum(r["physical_pass"] for r in report["rows"]),
        maximum_penetration_m=max(
            r["physical"]["penetration_max_m"] for r in report["rows"]
        ),
        maximum_penetrating_p99_m=max(
            r["physical"]["penetrating_p99_m"] for r in report["rows"]
        ),
        maximum_torque_Nm=max(r["physical"]["torque_max_Nm"] for r in report["rows"]),
    ),
    rows=report["rows"],
    interpretation="Four held-out base seeds, both sides, three modes: correlated games. Snapshot trials are correlated offsets from four other held-out trajectories. Completed-only medians exclude failures; status and censoring counts are retained. New recovery counts are not independent matched falls after trajectories diverge. No hardware or invariant winner claim.",
    summarizer_sha256=digest(__file__),
)
write_json(out / "final-evaluation.json", result)
with gzip.open(out / "raw-match-audits.json.gz", "wt") as f:
    json.dump(audits, f, ensure_ascii=False, indent=2)
with gzip.open(out / "raw-state-qualification.json.gz", "wt") as f:
    json.dump(states, f, ensure_ascii=False, indent=2)
print(
    json.dumps(dict(summary=summary, physics=result["physics"], pairs=pairs), indent=2)
)
