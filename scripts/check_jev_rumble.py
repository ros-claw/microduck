"""Check all-contact physical gates, progressive rules and applied Jev provenance."""
import argparse
import gzip
import json
from pathlib import Path
import statistics

from microduck_lab.arena.episode import sha
from microduck_lab.arena.jev_tactics import validate
from microduck_lab.arena.progressive_collapse import connected


def check(run):
    run = Path(run)
    audit = json.loads((run/"audit.json").read_text())
    assert audit["tactical_backend"] == "live_jev_shared_batch"
    for name, digest in audit["hashes"].items():
        assert sha(run/name) == digest
    assert audit["finite"] and not audit["time_reset"] and audit["external_forces_zero"]
    assert not any(audit["solver_warnings"])
    assert audit["penetration_max_m"] <= .003
    assert audit["motor_force_max_Nm"] <= .6405001
    assert audit["arena_motor_force_max_Nm"] <= 4.000001
    minimum_warning = audit["collapse_rules"]["warning_s"]
    active = set(range(25))
    deadlines, releases = {}, []
    for event in audit["events"]:
        if event["state"] == "FINAL_WARNING":
            assert not deadlines
            assert event["release_at_s"]-event["time"] >= minimum_warning-1e-9
            deadlines[event["tile"]] = (event["time"], event["release_at_s"])
        if event["state"] == "RECOVERY_GRACE":
            start, original = deadlines[event["tile"]]
            assert event["release_at_s"]-original <= 1.2+1e-9
            deadlines[event["tile"]] = (start, event["release_at_s"])
        if event["state"] == "RELEASED":
            start, deadline = deadlines.pop(event["tile"])
            assert event["tile"] != 12 and event["time"] >= deadline-1e-9
            assert event["time"]-start >= minimum_warning-1e-9
            active.remove(event["tile"])
            assert connected(active)
            releases.append(event["time"])
    assert releases and releases[0] <= 10.5
    gaps = [b-a for a, b in zip(releases, releases[1:])]
    assert all(gap >= minimum_warning-1e-9 for gap in gaps)
    rows = json.loads((run/"jev-requests.json").read_text())
    applied = 0
    goals_different_from_beacon = 0
    intents = {}
    with gzip.open(run/"decisions.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            intents[d["intent"]] = intents.get(d["intent"], 0)+1
            request_id = d["jev_request_id"]
            if request_id is None:
                continue
            applied += 1
            row = rows[request_id]
            assert row["accepted"][d["body_id"]]
            plan = validate(row["raw"], row["request"]["state"]["candidates"])[d["body_id"]]
            assert plan["tile"] == d["plan"]["tile"] == d["target"]
            assert d["intent"] == "JEV_"+plan["action"]
            assert row["delivered_s"] <= d["time"] <= d["plan"]["expires_s"]
            goals_different_from_beacon += plan["tile"] != d["input"]["island"]
    assert applied > 0 and goals_different_from_beacon > 0
    latencies = [r["latency_ms"] for r in rows if "latency_ms" in r]
    winner = audit["outcome"].get("winner")
    strict = bool(winner and audit["final_alive"] == [winner] and audit["final_observations"][winner]["upright"]
                  and bool(audit["final_observations"][winner]["supporting_tiles"])
                  and (audit["outcome"].get("support_rule") == "any remaining grounded tile" or 12 in audit["final_observations"][winner]["supporting_tiles"]))
    return dict(physical_gates_pass=True, progressive_rules_pass=True, real_jev_provenance_pass=True,
                strict_sole_winner=strict, outcome=audit["outcome"], first_release_s=releases[0],
                released_tiles=len(releases), minimum_release_gap_s=min(gaps) if gaps else None,
                applied_jev_decisions=applied, goals_different_from_beacon=goals_different_from_beacon,
                requests=len(rows), latency_median_ms=statistics.median(latencies), latency_max_ms=max(latencies),
                input_tokens=sum(r.get("raw", {}).get("usage", {}).get("input_tokens", 0) for r in rows),
                output_tokens=sum(r.get("raw", {}).get("usage", {}).get("output_tokens", 0) for r in rows),
                recoveries=[e for e in audit["events"] if e["state"] in ("RECOVERY_STARTED", "RECOVERY_COMPLETED", "RECOVERY_SKILL_STARTED")],
                intent_counts=intents, source_audit_sha256=sha(run/"audit.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = check(a.run)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
