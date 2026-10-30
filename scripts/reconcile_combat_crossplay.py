"""Derive strict completed/interrupted attack statistics without altering raw runs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from microduck_lab.combat import events as attribution
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest


def reconcile(path):
    if getattr(attribution, "ATTRIBUTION_SCHEMA", None) != "duckfu.attack-outcomes.v2":
        raise RuntimeError(
            "Use the completed-window attribution implementation, not legacy MISS labels"
        )
    path = Path(path)
    report = deepcopy(json.loads(path.read_text()))
    for row in report["rows"]:
        directory = Path(row["run"])
        manifest = json.loads((directory / "manifest.json").read_text())
        raw = [
            json.loads(s)
            for s in (directory / "runtime_events.jsonl").read_text().splitlines()
        ]
        episodes = json.loads((directory / "contacts_summary.json").read_text())[
            "episodes"
        ]
        teams = {n: b["team"] for n, b in manifest["spawns"].items()}
        base = [
            e
            for e in raw
            if e["state"]
            not in (
                "MISS",
                "CONTACTED_ATTACK",
                "ATTACK_INTERRUPTED",
                "COUNTER_SEQUENCE",
            )
        ]
        outcomes = attribution.attack_outcomes(base, episodes, teams)
        counter = attribution.counter_sequences(base + outcomes)
        row["attack_outcomes_v2"] = outcomes
        row["event_counts"]["MISS"] = sum(e["state"] == "MISS" for e in outcomes)
        row["event_counts"]["ATTACK_INTERRUPTED"] = sum(
            e["state"] == "ATTACK_INTERRUPTED" for e in outcomes
        )
        row["event_counts"]["COUNTER_SEQUENCE"] = len(counter)
        row["event_counts"]["RECOVERED"] = sum(
            e["state"] == "RECOVERED" and "duration_s" in e for e in base
        )
        row["event_counts"]["DOWN_COUNT_CLEARED"] = sum(
            e["state"] == "RECOVERED" and "duration_s" not in e for e in base
        )
    for key, value in report["matrix"].items():
        group = [
            r for r in report["rows"] if r["candidate"] + " vs " + r["opponent"] == key
        ]
        value["counter_sequences"] = sum(
            r["event_counts"]["COUNTER_SEQUENCE"] for r in group
        )
        value["completed_misses"] = sum(r["event_counts"]["MISS"] for r in group)
        value["interrupted_attacks"] = sum(
            r["event_counts"]["ATTACK_INTERRUPTED"] for r in group
        )
        value["guard_recoveries"] = sum(r["event_counts"]["RECOVERED"] for r in group)
    report["derived_event_attribution"] = dict(
        schema=attribution.ATTRIBUTION_SCHEMA,
        source_report_sha256=digest(path),
        postprocessor_sha256=digest(__file__),
        attribution_source_sha256=digest(Path(attribution.__file__)),
        physical_runs_unmodified=True,
        note="Raw legacy MISS included interrupted no-contact windows. This derived report separates completed misses and interruptions; RECOVERED counts guard events, with referee down-count clears separate. Controls, contacts, measured strikes, rewards, physical referee outcomes and W/D/L are unchanged.",
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = reconcile(a.report)
    write_json(a.out, result)
    print(json.dumps(result["matrix"], indent=2))


if __name__ == "__main__":
    main()
