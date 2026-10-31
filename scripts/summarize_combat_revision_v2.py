"""Archive paired revision evidence without changing actors or thresholds."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest
from measure_combat_response import measure


def summarize(report, actor_commit):
    report = Path(report)
    source = json.loads(report.read_text())
    rows, paired = [], []
    for row in source["rows"]:
        directory = Path(row["source_run"])
        response = measure(directory)
        events = [
            json.loads(s)
            for s in (directory / "runtime_events.jsonl").read_text().splitlines()
        ]
        adoptions = [e for e in events if e["state"] == "TEAM_ASSIGNMENT_ADOPTED"]
        team = row.get("team", {})
        rows.append(
            dict(
                group=row["group"],
                revision=row["revision"],
                seed=row["seed"],
                swap=row["swap"],
                source_run=str(directory.resolve()),
                source_manifest_sha256=digest(directory / "manifest.json"),
                source_sha256=row["source_sha256"],
                result=row["result"],
                physical_pass=row["physical_pass"],
                physical=row["physical"],
                events=row["event_counts"],
                red_down_fraction=row["red_down_fraction"],
                clean_counters=response["clean_evade_counters"],
                strict_interventions=team.get("interventions", []),
                angular_position_seconds=team.get(
                    "same_target_different_role_45_135_degree_seconds"
                ),
                adopted_assignments=adoptions,
            )
        )
    for key in sorted({(r["group"], r["seed"], r["swap"]) for r in rows}):
        arms = sorted(
            [r for r in rows if (r["group"], r["seed"], r["swap"]) == key],
            key=lambda r: r["revision"],
        )
        if len(arms) != 2:
            continue
        paths = [Path(r["source_run"]) for r in arms]
        initial = [np.load(d / "initial_state.npz")["state"] for d in paths]
        tapes = [np.load(d / "controls.npz") for d in paths]
        paired.append(
            dict(
                group=key[0],
                seed=key[1],
                swap=key[2],
                initial_state_identical=bool(np.array_equal(*initial)),
                controls_identical=bool(
                    np.array_equal(tapes[0]["ctrl"], tapes[1]["ctrl"])
                ),
                body_identical=bool(np.array_equal(tapes[0]["body"], tapes[1]["body"])),
                phases_identical=bool(
                    np.array_equal(tapes[0]["phases"], tapes[1]["phases"])
                ),
            )
        )
    groups = []
    for group, revision in sorted({(r["group"], r["revision"]) for r in rows}):
        batch = [r for r in rows if r["group"] == group and r["revision"] == revision]
        valid = [r for r in batch if r["physical_pass"]]
        interventions = [
            e
            for r in valid
            for e in r["strict_interventions"]
            if e["state"] == "TEAM_INTERVENTION"
        ]
        adoptions = [e for r in valid for e in r["adopted_assignments"]]
        delays = [e["time_s"] - e["plan_since_s"] for e in adoptions]
        groups.append(
            dict(
                group=group,
                revision=revision,
                scheduled=len(batch),
                numerical_pass=len(valid),
                results_all=dict(
                    Counter(
                        r["result"]["kind"] + ":" + str(r["result"]["winner"])
                        for r in batch
                    )
                ),
                all_down_events=sum(r["events"].get("DOWN", 0) for r in valid),
                clean_counter_count=sum(len(r["clean_counters"]) for r in valid),
                strict_intervention_count=len(interventions),
                strict_intervention_runs=sum(
                    any(
                        e["state"] == "TEAM_INTERVENTION"
                        for e in r["strict_interventions"]
                    )
                    for r in valid
                ),
                observed_relief_count=sum(
                    e["pressure_relief_observed"] for e in interventions
                ),
                angular_position_seconds=sum(
                    r["angular_position_seconds"] or 0 for r in valid
                ),
                assignment_adoptions=len(adoptions),
                adoption_delay_median_s=float(np.median(delays)) if delays else None,
                adoption_delay_p90_s=float(np.quantile(delays, 0.9))
                if delays
                else None,
                maximum_penetration_m=max(
                    r["physical"]["penetration_max_m"] for r in batch
                ),
            )
        )
    return dict(
        schema="duckfu.revision-evaluation.v1",
        actor_commit=actor_commit,
        source_report_sha256=digest(report),
        protocol=source["protocol"],
        groups=groups,
        pairs=paired,
        rows=rows,
        interpretation=[
            "Both revisions use the same pinned motors, referee and physics; initial-state equality is checked per pair.",
            "Sides, opponents, revisions and groups share four base seeds; scheduled games are not independent trials.",
            "Numerically rejected games remain in results_all; behavior counts are explicitly conditional on numerical validity.",
            "Strict target-change intervention v2 is unchanged; angular positions and proximity relief do not establish causal rescue.",
            "New seeds were run after the actor freeze. No 30% counter latency or 50% stagnation target is waived.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--actor-commit", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    result = summarize(a.report, a.actor_commit)
    write_json(out, result)
    with gzip.open(out.with_suffix(".raw-report.json.gz"), "wt") as stream:
        stream.write(Path(a.report).read_text())
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
