"""Compact frozen V2 evaluations, preserving raw reports and separate metrics."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest
from measure_combat_response import measure
from eval_combat_team_intents import analyze_team


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="artifacts/combat/local")
    p.add_argument("--out", default="artifacts/combat/v2/final-evaluation.json")
    a = p.parse_args()
    root = Path(a.root)
    dest = Path(a.out)
    if dest.exists():
        raise FileExistsError(dest)
    groups = {}
    raw_index = {}
    responses = []
    names = [
        "v2-validation-pressure-91001",
        "v2-validation-counter-91001",
        "v2-action-validation-94001",
        "v2-team-validation-92001",
        "v2-peel-validation-93001",
    ]
    for name in names:
        path = root / name / "report.json"
        data = json.loads(path.read_text())
        raw_index[name] = dict(path=str(path.resolve()), sha256=digest(path))
        with gzip.open(
            dest.parent / (name + "-report.json.gz"), "wt", compresslevel=9
        ) as stream:
            json.dump(data, stream)
        groups[name] = dict(
            scheduled=len(data["rows"]),
            physical_pass=sum(r["physical_pass"] for r in data["rows"]),
            penetration_max_m=max(
                r["physical"]["penetration_max_m"] for r in data["rows"]
            ),
            maximum_run_penetrating_p99_m=max(
                r["physical"]["penetrating_p99_m"] for r in data["rows"]
            ),
        )
        for row in data["rows"]:
            if "validation-pressure" in name or "validation-counter" in name:
                audit = measure(row["source_run"])
                audit.update(
                    version=row["version"],
                    opponent="pressure" if "pressure" in name else "counter",
                    seed=row["seed"],
                    swap=row["swap"],
                )
                responses.append(audit)
    duel = {}
    for opponent in ["pressure", "counter"]:
        matrix = {}
        data = json.loads(
            (root / f"v2-validation-{opponent}-91001/report.json").read_text()
        )
        for version in ["v1", "full_brake", "dynamic"]:
            rows = [r for r in data["rows"] if r["version"] == version]
            audits = [
                r
                for r in responses
                if r["version"] == version and r["opponent"] == opponent
            ]
            latencies = [
                c["latency_after_evade_s"]
                for r in audits
                for c in r["clean_evade_counters"]
                if c["latency_after_evade_s"] is not None
            ]
            matrix[version] = dict(
                games=len(rows),
                base_seeds=4,
                physical_pass=sum(r["physical_pass"] for r in rows),
                legacy_counter_sequences=sum(
                    r["event_counts"].get("COUNTER_SEQUENCE", 0) for r in rows
                ),
                observed_clean_counters=sum(
                    len(r["clean_evade_counters"]) for r in audits
                ),
                counter_response_latencies_s=latencies,
                conditional_median_latency_s=float(np.median(latencies))
                if latencies
                else None,
                down_events=sum(r["event_counts"].get("DOWN", 0) for r in rows),
                scheduled_results={
                    kind: sum(r["result"]["kind"] == kind for r in rows)
                    for kind in [
                        "WIN",
                        "TIMEOUT_DRAW",
                        "SIMULTANEOUS_DRAW",
                        "NO_CONTEST_PHYSICS",
                    ]
                },
            )
        duel[opponent] = matrix
    static = {}
    rows = json.loads((root / "v2-action-validation-94001/report.json").read_text())[
        "rows"
    ]
    for family in ["probe_bump", "committed_drive"]:
        sample = [r for r in rows if r["family"] == family]
        audits = [measure(r["source_run"]) for r in sample]
        attacks = [d for r in sample for d in r["attack_trajectories"]]
        static[family] = dict(
            games=len(sample),
            base_seeds=4,
            physical_pass=sum(r["physical_pass"] for r in sample),
            active_response_runs=sum(r["red_valid_responses"] > 0 for r in sample),
            response_audit_runs=sum(
                any(e["actor"] == "lavender" for e in r["observed_responses"])
                for r in audits
            ),
            median_complete_maneuver_s=float(
                np.median([d["duration_s"] for d in attacks])
            ),
            median_complete_maneuver_path_m=float(
                np.median([d["path_m"] for d in attacks])
            ),
            worst_attack_upright=min(d["minimum_upright"] for d in attacks),
            action_ids_are_not_the_diversity_metric=True,
        )
    teams = {}
    for label, name in [
        ("free", "v2-team-validation-92001"),
        ("pressure_course", "v2-peel-validation-93001"),
    ]:
        rows = json.loads((root / name / "report.json").read_text())["rows"]
        contact_only_count = sum(
            e["state"] == "TEAM_INTERVENTION" for r in rows for e in r["interventions"]
        )
        rows = [analyze_team(r["source_run"]) for r in rows]
        interventions = [
            e
            for r in rows
            for e in r["interventions"]
            if e["state"] == "TEAM_INTERVENTION"
        ]
        teams[label] = dict(
            games=len(rows),
            base_seeds=4,
            physical_pass=sum(r["physical_pass"] for r in rows),
            contact_only_intermediate_count=contact_only_count,
            intervention_schema="duckfu.team-intervention.v2",
            actual_interventions=len(interventions),
            runs_with_intervention=sum(
                any(e["state"] == "TEAM_INTERVENTION" for e in r["interventions"])
                for r in rows
            ),
            pressure_relief_observed=sum(
                e["pressure_relief_observed"] for e in interventions
            ),
            angular_position_seconds=sum(
                r["same_target_different_role_45_135_degree_seconds"] for r in rows
            ),
            interventions=interventions,
        )
    result = dict(
        schema="duckfu.v2-first-round-evaluation.v2",
        actor_commit="1e5f3a6",
        groups=groups,
        total_scheduled=sum(g["scheduled"] for g in groups.values()),
        total_numerical_pass=sum(g["physical_pass"] for g in groups.values()),
        distinct_base_seeds=16,
        duel=duel,
        standing_target=static,
        teams=teams,
        raw_reports=raw_index,
        limitations=[
            "Four base seeds per group; paired sides/opponents are correlated. No stable tactical superiority is established.",
            "The delayed-response audit was introduced from development diagnosis; it is a secondary observational metric, not a changed winner/reward.",
            "Conditional latency compares successful sequences and is not itself a paired causal improvement estimate.",
            "Team angular positions alone do not prove effective coordinated pressure.",
            "Human viewing acceptance and official ROSClaw authorization receipt remain pending.",
        ],
    )
    write_json(dest, result)
    write_json(
        dest.parent / "final-response-audits.json",
        dict(schema="duckfu.response-batch.v2", runs=responses),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
