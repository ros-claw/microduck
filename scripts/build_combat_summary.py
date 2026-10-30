"""Assemble compact evidence from immutable reports and final film checks."""

import argparse
import gzip
import json
from pathlib import Path
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import ROOT, digest


def read(path):
    path = Path(path)
    return json.loads(
        gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_text()
    )


def build():
    directory = ROOT / "artifacts/combat"
    motor = read(directory / "complete-motor-validation.json.gz")
    games = []
    reports = {}
    for name in (
        "duel-final-holdout-40001",
        "team-final-holdout-50001",
        "team-rule-final-holdout-50001",
    ):
        file = directory / (name + "-attribution-v2.json.gz")
        data = read(file)
        games.extend(data["rows"])
        reports[name] = dict(
            path=str(file.relative_to(ROOT)),
            sha256=digest(file),
            games=len(data["rows"]),
            matrix=data["matrix"],
            attribution=data["derived_event_attribution"],
        )
    rejected = [r for r in games if not r["physical_pass"]]
    recoveries = {}
    for group, seed in [("R0", 28001), ("R3", 28101), ("R1", 29001), ("R2", 29201)]:
        file = directory / f"recovery-v5-{group}-{seed}.json.gz"
        data = read(file)
        recoveries[group] = {k: v for k, v in data.items() if k != "rows"}
        recoveries[group].update(
            report=str(file.relative_to(ROOT)), sha256=digest(file)
        )
    films = {}
    for name in (
        "team-hero",
        "team-short",
        "team-technical",
        "team-reference",
        "duel",
        "numerical-rejection",
    ):
        file = directory / (name + "-film-check.json")
        data = read(file)
        if not data["passed"]:
            raise ValueError("Failed final film audit")
        films[name] = data
    paired = {
        name: read(directory / (name + "-paired-summary.json"))
        for name in ("duel", "team")
    }
    return dict(
        schema="duckfu.complete-summary.v1",
        evidence_domain="simulation",
        motor=dict(
            paired_initial_states=motor["paired_initial_states"],
            runs=len(motor["rows"]),
            summaries=motor["summaries"],
        ),
        recovery=recoveries,
        recovery_profile_provenance="artifacts/combat/recovery-profile-provenance.json",
        games=dict(
            scheduled=len(games),
            numerical_pass=sum(r["physical_pass"] for r in games),
            no_contest=len(rejected),
            independent_seed_groups=dict(duel=32, team=32),
            paired_sides=True,
            reports=reports,
            maximum_penetration_m=max(
                r["physical"]["penetration_max_m"] for r in games
            ),
            maximum_per_game_penetrating_p99_m=max(
                r["physical"]["penetrating_p99_m"] for r in games
            ),
            maximum_torque_Nm=max(r["physical"]["torque_max_Nm"] for r in games),
            rejections=[
                dict(
                    run=r["run"],
                    seed=r["base_seed"],
                    side=r["side"],
                    result=r["result"],
                    physical=r["physical"],
                )
                for r in rejected
            ],
        ),
        tactical_comparisons=paired,
        training=dict(
            source_commit="1bb01ff9c1780c13c753f660894f30cf96c30a18",
            solo_online_ppo_updates=12,
            solo_online_games=192,
            teacher_initialization="explicit rule-trajectory distillation; see solo-distillation-audit.json",
            team_final_online_ppo_updates=8,
            team_final_online_games=64,
            final_team_policy_sha256=digest(ROOT / "policies/tactics/team_008.npz"),
            checkpoints="training/combat/selected-checkpoints.json",
            tactical_promotion="NOT_PROMOTED: no stable PPO advantage over the frozen rule baseline",
        ),
        selected_matches={
            name: dict(
                seed=(m := read(directory / f"final-{name}-manifest.json"))["seed"],
                result=m["result"],
                duration_s=m["duration_s"],
                physical=m["physical"],
                source_sha256=m["source_sha256"],
                replay=read(directory / f"final-{name}-replay.json"),
            )
            for name in ("duel", "team")
        },
        films=films,
        verification=read(directory / "verification.json"),
        source_lock=dict(
            path="configs/combat/complete.source.lock.yaml",
            sha256=digest(ROOT / "configs/combat/complete.source.lock.yaml"),
        ),
        gates=dict(
            motor="passed frozen attack/evasion battery; counter contact is descriptive",
            physics="831/832 games pass; the remaining numerical fault is explicitly rejected",
            tactics="candidate; PPO not promoted",
            rosclaw="typed SIM executor only; official daemon-authorized combat receipt pending",
            human_viewing="PENDING_HUMAN_REVIEW",
        ),
        limitations=[
            "Component-convex geometry and native self/ground shapes approximate the robot.",
            "Nominal R1 recovery missed the early 80% within3s target; all attempts retained.",
            "Full closed-loop winner is timestep-sensitive: duel32001 changed from8kHz draw to16kHz BLUEwin.",
            "Post-match attribution v2 separates interruptions from completed misses; raw reports remain available and physics parity was verified.",
            "No hardware transfer, network-LLM reflex loop or autonomous evolution is claimed.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/combat/complete-summary.json")
    args = parser.parse_args()
    result = build()
    write_json(args.out, result)
    print(
        json.dumps(
            dict(
                games=result["games"]["scheduled"],
                physical_pass=result["games"]["numerical_pass"],
                film_editions=len(result["films"]),
                human_viewing=result["gates"]["human_viewing"],
            )
        )
    )


if __name__ == "__main__":
    main()
