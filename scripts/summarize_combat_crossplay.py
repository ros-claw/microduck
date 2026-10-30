"""Paired-seed uncertainty; do not treat two arena sides as independent seeds."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.world import digest


def summarize(path):
    path = Path(path)
    report = json.loads(path.read_text())
    if report["schema"] != "duckfu.crossplay-report.v1":
        raise ValueError("Expected a complete frozen cross-play report")
    groups = defaultdict(list)
    for row in report["rows"]:
        groups[(row["candidate"], row["opponent"])].append(row)
    result = {}
    pairs = {}
    rng = np.random.default_rng(1030)
    for (candidate, opponent), rows in groups.items():
        seeds = sorted({r["base_seed"] for r in rows})
        values = []
        for seed in seeds:
            pair = [r for r in rows if r["base_seed"] == seed]
            if len(pair) != 2 or {r["side"] for r in pair} != {0, 1}:
                raise ValueError("Missing or duplicated paired side")

            def win(r):
                return float(
                    r["result"]["kind"] == "WIN" and r["result"]["winner"] == "RED"
                )

            values.append(sum(win(r) for r in pair) / 2)
        values = np.asarray(values)
        samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
        key = candidate + " vs " + opponent
        result[key] = {
            **report["matrix"][key],
            "red_win_rate_all_scheduled_games": float(values.mean()),
            "paired_seed_bootstrap95_win_rate": np.quantile(
                samples, [0.025, 0.975]
            ).tolist(),
            "independent_base_seeds": len(seeds),
            "numerical_rejections_are_not_dropped": True,
        }
        pairs[(candidate, opponent)] = {
            seed: value for seed, value in zip(seeds, values)
        }
    comparisons = {}
    for candidate in sorted({k[0] for k in groups}):
        if candidate == "rule":
            continue
        opponents = sorted(
            {k[1] for k in groups if k[0] == candidate and ("rule", k[1]) in pairs}
        )
        if not opponents:
            continue
        seeds = sorted(
            set.intersection(
                *(
                    set(pairs[(candidate, o)]) & set(pairs[("rule", o)])
                    for o in opponents
                )
            )
        )
        values = np.array(
            [
                np.mean(
                    [
                        pairs[(candidate, o)][s] - pairs[("rule", o)][s]
                        for o in opponents
                    ]
                )
                for s in seeds
            ]
        )
        sample = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
        comparisons[candidate + " minus rule"] = dict(
            mean_win_rate_difference=float(values.mean()),
            paired_seed_bootstrap95=np.quantile(sample, [0.025, 0.975]).tolist(),
            base_seeds=len(seeds),
            opponents=opponents,
            note="Same base seeds and motor/physics package; no-contests remain in scheduled-game denominators. This is a win-rate comparison, not evidence that every maneuver improved.",
        )
    return dict(
        schema="duckfu.paired-crossplay-summary.v1",
        source_report_sha256=digest(path),
        split=report["split"],
        bootstrap_resamples=10000,
        bootstrap_seed=1030,
        matrix=result,
        paired_comparisons=comparisons,
        film_gate="PENDING_HUMAN_REVIEW",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = summarize(a.report)
    write_json(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
