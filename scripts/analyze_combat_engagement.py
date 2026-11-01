"""Read recorded matches without rerunning physics; save diagnoses and timelines."""

import argparse
import json
from pathlib import Path
from microduck_lab.combat.engagement_metrics import analyze, timeline_svg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, action="append")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    for i, directory in enumerate(a.run):
        report = analyze(directory)
        stem = f"{i:02d}-{report['seed']}"
        (out / (stem + ".json")).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (out / (stem + ".svg")).write_text(timeline_svg(report))
        summary.append(
            dict(
                file=stem + ".json",
                source=report["source_run"],
                seed=report["seed"],
                result=report["result"],
                duration_s=report["duration_s"],
                reward_components=report["reward_components"],
                actors={
                    n: {
                        k: v for k, v in b.items() if k not in ("phases", "transitions")
                    }
                    for n, b in report["actors"].items()
                },
            )
        )
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                schema="duckfu.diagnosis-batch.v1",
                selection="Purposive development diagnosis; not a new holdout.",
                runs=summary,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(dict(runs=len(summary), out=str(out))))


if __name__ == "__main__":
    main()
