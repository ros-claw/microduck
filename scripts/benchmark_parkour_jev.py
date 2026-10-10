"""Matched, synthetic encounter fixtures for hot-path latency/choice comparison.

These are decision-contract cases, not live-game wins or unseen-seed evidence.
Every response and actual token usage is retained; the API key is never logged.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microduck_lab.parkour.tactics import TacticalClient

CASES = [
    dict(
        name="clear_corridor",
        expected="RUN",
        candidates=["RUN", "BRAKE_AND_WAIT"],
        state={
            "mission": {
                "goal": "escape",
                "lives": 3,
                "speed": "high",
                "risk_budget": "medium",
            },
            "encounter": "clear corridor",
            "measured_speed_m_s": 0.44,
            "feasibility": {
                "RUN": {"clear_for_seconds": 3.0},
                "BRAKE_AND_WAIT": {"safe": True},
            },
            "boss_ttc_seconds": 5.0,
        },
    ),
    dict(
        name="roll_or_wait",
        expected="ROLL_CENTER",
        candidates=["ROLL_CENTER", "BRAKE_AND_WAIT"],
        state={
            "mission": {
                "goal": "escape",
                "lives": 3,
                "speed": "high",
                "risk_budget": "medium",
            },
            "encounter": "low crossbar",
            "feasibility": {
                "ROLL_CENTER": {
                    "clear_over_entire_profile": True,
                    "duration_seconds": 1.32,
                },
                "BRAKE_AND_WAIT": {"safe_wait_seconds": 1.0},
            },
            "boss_ttc_seconds": 3.5,
        },
    ),
    dict(
        name="right_route",
        expected="TAKE_RIGHT_ROUTE",
        candidates=["TAKE_RIGHT_ROUTE", "BRAKE_AND_WAIT"],
        state={
            "mission": {
                "goal": "escape",
                "lives": 3,
                "speed": "high",
                "risk_budget": "medium",
            },
            "encounter": "telegraphed falling crate in center",
            "feasibility": {
                "TAKE_RIGHT_ROUTE": {
                    "clear_over_entire_profile": True,
                    "duration_seconds": 1.1,
                },
                "BRAKE_AND_WAIT": {"safe_wait_seconds": 1.5},
            },
            "boss_ttc_seconds": 4.0,
        },
    ),
]

if __name__ == "__main__":
    clients = {
        p: TacticalClient(profile=p, timeout=3.0)
        for p in ("choice", "choice_noul", "four")
    }
    results = []
    # Rotate profiles to avoid attributing warm-up or time-of-day drift to a profile.
    for repetition in range(10):
        for case in CASES:
            profiles = list(clients)
            profiles = profiles[repetition % 3 :] + profiles[: repetition % 3]
            for profile in profiles:
                record = dict(
                    profile=profile,
                    case=case["name"],
                    expected=case["expected"],
                    repetition=repetition,
                )
                try:
                    record["decision"] = clients[profile].decide(
                        case["state"], case["candidates"]
                    )
                except Exception as exc:
                    record["error"] = type(exc).__name__
                results.append(record)
        print("completed matched round", repetition + 1, flush=True)
    summary = {}
    for profile in clients:
        rows = [r for r in results if r["profile"] == profile]
        valid = [r for r in rows if "decision" in r]
        lat = [r["decision"]["latency_ms"] for r in valid]
        summary[profile] = dict(
            requests=len(rows),
            valid=len(valid),
            p50_ms=float(np.median(lat)) if lat else None,
            p95_ms=float(np.quantile(lat, 0.95)) if lat else None,
            expected_choice_matches=sum(
                r["decision"]["action"] == r["expected"] for r in valid
            ),
        )
    out = dict(
        scope="Synthetic encounter contract and latency benchmark, not gameplay success.",
        cases=CASES,
        summary=summary,
        results=results,
    )
    (ROOT / "artifacts/neon-escape-v2/jev-hot-path.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)
