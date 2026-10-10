#!/usr/bin/env python3
"""Run a physical shared-world elimination match, optionally recording every step."""

import argparse
import json
from microduck_lab.arena.multiplayer import GameConfig, run_game

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=101)
p.add_argument("--players", type=int, choices=(2, 4), default=4)
p.add_argument(
    "--layout", choices=("square", "ring", "cross", "terraces"), default="square"
)
p.add_argument("--cadence", choices=("steady", "rapid"), default="steady")
p.add_argument("--duration", type=float, default=45.0)
p.add_argument("--dt", type=float, default=0.00025)
p.add_argument("--capture", action="store_true")
a = p.parse_args()
result = run_game(
    a.out,
    a.seed,
    GameConfig(
        players=a.players,
        grid=4 if a.players == 2 else 5,
        layout=a.layout,
        cadence=a.cadence,
        duration=a.duration,
        dt=a.dt,
    ),
    a.capture,
)
print(
    json.dumps(
        {
            k: result[k]
            for k in (
                "seed",
                "outcome",
                "duration",
                "penetration_max_m",
                "body_contact_steps",
                "performance",
            )
        },
        indent=2,
    )
)
