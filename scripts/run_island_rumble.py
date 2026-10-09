#!/usr/bin/env python3
"""Separate contact-driven prototype; never replaces archived Duckverse rules."""

import argparse
import json
from microduck_lab.arena.island_rumble import RumbleConfig, run_rumble

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=31001)
p.add_argument("--players", type=int, choices=(2, 4), default=2)
p.add_argument("--grid", type=int, choices=(3, 4), default=3)
p.add_argument("--tile-size", type=float, default=0.36)
p.add_argument("--lifetime", type=float, default=3.2)
p.add_argument("--final-at", type=float, default=6.0)
p.add_argument("--duration", type=float, default=20.0)
p.add_argument("--passive", action="store_true")
p.add_argument("--capture", action="store_true")
p.add_argument("--claim-radius", type=float, default=0.0)
a = p.parse_args()
r = run_rumble(
    a.out,
    a.seed,
    RumbleConfig(
        players=a.players,
        grid=a.grid,
        tile_size=a.tile_size,
        lifetime_s=a.lifetime,
        final_at_s=a.final_at,
        duration=a.duration,
        passive=a.passive,
        claim_radius_m=a.claim_radius,
    ),
    a.capture,
)
print(
    json.dumps(
        {
            k: r[k]
            for k in (
                "seed",
                "outcome",
                "duration",
                "body_contact_steps",
                "penetration_max_m",
                "stationary_fraction",
                "no_safe_option_fraction",
            )
        },
        indent=2,
    )
)
