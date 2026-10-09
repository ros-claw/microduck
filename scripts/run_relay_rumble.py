#!/usr/bin/env python3
"""Unpublished moving-objective experiment; never creates releases/tags."""

import argparse
import json
from microduck_lab.arena.relay_rumble import RelayConfig, run_relay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=51001)
p.add_argument("--capture", action="store_true")
p.add_argument("--passive", action="store_true")
p.add_argument("--duration", type=float, default=65)
p.add_argument("--tile-size", type=float, default=0.40)
p.add_argument("--lifetime", type=float, default=8)
a = p.parse_args()
r = run_relay(
    a.out,
    a.seed,
    RelayConfig(
        duration=a.duration,
        tile_size=a.tile_size,
        lifetime_s=a.lifetime,
        passive=a.passive,
    ),
    a.capture,
)
print(
    json.dumps(
        {
            k: r[k]
            for k in (
                "outcome",
                "duration",
                "body_contact_steps",
                "travelled_m",
                "points",
                "penetration_max_m",
                "stationary_fraction",
            )
        },
        indent=2,
    )
)
