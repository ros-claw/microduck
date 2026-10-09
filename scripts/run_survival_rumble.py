#!/usr/bin/env python3
import argparse
import json
from microduck_lab.arena.survival_rumble import SurvivalConfig, run_survival

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=61001)
p.add_argument("--capture", action="store_true")
p.add_argument("--passive", action="store_true")
p.add_argument("--tilt", type=float, default=0.14)
p.add_argument("--duration", type=float, default=85)
a = p.parse_args()
r = run_survival(
    a.out,
    a.seed,
    SurvivalConfig(tilt_max_rad=a.tilt, duration=a.duration, passive=a.passive),
    a.capture,
)
print(
    json.dumps(
        {
            k: r[k]
            for k in (
                "outcome",
                "duration",
                "final_alive",
                "penetration_max_m",
                "arena_motor_force_max_Nm",
                "stationary_fraction",
                "finale_started_s",
            )
        },
        indent=2,
    )
)
