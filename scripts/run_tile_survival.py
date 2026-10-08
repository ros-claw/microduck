"""DG-02 single-duck schedule and contact referee experiment."""

import argparse
import json
from microduck_lab.arena.survival import run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--duration", type=float, default=22.0)
p.add_argument("--brain", choices=["warning", "hold"], default="warning")
p.add_argument("--mode", choices=["seeded", "drop_test"], default="seeded")
a = p.parse_args()
r = run(a.out, a.seed, a.duration, a.mode, a.brain)
print(
    json.dumps(
        {k: v for k, v in r.items() if k not in ["decisions", "source_sha256"]},
        indent=2,
    )
)
