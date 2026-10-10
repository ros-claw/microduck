"""Single-duck DG-01 greybox; deliberately not the future multiplayer CLI."""

import argparse
import json
from microduck_lab.arena.episode import run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--dt", type=float, default=0.0005)
p.add_argument("--duration", type=float, default=12)
p.add_argument("--out", default="artifacts/duckverse/seed11")
a = p.parse_args()
print(json.dumps(run(a.out, a.seed, a.dt, a.duration), indent=2))
