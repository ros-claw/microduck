#!/usr/bin/env python3
import argparse
import json
from microduck_lab.presentation.island_rumble import render

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument(
    "--mode",
    choices=("prototype", "hero", "vertical", "reference"),
    default="prototype",
)
p.add_argument("--preview", type=float)
a = p.parse_args()
print(json.dumps(render(a.run, a.out, a.mode, a.preview), indent=2))
