#!/usr/bin/env python3
"""Present one source-bound physical match with read-only cameras."""

import argparse
from microduck_lab.presentation.duckverse import render

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--mode", choices=("hero", "vertical", "reference"), default="hero")
p.add_argument("--preview", type=float)
a = p.parse_args()
render(a.run, a.out, a.mode, a.preview)
