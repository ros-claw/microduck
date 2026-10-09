#!/usr/bin/env python3
"""Strict closed-loop replay of source/model/policy-bound full-rate evidence."""

import argparse
import json
from pathlib import Path
from microduck_lab.arena.island_rumble_verify import verify_rumble

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
r = verify_rumble(a.run)
Path(a.out).write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
