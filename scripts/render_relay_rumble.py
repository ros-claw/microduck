#!/usr/bin/env python3
import argparse
import json
from microduck_lab.presentation.relay_rumble import render_relay

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--preview", type=float)
a = p.parse_args()
print(json.dumps(render_relay(a.run, a.out, a.preview), ensure_ascii=False))
