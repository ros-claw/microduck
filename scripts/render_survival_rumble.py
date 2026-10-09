#!/usr/bin/env python3
import argparse
import json
from microduck_lab.presentation.survival_rumble import render_survival

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--preview", type=float)
p.add_argument("--speech")
p.add_argument("--plan-only", action="store_true")
a = p.parse_args()
print(
    json.dumps(
        render_survival(a.run, a.out, a.preview, a.speech, a.plan_only),
        ensure_ascii=False,
    )
)
