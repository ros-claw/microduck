#!/usr/bin/env python3
import argparse
import json
from microduck_lab.presentation.jev_rumble import render_jev

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--preview", type=float)
p.add_argument("--speech")
p.add_argument("--plan-only", action="store_true")
p.add_argument("--language", choices=("zh-CN", "en"), default="zh-CN")
a = p.parse_args()
result = render_jev(a.run, a.out, a.preview, a.speech, a.plan_only, a.language)
# The complete text audit is stored in the manifest, not echoed to the terminal.
result.pop("rendered_text", None)
print(json.dumps(result, ensure_ascii=False))
