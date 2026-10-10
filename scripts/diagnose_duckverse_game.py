#!/usr/bin/env python3
"""Quantify old-game event timing and stationary time from published decision logs."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
rows = []
for seed in (101, 20008, 20017):
    path = Path(a.source) / str(seed)
    audit = json.loads((path / "audit.json").read_text())
    with gzip.open(path / "decisions.jsonl.gz", "rt") as f:
        decisions = [json.loads(s) for s in f]
    counts = Counter((int(e["time"]), e["state"]) for e in audit["events"])
    rows.append(
        dict(
            seed=seed,
            layout=audit["config"]["layout"],
            cadence=audit["config"]["cadence"],
            duration=audit["duration"],
            first_elimination_s=min(
                (e["time"] for e in audit["events"] if e["state"] == "ELIMINATED"),
                default=None,
            ),
            stationary_speed_below_003=sum(
                d["input"]["robot"]["speed_m_s"] < 0.03 for d in decisions
            )
            / len(decisions),
            stand_policy_fraction=sum(d["policy"] == "stand" for d in decisions)
            / len(decisions),
            events_by_second=[
                dict(second=i, events={k: v for (s, k), v in counts.items() if s == i})
                for i in range(int(audit["duration"]) + 1)
            ],
        )
    )
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "baseline-diagnosis.json").write_text(json.dumps(rows, indent=2) + "\n")
im = Image.new("RGB", (1280, 610), "white")
d = ImageDraw.Draw(im)
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
d.text(
    (22, 12),
    "V1 audit: event timing and measured stationary fraction",
    font=b,
    fill="black",
)
colours = {
    "WARNING": "#de9d28",
    "RELEASED": "#b84b48",
    "BODY_CONTACT": "#6084b2",
    "ELIMINATED": "#1b1b1b",
}
for j, r in enumerate(rows):
    y = 95 + j * 160
    d.text(
        (22, y - 32),
        f"seed {r['seed']} | {r['layout']}/{r['cadence']} | stationary {r['stationary_speed_below_003']:.1%} | first loss {r['first_elimination_s']:.2f}s",
        font=f,
        fill="black",
    )
    d.line((60, y + 85, 1210, y + 85), fill="#aaaaaa")
    for cell in r["events_by_second"]:
        x = 60 + cell["second"] * 28
        bottom = y + 85
        for key, colour in colours.items():
            value = cell["events"].get(key, 0)
            if value:
                d.rectangle((x, bottom - value * 15, x + 23, bottom), fill=colour)
                bottom -= value * 15
    for t in range(0, 41, 5):
        d.text((60 + t * 28, y + 90), str(t) + "s", font=f, fill="black")
for j, (key, col) in enumerate(colours.items()):
    x = 22 + j * 310
    d.rectangle((x, 575, x + 16, 591), fill=col)
    d.text((x + 23, 570), key, font=f, fill="black")
im.save(out / "baseline-event-density.png")
