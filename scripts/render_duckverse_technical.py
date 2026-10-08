#!/usr/bin/env python3
"""Three clearly labelled matches and detail replays from immutable captures."""

import argparse
import json
import gzip
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from microduck_lab.presentation.duckverse import Film, FONT, BOLD, INK
from microduck_lab.arena.episode import sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs", nargs=3, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
records = []
sources = []
fps = 50


def card(writer, title, lines, seconds):
    im = Image.new("RGB", (1280, 720), INK)
    d = ImageDraw.Draw(im)
    d.rectangle((48, 100, 58, 610), fill=(109, 216, 220))
    d.text(
        (88, 102),
        "ROSCLAW / DUCKVERSE",
        font=ImageFont.truetype(FONT, 22),
        fill=(109, 216, 220),
    )
    d.text((88, 159), title, font=ImageFont.truetype(BOLD, 38), fill="white")
    for j, line in enumerate(lines):
        d.text(
            (88, 255 + j * 55),
            line,
            font=ImageFont.truetype(FONT, 25),
            fill=(208, 223, 235),
        )
    for _ in range(round(seconds * fps)):
        writer.append_data(np.asarray(im))
        records.append(
            {
                "frame": len(records),
                "kind": "method_card",
                "title": title,
                "physics_edited": False,
            }
        )


with imageio.get_writer(
    out, fps=fps, codec="libx264", quality=8, macro_block_size=1
) as writer:
    card(
        writer,
        "Last Duck Standing / methods & evidence",
        [
            "Four native Microducks share one MuJoCo world.",
            "Current warnings guide independent tactical controllers.",
            "Existing learned stand/walk policies drive 14 torque-limited joints.",
            "Gravity, contact support and upright posture determine the result.",
            "SIMULATION ONLY / no vision, new training or hardware deployment.",
        ],
        8,
    )
    for run_number, path in enumerate(a.runs):
        f = Film(path)
        sources.append(
            {
                "seed": f.audit["seed"],
                "audit_sha256": sha(Path(path) / "audit.json"),
                "hashes": f.audit["hashes"],
                "outcome": f.audit["outcome"],
            }
        )
        try:

            def clip(start, end, speed, caption, focus=None, overlay=False):
                base = f.shot
                if focus:
                    f.shot = lambda t, mode: focus
                try:
                    t = start
                    while t < min(end, f.audit["duration"]):
                        im = f.frame(t, speed, "reference", caption, overlay)
                        writer.append_data(np.asarray(im))
                        records.append(
                            {
                                "frame": len(records),
                                "run": run_number,
                                "seed": f.audit["seed"],
                                **{
                                    k: v
                                    for k, v in f.records[-1].items()
                                    if k != "frame"
                                },
                                "caption": caption,
                            }
                        )
                        t += speed / fps
                finally:
                    f.shot = base

            label = f"MATCH {run_number + 1} / seed {f.audit['seed']} / {f.audit['config']['layout'].upper()}"
            clip(0, f.audit["duration"], 1, label + " / continuous reference")
            if run_number == 0:
                clip(
                    3.1,
                    5.4,
                    0.5,
                    "DETAIL REPLAY / warning response / cream crosses a real seam",
                    ("follow", "cream"),
                )
                clip(
                    16.6,
                    18.5,
                    0.25,
                    "DETAIL REPLAY / torque-limited learned walking / lavender",
                    ("follow", "lavender"),
                )
                clip(
                    3.1,
                    5.2,
                    0.5,
                    "POV REPLAY / camera follows cream's recorded head pose",
                    ("duck_eye", "cream"),
                )
                tc = f.first_contact
                if tc is not None:
                    clip(
                        tc - 0.2,
                        tc + 0.6,
                        0.125,
                        "DETAIL REPLAY / actual duck-duck contact / 0.125x",
                        ("contact", None),
                    )
                    for g in range(f.m.ngeom):
                        if "external_envelope" in (f.m.geom(g).name or ""):
                            f.m.geom_rgba[g] = [0.85, 0.18, 0.75, 0.25]
                    clip(
                        tc - 0.2,
                        tc + 0.6,
                        0.25,
                        "GEOMETRY REPLAY / conservative head/torso boxes + native contacts",
                        ("contact", None),
                        True,
                    )
                first = f.losses[0]["time"]
                clip(
                    first - 0.8,
                    first + 0.2,
                    0.25,
                    "DETAIL REPLAY / released support > physical fall > elimination",
                    ("fall", None),
                )
                card(
                    writer,
                    "What the film does and does not prove",
                    [
                        "4 kHz soft-contact simulation; motor updates at 50 Hz.",
                        "Full-rate state, control, contact and public-decision logs.",
                        "Strict replay regenerates decisions, contacts and referee output.",
                        "Head/torso use conservative visual-bounds collision boxes.",
                        "This is simulator consistency, not independent hardware validation.",
                    ],
                    8,
                )
        finally:
            f.close()
    card(
        writer,
        "A baseline, with reproducible limits",
        [
            "Four layouts and two release cadences; all outcomes remain in the report.",
            "Draws are genuine: a final warning may leave no legal safe neighbour.",
            "Current-state heuristics are not a trained survival policy.",
            "Native ROSClaw launches the game; it does not steer every motor step.",
            "Code, qualification results and evidence: github.com/ros-claw/microduck",
        ],
        8,
    )
mapfile = out.with_suffix(".frames.jsonl.gz")
with gzip.open(mapfile, "wt") as fp:
    for r in records:
        fp.write(json.dumps(r) + "\n")
manifest = {
    "schema": "microduck.duckverse.technical-film.v1",
    "video": out.name,
    "video_sha256": sha(out),
    "fps": fps,
    "duration_s": len(records) / fps,
    "runs": sources,
    "frame_map": mapfile.name,
    "frame_map_sha256": sha(mapfile),
    "physics_edited": False,
    "audio": "none",
    "evidence_domain": "simulation",
    "editing": "Three labelled continuous matches followed by explicitly labelled detail replays; no pose interpolation",
}
out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
