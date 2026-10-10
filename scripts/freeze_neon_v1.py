"""Checksum the released V1 implementation and evidence without modifying them."""

import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "a1219f309bbfd4f2edb6ffb3b8ecba0cd3eb8d64"
paths = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True
).splitlines()
selected = [
    p
    for p in paths
    if p.startswith(
        ("src/microduck_lab/game/", "src/microduck_lab/jev/", "artifacts/neon-escape/")
    )
    or p
    in [
        "src/microduck_lab/demos/neon_run.py",
        "scripts/replay_neon_run.py",
        "scripts/render_neon_escape.py",
        "out/microduck_neon_escape_en.json",
    ]
]
manifest = {"commit": BASE, "release": "neon-escape-2026-09-26", "sha256": {}}
for p in selected:
    blob = subprocess.check_output(["git", "show", BASE + ":" + p], cwd=ROOT)
    manifest["sha256"][p] = hashlib.sha256(blob).hexdigest()
    if (ROOT / p).read_bytes() != blob:
        raise RuntimeError("V1 was modified: " + p)
video = ROOT / "out/microduck_neon_escape_en.mp4"
manifest["video_sha256"] = (
    hashlib.sha256(video.read_bytes()).hexdigest()
    if video.exists()
    else "download release to verify"
)
(ROOT / "artifacts/neon-escape-v2/v1-freeze.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("V1 unchanged:", len(selected), "files")
