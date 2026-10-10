#!/usr/bin/env python3
"""Package selected full-rate evidence and small metadata without credentials."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", default="out/island-rumble-release")
a = p.parse_args()
root = Path(".").resolve()
dest = Path(a.out)
dest.mkdir(parents=True, exist_ok=True)
entries = []
for label, paths in [
    (
        "island-rumble-evidence",
        [
            root / "artifacts/island-rumble/selected-four",
            root / "artifacts/island-rumble/selected-two-current",
        ],
    ),
    ("island-rumble-metadata", [root / "artifacts/island-rumble"]),
]:
    target = dest / (label + ".tar.gz")
    with tarfile.open(target, "w:gz", compresslevel=3) as tar:
        for path in paths:
            for file in sorted(path.rglob("*")):
                if not file.is_file():
                    continue
                if label.endswith("metadata") and (
                    file.suffix in (".mjb", ".npz", ".npy")
                    or file.name.endswith(".jsonl.gz")
                    or file.name in ("release-bundles.json", "release-provenance.json")
                ):
                    continue
                info = tar.gettarinfo(str(file), str(file.relative_to(root)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with file.open("rb") as f:
                    tar.addfile(info, f)
    h = hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    entries.append(
        dict(file=target.name, bytes=target.stat().st_size, sha256=h.hexdigest())
    )
    print(json.dumps(entries[-1]), flush=True)
(root / "artifacts/island-rumble/release-bundles.json").write_text(
    json.dumps(entries, indent=2) + "\n"
)
