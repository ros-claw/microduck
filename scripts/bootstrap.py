"""Bootstrap: fetch pinned upstream assets and verify integrity.

Assets come from the official repos (never vendored here — the 3D models are
CC BY-SA-NC). Set MICRODUCK_ROOT to a directory containing local clones of
`microduck_rl` and `microduck` to skip downloading.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "upstream.lock.yaml"


def main():
    root = pathlib.Path(os.environ.get("MICRODUCK_ROOT", ROOT.parent)).expanduser()
    need = {
        "microduck_rl": "https://github.com/pollen-robotics/microduck_rl",
        "microduck": "https://github.com/pollen-robotics/microduck",
    }
    import re
    lock = LOCK.read_text()
    for name, url in need.items():
        dest = root / name
        m = re.search(rf"{name}:.*?commit: ([0-9a-f]{{40}})", lock, re.S)
        sha = m.group(1)
        if not dest.exists():
            print(f"cloning {url} → {dest}")
            subprocess.run(["git", "clone", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "origin", sha],
                       check=False, capture_output=True)
        subprocess.run(["git", "-C", str(dest), "checkout", sha], check=True)
        print(f"{name} @ {sha[:8]} ✓")
    print("bootstrap OK")


if __name__ == "__main__":
    main()
