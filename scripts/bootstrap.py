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
        "microduck_rl": ("microduck_rl", "https://github.com/pollen-robotics/microduck_rl"),
        "microduck": ("microduck_runtime", "https://github.com/pollen-robotics/microduck"),
    }
    import re
    lock = LOCK.read_text()
    root.mkdir(parents=True, exist_ok=True)
    for name, (lock_key, url) in need.items():
        dest = root / name
        m = re.search(rf"^{lock_key}:.*?commit: ([0-9a-f]{{40}})", lock, re.S | re.M)
        if m is None:
            raise ValueError(f"Missing pinned commit for {lock_key}")
        sha = m.group(1)
        if dest.exists():
            # Existing training checkouts may contain valuable local work.
            head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
            print(f"{name}: keeping existing checkout @ {head[:8]} (pin {sha[:8]})")
            continue
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
