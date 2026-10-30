import argparse
import json
from pathlib import Path
from microduck_lab.combat.replay import verify

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Closed-loop Duck Fu replay, not state playback"
    )
    p.add_argument("--run", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    result = verify(a.run)
    if a.out:
        path = Path(a.out)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
