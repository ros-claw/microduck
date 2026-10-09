import argparse
import json
from pathlib import Path
from microduck_lab.arena.jev_rumble_verify import verify_jev

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out")
a = p.parse_args()
result = verify_jev(a.run)
if a.out:
    Path(a.out).write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
