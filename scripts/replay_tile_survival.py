import argparse
import json
from microduck_lab.arena.verify import verify

p = argparse.ArgumentParser(description="Strict DG-02 input and semantic replay")
p.add_argument("--run", required=True)
print(json.dumps(verify(p.parse_args().run), indent=2))
