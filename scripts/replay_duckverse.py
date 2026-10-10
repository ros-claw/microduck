#!/usr/bin/env python3
import argparse
import json
from microduck_lab.arena.multiplayer_verify import verify_game

p = argparse.ArgumentParser(description="Strictly regenerate a Duckverse match")
p.add_argument("capture")
a = p.parse_args()
print(json.dumps(verify_game(a.capture), indent=2))
