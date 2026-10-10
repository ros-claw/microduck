#!/usr/bin/env python3
import argparse
import json
from microduck_lab.arena.relay_rumble_verify import verify_relay

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
print(json.dumps(verify_relay(a.run), indent=2))
