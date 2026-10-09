"""Real online Jev play. Requires the private TYPESAFE_API_KEY configuration."""
import argparse
import json
from microduck_lab.arena.jev_fair_rumble import run_jev_fair
from microduck_lab.arena.survival_rumble import SurvivalConfig

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=61204)
p.add_argument("--duration", type=float, default=90.)
p.add_argument("--tilt", type=float, default=.08)
p.add_argument("--capture", action="store_true")
a = p.parse_args()
r = run_jev_fair(a.out, a.seed, SurvivalConfig(duration=a.duration, tilt_max_rad=a.tilt), a.capture)
print(json.dumps({k: r[k] for k in ("outcome", "duration", "jev_summary", "penetration_max_m", "final_observations", "performance")}, ensure_ascii=False, indent=2))
