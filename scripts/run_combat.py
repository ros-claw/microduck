"""Run the development duel; learned/Jev/team modes are not yet qualified."""

import argparse
import json
from microduck_lab.combat.config import load
from microduck_lab.combat.episode import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--brain", choices=["rule", "learned", "jev"], default="rule")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.brain != "rule":
        p.error(f"{a.brain} combat is not implemented/qualified; no fallback")
    config, physics = load(a.config)
    result = run(a.out, config, physics, a.seed, a.capture)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("steps", "duration_s", "physical", "performance", "result")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
