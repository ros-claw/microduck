"""Run an explicit simulation duel or shared-world competition; unsupported modes fail."""

import argparse
import json
from pathlib import Path
import yaml
from microduck_lab.combat.config import load, strict_load, Physics
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.episode import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--brain", choices=["rule", "learned", "jev"], default="rule")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        content = yaml.safe_load(Path(a.config).read_text())
    except (OSError, yaml.YAMLError) as error:
        p.error(f"Cannot load the explicit configuration: {error}; no fallback")
    competition = isinstance(content, dict) and content.get("schema") in (
        "duckfu.match.v1",
        "duckfu.match.v2",
    )
    if a.brain == "jev" or (a.brain == "learned" and not competition):
        p.error(f"{a.brain} combat is not implemented/qualified; no fallback")
    if competition:
        config = strict_load(
            a.config,
            V2Config if content["schema"] == "duckfu.match.v2" else MatchConfig,
        )
        physics = strict_load(Path(a.config).parent / config.physics_profile, Physics)
        if (a.brain == "learned") != bool(config.tactics_path):
            p.error("Brain must match the explicit tactical policy in the config")
    else:
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
