"""Isolated MuJoCo worker invoked by the SIMULATION executor, never a shell."""

import json
import sys
from microduck_lab.arena.multiplayer import GameConfig, run_game

if __name__ == "__main__":
    request = json.load(sys.stdin)
    config = GameConfig(
        players=request["players"],
        grid=4 if request["players"] == 2 else 5,
        layout=request["layout"],
        cadence=request["cadence"],
    )
    audit = run_game(request["out"], request["seed"], config, capture=True)
    print(
        json.dumps(
            dict(
                outcome=audit["outcome"],
                steps=audit["steps"],
                duration=audit["duration"],
                penetration_max_m=audit["penetration_max_m"],
                finite=audit["finite"],
                solver_warnings=audit["solver_warnings"],
                body_contact_steps=audit["body_contact_steps"],
            )
        )
    )
