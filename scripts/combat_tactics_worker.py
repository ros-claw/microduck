"""One frozen-CPU-physics tactical training rollout, isolated from Torch's env."""

import argparse
import mujoco
from microduck_lab.combat.config import Physics, strict_load
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.episode import run
from microduck_lab.combat.world import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--training", action="store_true")
    a = p.parse_args()
    if mujoco.__version__ != "3.12.0":
        raise RuntimeError(
            "Tactical rollouts require the locked CPU MuJoCo 3.12.0 runtime"
        )
    cfg = strict_load(a.config, MatchConfig)
    physics = strict_load(ROOT / "configs/combat" / cfg.physics_profile, Physics)
    run(a.out, cfg, physics, a.seed, training=a.training)


if __name__ == "__main__":
    main()
