"""Compact V2 course blueprint. The gap remains locked until skill certification."""

from dataclasses import dataclass
import numpy as np
from .hazards import Hazard
from .world import build_world


@dataclass(frozen=True)
class Level:
    seed: int
    hazards: tuple
    gaps: tuple
    finish_x: float


def generate(
    seed=0,
    sweeper_phase=0.0,
    sweeper_mass=0.20,
    sweeper_y=0.0,
    sweeper_height=0.16,
    sweeper_speed=-3.0,
    sweeper_torque=0.10,
    difficulty="classic",
):
    if difficulty not in ("classic", "chase", "arcade"):
        raise ValueError("Unknown difficulty")
    rng = np.random.default_rng(seed)
    crate_y = float(rng.choice([-0.12, 0.12])) if difficulty != "classic" else 0.0
    jitter = lambda: float(rng.uniform(-0.025, 0.025))
    hazards = (
        Hazard("bar", "push_bar", 0.9 + jitter(), z=0.27),
        Hazard("crate", "crate", 1.85 + jitter(), y=crate_y, z=0.65),
        Hazard(
            "sweeper",
            "sweeper",
            4.0 + jitter(),
            y=sweeper_y,
            z=sweeper_height,
            speed=sweeper_speed,
            rotor_mass=sweeper_mass,
            phase=sweeper_phase,
            drive_torque=sweeper_torque,
        ),
        Hazard(
            "boss",
            "boulder",
            -4.2 if difficulty == "arcade" else -2.4 if difficulty == "chase" else -2.7,
            z=0.8,
            speed=0.44 if difficulty != "classic" else 0.40,
            centering_force=0.35 if difficulty != "classic" else 0.0,
        ),
        Hazard("portal", "finish_gate", 5.5, z=0.905),
    )
    if difficulty == "arcade":
        hazards = hazards[:-1] + (
            Hazard("playball", "bowling_ball", 5.5, z=0.085),
            Hazard("gutter_left", "guide", 5.82, y=0.14, z=0.075, half_width=0.52),
            Hazard("gutter_right", "guide", 5.82, y=-0.14, z=0.075, half_width=0.52),
            Hazard("pin0", "pin", 6.20, z=0.065),
            Hazard("pin1", "pin", 6.255, z=0.065),
            Hazard("pin2", "pin", 6.31, z=0.065),
            Hazard("exit_bar", "push_bar", 6.85, z=0.27),
            Hazard("portal", "finish_gate", 7.6, z=0.905),
        )
        return Level(seed, hazards, ((3.0, 3.15),), 7.9)
    return Level(seed, hazards, ((3.0, 3.15),), 5.8)


def build_level(seed=0, duck_floor_ref=None, hard_contacts=False, **kwargs):
    level = generate(seed, **kwargs)
    return (
        *build_world(
            gaps=level.gaps,
            end=8.5 if kwargs.get("difficulty") == "arcade" else 6.7,
            hazards=level.hazards,
            rails=True,
            recovery_bay=(3.5, 7.5, 0.9)
            if kwargs.get("difficulty") == "arcade"
            else (3.5, 5.2, 0.9),
            start=-4.8 if kwargs.get("difficulty") == "arcade" else -3.2,
            duck_floor_ref=duck_floor_ref,
            hard_contacts=hard_contacts,
        ),
        level,
    )
