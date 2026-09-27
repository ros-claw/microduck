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
):
    rng = np.random.default_rng(seed)
    jitter = lambda: float(rng.uniform(-0.025, 0.025))
    hazards = (
        Hazard("bar", "push_bar", 0.9 + jitter()),
        Hazard("crate", "crate", 1.85 + jitter(), z=0.65),
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
        Hazard("boss", "boulder", -2.7, z=0.8, speed=0.40),
        Hazard("portal", "finish_gate", 5.5, z=0.905),
    )
    return Level(seed, hazards, ((3.0, 3.15),), 5.8)


def build_level(seed=0, duck_floor_ref=None, **kwargs):
    level = generate(seed, **kwargs)
    return (
        *build_world(
            gaps=level.gaps,
            end=6.7,
            hazards=level.hazards,
            rails=True,
            recovery_bay=(3.5, 5.2, 0.9),
            start=-3.2,
            duck_floor_ref=duck_floor_ref,
        ),
        level,
    )
