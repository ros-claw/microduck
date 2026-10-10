"""Longer physical match budget for serial, four-second floor warnings."""
from dataclasses import dataclass, asdict
from .survival_rumble import SurvivalConfig


@dataclass(frozen=True)
class JevGameConfig(SurvivalConfig):
    duration: float = 120.
    tilt_max_rad: float = .08

    def __post_init__(self):
        if not 20 <= self.duration <= 120:
            raise ValueError("Jev matches require 20–120 simulation seconds")
        # Reuse every archived physical calibration guard. Only the match time
        # budget is extended; the captured config records its actual duration.
        values = asdict(self)
        values["duration"] = min(self.duration, 90.)
        SurvivalConfig(**values)
