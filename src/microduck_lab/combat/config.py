"""Strict first-round contracts. Unimplemented modes fail before simulation."""

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class Physics:
    schema: str = "duckfu.physics.v1"
    dt: float = 0.00025
    motor_hz: int = 50
    iterations: int = 80
    contact_ref_s: float = 0.0005
    torque_limit_Nm: float = 0.6405
    collision: str = "visual_convex_parts"
    actuator: str = "position_servo_simulation_only"

    def __post_init__(self):
        if (
            self.schema != "duckfu.physics.v1"
            or self.actuator != "position_servo_simulation_only"
        ):
            raise ValueError("Unsupported physics/actuator contract")
        if self.dt not in (0.000125, 0.00025) or self.motor_hz != 50:
            raise ValueError("Only calibrated-rate 50 Hz CPU profiles are supported")
        if (
            self.iterations != 80
            or self.contact_ref_s != 0.0005
            or self.torque_limit_Nm != 0.6405
        ):
            raise ValueError("Physics calibration constants are locked")
        if self.collision not in ("native", "visual_convex_parts"):
            raise ValueError("Unsupported convex collision profile")
        if not math.isclose(self.substeps * self.dt, 1 / self.motor_hz, abs_tol=1e-12):
            raise ValueError("Physics must divide the motor period")

    @property
    def substeps(self):
        return round(1 / self.motor_hz / self.dt)


@dataclass(frozen=True)
class Config:
    schema: str = "duckfu.duel.v1"
    mode: str = "simulation"
    ruleset: str = "sparring_v1"
    arena: str = "lab"
    radius_m: float = 1.5
    duration_s: float = 12.0
    separation_m: float = 0.54
    lateral_offset_m: float = 0.0
    heading_rad: float = 0.0
    swap: bool = False
    defense: str = "counter"
    evade_sign: int = 1
    dash_speed: float = 0.45
    dash_start_distance_m: float = 0.40
    dash_active_s: float = 1.25
    dash_contact_min_s: float = 0.16
    dash_neck_rad: float = -0.12
    dash_head_rad: float = -0.10
    evade_speed: float = 0.30
    evade_distance_m: float = 0.12
    evade_forward_speed: float = -0.26
    threat_closing_m_s: float = 0.18
    threat_distance_m: float = 0.40
    down_count_s: float = 10.0
    physics_profile: str = "physics_profile.yaml"
    candidate_motor_path: str = ""
    candidate_motor_sha256: str = ""

    def __post_init__(self):
        if self.schema != "duckfu.duel.v1" or self.mode != "simulation":
            raise ValueError("Unsupported contract; this is a simulation duel lab")
        if self.ruleset not in (
            "sparring_v1",
            "contact_stress_v1",
        ) or self.arena not in ("lab", "fixed_disc"):
            raise ValueError("Unsupported rules/arena")
        if self.defense not in (
            "stand",
            "evade",
            "counter",
            "absent",
        ) or self.evade_sign not in (-1, 1):
            raise ValueError("Unsupported defense")
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"Nonfinite {f.name}")
        if type(self.swap) is not bool or type(self.evade_sign) is not int:
            raise ValueError("Invalid bool/integer fields")
        if not (
            2 <= self.duration_s <= 45
            and 0.4 <= self.radius_m <= 2
            and 0.28 <= self.separation_m <= 1.2
        ):
            raise ValueError("Invalid arena/duration/separation")
        if not (
            0.1 <= self.dash_speed <= 0.55
            and 0.1 <= self.evade_speed <= 0.30
            and 0.23 <= self.threat_distance_m <= 0.5
        ):
            raise ValueError("Commands exceed the development search envelope")
        if not (8 <= self.down_count_s <= 15) or abs(self.lateral_offset_m) > 0.15:
            raise ValueError("Invalid down-count/offset")
        if not 0.26 <= self.dash_start_distance_m <= 0.44:
            raise ValueError("Dash entry is outside the calibration envelope")
        if not (0.4 <= self.dash_active_s <= 2.5 and 0.1 <= self.dash_contact_min_s <= 1.5):
            raise ValueError("Invalid bounded attack/contact duration")
        if not (-0.4 <= self.dash_neck_rad <= 0.4 and -0.3 <= self.dash_head_rad <= 0.3):
            raise ValueError("Head commands exceed the probed native range")
        if not .10 <= self.evade_distance_m <= .24:
            raise ValueError("Invalid bounded evade travel")
        if not (-.30 <= self.evade_forward_speed <= 0. and .08 <= self.threat_closing_m_s <= .25):
            raise ValueError("Invalid public threat/evade parameters")
        if bool(self.candidate_motor_path) != bool(self.candidate_motor_sha256):
            raise ValueError("Candidate motor requires an explicit file and digest")
        if self.candidate_motor_sha256 and (
            len(self.candidate_motor_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.candidate_motor_sha256)
        ):
            raise ValueError("Invalid candidate digest")

    def dict(self):
        return asdict(self)


def strict_load(path, cls):
    content = yaml.safe_load(Path(path).read_text())
    if not isinstance(content, dict) or set(content) - {f.name for f in fields(cls)}:
        raise ValueError(f"Unknown configuration fields in {path}")
    # Reject coercion such as 'false', booleans as numeric values and string NaNs.
    for f in fields(cls):
        if f.name not in content:
            continue
        v, expected = content[f.name], type(f.default)
        if expected is float:
            valid = type(v) in (int, float) and math.isfinite(v)
        else:
            valid = type(v) is expected
        if not valid:
            raise ValueError(f"Invalid type for {f.name}")
    return cls(**content)


def load(path):
    path = Path(path)
    config = strict_load(path, Config)
    return config, strict_load(path.parent / config.physics_profile, Physics)
