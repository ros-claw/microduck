"""Versioned competition settings, separate from the archived one-exchange lab."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MatchConfig:
    schema: str = "duckfu.match.v1"
    mode: str = "simulation"
    players: int = 2
    ruleset: str = "team_ringout_v1"
    arena: str = "fixed_disc"
    radius_m: float = 0.75
    duration_s: float = 40.0
    separation_m: float = 0.85
    heading_rad: float = 0.0
    swap: bool = False
    down_count_s: float = 10.0
    dash_start_distance_m: float = 0.36
    dash_speed: float = 0.4
    dash_active_s: float = 2.2
    dash_windup_s: float = 0.24
    dash_contact_min_s: float = 1.0
    dash_neck_rad: float = -0.3
    dash_head_rad: float = -0.15
    evade_speed: float = 0.30
    evade_distance_m: float = 0.20
    evade_forward_speed: float = 0.0
    candidate_motor_path: str = ""
    candidate_motor_sha256: str = ""
    evade_motor_path: str = ""
    evade_motor_sha256: str = ""
    walk_motor_path: str = ""
    walk_motor_sha256: str = ""
    trained_motor_braking: bool = True
    threat_closing_m_s: float = 0.25
    allow_recovery_roll: bool = False
    tactics_path: str = ""
    tactics_sha256: str = ""
    opponent_path: str = ""
    opponent_sha256: str = ""
    red_style: str = "pressure"
    blue_style: str = "counter"
    physics_profile: str = "physics_profile.yaml"

    def __post_init__(self):
        if (
            type(self.trained_motor_braking) is not bool
            or type(self.allow_recovery_roll) is not bool
        ):
            raise ValueError("Braking contract flag must be boolean")
        if self.schema != "duckfu.match.v1" or self.mode != "simulation":
            raise ValueError("Competition is simulation only")
        if self.players not in (2, 4) or type(self.players) is not int:
            raise ValueError("Choose 1v1 or 2v2")
        if self.ruleset != "team_ringout_v1" or self.arena not in (
            "fixed_disc",
            "fixed_octagon",
        ):
            raise ValueError("Frozen ring-out/down-count rules required")
        if type(self.swap) is not bool:
            raise ValueError("swap must be a boolean")
        for value in asdict(self).values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite configuration")
        if not (
            0.55 <= self.radius_m <= 1.5
            and 8 <= self.duration_s <= 120
            and 0.5 <= self.separation_m <= 1.2
        ):
            raise ValueError("Invalid arena or episode bounds")
        if not (
            8 <= self.down_count_s <= 15
            and 0.28 <= self.dash_start_distance_m <= 0.44
            and 0.1 <= self.dash_speed <= 0.55
            and 0.4 <= self.dash_active_s <= 2.5
            and 0.1 <= self.dash_contact_min_s <= 1.5
            and 0.1 <= self.evade_speed <= 0.3
            and 0.1 <= self.evade_distance_m <= 0.24
            and -0.4 <= self.dash_neck_rad <= 0.4
            and -0.3 <= self.dash_head_rad <= 0.3
            and -0.3 <= self.evade_forward_speed <= 0.0
            and 0.05 <= self.threat_closing_m_s <= 0.25
        ):
            raise ValueError("Uncalibrated motor parameters")
        if not 0.1 <= self.dash_windup_s <= 0.4:
            raise ValueError("Invalid visible wind-up duration")
        for key in (
            "candidate_motor",
            "evade_motor",
            "walk_motor",
            "tactics",
            "opponent",
        ):
            path = getattr(self, key + "_path")
            sha = getattr(self, key + "_sha256")
            if bool(path) != bool(sha) or (
                sha
                and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha))
            ):
                raise ValueError(f"{key} needs its exact file and SHA256")
        if self.red_style not in (
            "pressure",
            "counter",
            "orbit",
            "defensive",
        ) or self.blue_style not in ("pressure", "counter", "orbit", "defensive"):
            raise ValueError("Unknown frozen rule opponent")

    def dict(self):
        return asdict(self)
