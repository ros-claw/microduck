"""Opt-in V2 contract; old 34D/9-action tactical weights are incompatible."""

from dataclasses import dataclass, fields
from .match_config import MatchConfig


@dataclass(frozen=True)
class V2Config(MatchConfig):
    schema: str = "duckfu.match.v2"
    red_v2: bool = True
    blue_v2: bool = False
    dynamic_handoff: bool = True
    team_intents: bool = False
    red_v2_style: str = "mixed"
    blue_v2_style: str = "counter"
    probe_active_s: float = 0.64
    probe_windup_s: float = 0.12
    probe_travel_m: float = 0.14
    exit_contact_after_s: float = 0.65
    team_hold_s: float = 1.6
    initial_layout: str = "standard"
    opponent_course: str = "none"
    action_course: str = "none"
    controller_revision: int = 1
    recovery_mode: str = "legacy"
    recovery_motor_path: str = ""
    recovery_motor_sha256: str = ""
    finish_active_s: float = 2.2
    finish_contact_s: float = 2.0
    finish_travel_m: float = 0.38
    finish_entry_m: float = 0.34
    finish_safety_m: float = 0.14
    blue_motor_path: str = ""
    blue_motor_sha256: str = ""

    def __post_init__(self):
        if self.schema != "duckfu.match.v2":
            raise ValueError("V2 requires its own explicit schema")
        if bool(self.blue_motor_path) != bool(self.blue_motor_sha256) or (
            self.blue_motor_sha256
            and (
                len(self.blue_motor_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.blue_motor_sha256)
            )
        ):
            raise ValueError("Blue motor requires an exact pinned digest")
        if self.recovery_mode not in (
            "legacy",
            "stand_first",
            "contact_learned",
            "learned_stabilize",
        ):
            raise ValueError("Unknown measured body recovery mode")
        if self.recovery_mode in ("contact_learned", "learned_stabilize"):
            if (
                not self.recovery_motor_path
                or len(self.recovery_motor_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.recovery_motor_sha256)
            ):
                raise ValueError("Learned recovery requires an exact motor and digest")
        elif self.recovery_motor_path or self.recovery_motor_sha256:
            raise ValueError("Recovery motor requires explicit contact_learned mode")
        if type(
            self.controller_revision
        ) is not int or self.controller_revision not in (1, 2, 3, 4):
            raise ValueError("Unknown V2 controller revision")
        if not (
            1.3 <= self.finish_active_s <= 2.5
            and 0.7 <= self.finish_contact_s <= 2.4
            and 0.24 <= self.finish_travel_m <= 0.45
            and 0.24 <= self.finish_entry_m <= 0.40
            and 0.10 <= self.finish_safety_m <= 0.22
        ):
            raise ValueError("Finishing request outside bounded development envelope")
        values = {f.name: getattr(self, f.name) for f in fields(MatchConfig)}
        values["schema"] = "duckfu.match.v1"
        MatchConfig(**values)
        if self.tactics_path or self.opponent_path:
            raise ValueError(
                "V1 34D/9-action tactical weights are incompatible with V2; no padding/fallback"
            )
        if not all(
            type(getattr(self, k)) is bool
            for k in ("red_v2", "blue_v2", "dynamic_handoff", "team_intents")
        ):
            raise ValueError("V2 switches must be boolean")
        if not self.red_v2 and not self.blue_v2 and self.opponent_course == "none":
            raise ValueError("Use the frozen V1 schema for two V1 controllers")
        if not (
            0.24 <= self.probe_active_s <= 0.7
            and 0.1 <= self.probe_windup_s <= 0.24
            and 0.05 <= self.probe_travel_m <= 0.15
            and 0.4 <= self.exit_contact_after_s <= 1.5
            and 1 <= self.team_hold_s <= 3
        ):
            raise ValueError("V2 candidate parameters outside the development envelope")
        if self.red_v2_style not in (
            "mixed",
            "counter",
            "pressure",
        ) or self.blue_v2_style not in ("mixed", "counter", "pressure"):
            raise ValueError("Unknown V2 rule style")
        if not all(
            (self.candidate_motor_path, self.evade_motor_path, self.walk_motor_path)
        ):
            raise ValueError("V2 requires all three pinned qualified motor policies")
        if self.initial_layout not in ("standard", "peel_course") or (
            self.initial_layout == "peel_course"
            and (self.players != 4 or not self.team_intents)
        ):
            raise ValueError(
                "The labelled peel course requires four ducks and team intents"
            )
        if self.opponent_course not in ("none", "single_drive", "stand") or (
            self.opponent_course != "none" and (self.players != 2 or self.blue_v2)
        ):
            raise ValueError(
                "Single-drive coach is an explicitly labelled two-duck course"
            )
        if self.action_course not in ("none", "probe_bump", "committed_drive") or (
            self.action_course != "none" and self.opponent_course != "stand"
        ):
            raise ValueError(
                "Single-maneuver qualification needs its standing-target course"
            )
