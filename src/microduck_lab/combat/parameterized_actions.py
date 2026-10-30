"""Bounded maneuver requests, not arbitrary motor or joint commands."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ManeuverRequest:
    actor: str
    target: str
    family: str
    side: int = 1
    role: str = "DUEL"
    schema: str = "duckfu.maneuver.v2"

    def __post_init__(self):
        if self.schema != "duckfu.maneuver.v2" or self.family not in (
            "approach",
            "probe_bump",
            "committed_drive",
            "cut_counter",
            "contact_exit",
            "center",
            "brace",
        ):
            raise ValueError("Unknown bounded maneuver family")
        if not self.actor or not self.target or self.actor == self.target:
            raise ValueError("A distinct physical target is required")
        if type(self.side) is not int or self.side not in (-1, 1):
            raise ValueError("Side must be -1 or +1")
