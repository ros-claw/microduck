"""Declared one-attack coach. This is not a frozen reactive free-match opponent."""

from .tactics import Pilot
from .maneuver import MotorCommand


class SingleDriveCoach(Pilot):
    def choose(self, now, public, alive):
        if self.serial > 0 and self.action is None:
            recovery = self.guard.choose(now, public[self.name])
            if recovery is not None:
                self.phase, self.skill = "STABILIZE", "recover_guard"
                return MotorCommand("recovery", recovery)
            self.phase, self.skill = "READY", "single_attack_coach_hold"
            return MotorCommand("course_coach", "stand")
        return super().choose(now, public, alive)


class StandingCoach(Pilot):
    def choose(self, now, public, alive):
        recovery = self.guard.choose(now, public[self.name])
        if recovery is not None:
            self.phase, self.skill = "STABILIZE", "recover_guard"
            return MotorCommand("recovery", recovery)
        self.phase, self.skill = "READY", "standing_target_course"
        return MotorCommand("course_coach", "stand")
