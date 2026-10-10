"""Contact and height based elimination, with explicit one-body calibration outcome."""

from dataclasses import dataclass, field


@dataclass
class Referee:
    bodies: tuple[str, ...] = ("cream",)
    height_line: float = -0.12
    debounce_s: float = 0.3
    unsupported_since: dict = field(default_factory=dict)
    support_since: dict = field(default_factory=dict)
    last_time: float = 0.0
    eliminated: dict = field(default_factory=dict)

    def update(self, now, observations):
        events = []
        self.last_time = now
        # Process all bodies before choosing a winner: simultaneous losses => DRAW.
        for name in self.bodies:
            if name in self.eliminated:
                continue
            robot = observations[name]
            if robot["supporting_tiles"] and robot["z"] >= self.height_line:
                self.support_since.setdefault(name, now)
            else:
                self.support_since.pop(name, None)
            falling = robot["z"] < self.height_line and not robot["supporting_tiles"]
            if not falling:
                self.unsupported_since.pop(name, None)
                continue
            since = self.unsupported_since.setdefault(name, now)
            if now - since + 1e-9 >= self.debounce_s:
                evidence = dict(
                    time=now,
                    body_id=name,
                    state="ELIMINATED",
                    z=robot["z"],
                    vz=robot["vz"],
                    supporting_tile_ids=robot["supporting_tiles"],
                    contact_step=robot["contact_step"],
                )
                self.eliminated[name] = evidence
                events.append(evidence)
        return events

    def outcome(self, timeout=False):
        alive = [name for name in self.bodies if name not in self.eliminated]
        if not alive:
            return dict(
                status="ELIMINATED" if len(self.bodies) == 1 else "DRAW",
                winner=None,
                alive=[],
            )
        if (
            len(self.bodies) > 1
            and len(alive) == 1
            and alive[0] in self.support_since
            and self.last_time - self.support_since[alive[0]] >= 0.2
        ):
            return dict(status="WINNER", winner=alive[0], alive=alive)
        return dict(
            status="TIME_LIMIT" if timeout else "RUNNING", winner=None, alive=alive
        )
