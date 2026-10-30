"""Symmetric ring-out/down-count outcomes; observes and never edits physics."""


class Referee:
    def __init__(self, names, down_count, teams=None):
        self.names = tuple(names)
        self.down_count = down_count
        self.down_since = dict.fromkeys(names, None)
        self.out_since = dict.fromkeys(names, None)
        self.eliminated = {}
        self.events = []
        self.result = None
        self.teams = teams
        if teams is not None and set(teams) != set(self.names):
            raise ValueError("Every competitor needs a team")

    def update(self, now, bodies, physics_fault=False):
        if physics_fault:
            self.result = dict(kind="NO_CONTEST_PHYSICS", winner=None, time_s=now)
            return
        pending = {}
        for name in self.names:
            if name in self.eliminated:
                continue
            b = bodies[name]
            # Receiver is below -.5 m; it never counts as stage foot support.
            out = not b.foot_support and b.z < -0.15
            self.out_since[name] = (
                (self.out_since[name] if self.out_since[name] is not None else now)
                if out
                else None
            )
            down = b.upright < 0.55 and b.body_ground
            # Once DOWN, short upright flickers cannot erase its timer.
            recovered = b.upright > 0.94 and b.foot_support and b.speed < 0.16
            if down and self.down_since[name] is None:
                self.down_since[name] = now
                self.events.append(dict(state="DOWN", actor=name, time_s=now))
            if recovered and self.down_since[name] is not None:
                # Completion stability is confirmed by the motor's own guard;
                # referee holds .3 s too, without consulting its skill labels.
                recovered_since = getattr(self, "_stable", {}).get(name)
                if recovered_since is None:
                    if not hasattr(self, "_stable"):
                        self._stable = {}
                    self._stable[name] = now
                elif now - recovered_since >= 0.3 - 1e-9:
                    self.down_since[name] = None
                    self.events.append(dict(state="RECOVERED", actor=name, time_s=now))
            elif hasattr(self, "_stable"):
                self._stable.pop(name, None)
            if (
                self.out_since[name] is not None
                and now - self.out_since[name] >= 0.12 - 1e-9
            ):
                pending[name] = "RING_OUT"
            elif (
                self.down_since[name] is not None
                and now - self.down_since[name] >= self.down_count - 1e-9
            ):
                pending[name] = "DOWN_COUNT"
        for name, reason in pending.items():
            self.eliminated[name] = reason
            self.events.append(dict(state=reason, actor=name, time_s=now))
        alive = [n for n in self.names if n not in self.eliminated]
        remaining = sorted({self.teams[n] for n in alive}) if self.teams is not None else alive
        if len(remaining) <= 1:
            self.result = dict(
                kind="WIN" if alive else "DRAW",
                winner=remaining[0] if alive else None,
                time_s=now,
            )

    def finish(self, now):
        return self.result or dict(kind="TIMEOUT_DRAW", winner=None, time_s=now)
