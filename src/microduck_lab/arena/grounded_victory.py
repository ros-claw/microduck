"""A sole supported survivor need not make an additional dangerous trip."""

class GroundedVictory:
    """Only one live body plus upright actual grounded floor support may win.

    A 0.75s stability window must contain >=90% actual foot support and no
    unsupported gap longer than 30ms. Any rival still alive prevents victory.
    """

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = None
        self.last_loaded = None
        self.loaded_s = 0.0
        self.max_gap = 0.0

    def update(self, now, observations, eliminated):
        alive = [n for n in observations if n not in eliminated]
        if not alive:
            self.reset()
            return dict(status="DRAW", winner=None, alive=[], rule="LastDuckAlive")
        if len(alive) != 1:
            self.reset()
            return None
        n = alive[0]
        o = observations[n]
        if not o["upright"] or o["z"] < -0.12:
            self.reset()
            return None
        loaded = bool(o["supporting_tiles"])
        if self.candidate != n:
            self.reset()
            if not loaded:
                return None
            self.candidate = n
            self.since = now
            self.last_loaded = now
        if loaded:
            self.loaded_s += self.config.dt
            self.last_loaded = now
        gap = now - self.last_loaded
        self.max_gap = max(self.max_gap, gap)
        if gap > 0.03 + 1e-9:
            self.reset()
            return None
        elapsed = now - self.since
        fraction = self.loaded_s / max(elapsed, self.config.dt)
        if loaded and elapsed + 1e-9 >= self.config.winner_hold_s and fraction >= 0.9:
            return dict(
                status="WINNER",
                winner=n,
                alive=alive,
                rule="LastDuckAlive",
                support_rule="any remaining grounded tile",
                supporting_tiles_at_win=list(o["supporting_tiles"]),
                stable_supported_s=elapsed,
                loaded_fraction=min(1.0, fraction),
                max_unsupported_gap_s=self.max_gap,
                eliminated=sorted(eliminated),
            )
        return None

