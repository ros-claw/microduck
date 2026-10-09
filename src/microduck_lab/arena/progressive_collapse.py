"""One warned release at a time, preserving a connected route to the centre.

This is a disclosed game hazard, NOT a structural fracture simulation. Actual
foot-force fatigue ranks candidates; the public tempo creates early pressure.
No player identity enters tile selection or the bounded recovery grace.
"""

import math


def neighbours(i, nodes):
    return [j for j in nodes if abs(j//5-i//5)+abs(j%5-i%5) == 1]


def connected(nodes):
    if 12 not in nodes:
        return False
    reached, stack = {12}, [12]
    while stack:
        for j in neighbours(stack.pop(), nodes):
            if j not in reached:
                reached.add(j)
                stack.append(j)
    return reached == set(nodes)


class ProgressiveCollapse:
    first_warning_s = 6.0
    warning_s = 3.2
    grace_s = 1.2

    def __init__(self, config, graph, body_weight):
        self.config, self.graph, self.weight = config, graph, body_weight
        self.damage = dict.fromkeys(config.ids, 0.)
        self.filtered = dict.fromkeys(config.ids, 0.)
        self.last_emit = dict.fromkeys(config.ids, 0)
        self.pending = None
        self.deadline = None
        self.original_deadline = None
        self.next_warning = self.first_warning_s
        self.released = 0

    def remaining(self, tile, now):
        return max(0., self.deadline-now) if tile == self.pending else None

    def advance(self, d, tiles, pairs, now, observations, eliminated, enabled=True):
        if not enabled:
            return []
        events = []
        loads = dict.fromkeys(tiles, 0.)
        for p in pairs:
            a, b = p[:2]
            i = self.graph.tiles.get(a, self.graph.tiles.get(b))
            owner = self.graph.feet.get(a, self.graph.feet.get(b))
            up = p[14]*(1 if b in self.graph.feet else -1)
            if i is not None and owner is not None and d.eq_active[tiles[i].eq_id] and p[3] > .05 and up > .5:
                loads[i] += p[3]*up
        alpha = -math.expm1(-self.config.dt/self.config.filter_s)
        for i in tiles:
            self.filtered[i] += alpha*(min(3*self.weight, loads[i])-self.filtered[i])
            if i != 12 and d.eq_active[tiles[i].eq_id]:
                self.damage[i] = min(1., self.damage[i]+self.filtered[i]/self.weight*self.config.dt/self.config.lifetime_s)
                level = int(self.damage[i]*10)
                if level > self.last_emit[i]:
                    events.append(dict(time=now, state="DAMAGE", tile=i, damage=self.damage[i], load_N=self.filtered[i]))
                    self.last_emit[i] = level
        if self.pending is not None and now+1e-9 >= self.deadline:
            i = self.pending
            # Any currently toppled live duck over this footprint receives the
            # same bounded floor grace. Physics, collisions and the referee keep running.
            centre = self.config.centre(i)
            fallen = any(n not in eliminated and not o["upright"] and o["z"] >= -.12
                         and abs(o["x"]-centre[0]) <= self.config.tile_size/2
                         and abs(o["y"]-centre[1]) <= self.config.tile_size/2
                         for n, o in observations.items())
            if fallen and self.deadline == self.original_deadline:
                self.deadline += self.grace_s
                events.append(dict(time=now, state="RECOVERY_GRACE", tile=i, release_at_s=self.deadline, extension_s=self.grace_s))
            else:
                tiles[i].release(d, now)
                events.append(dict(time=now, state="RELEASED", tile=i, cause="progressive_public_hazard", damage=self.damage[i], warning_s=now-tiles[i].warning_at))
                self.pending = None
                self.released += 1
                self.next_warning = now
        if self.pending is None and now+1e-9 >= self.next_warning:
            active = {i for i, tile in tiles.items() if d.eq_active[tile.eq_id]}
            legal = [i for i in sorted(active-{12}) if connected(active-{i})]
            if legal:
                # Geometry + accumulated load, never names, scores or winner preference.
                i = max(legal, key=lambda j: (max(abs(j//5-2), abs(j%5-2)) + .6*self.damage[j], self.damage[j], -j))
                tiles[i].warn(now)
                self.pending = i
                self.deadline = self.original_deadline = now+self.warning_s
                events.append(dict(time=now, state="FINAL_WARNING", tile=i, release_at_s=self.deadline, cause="progressive_public_hazard", active_tiles=len(active), route_connected_after_release=True))
        return events


class ConnectedBeacons:
    """Roaming objectives stay on existing connected tiles; capture never wins."""

    def __init__(self, config, names):
        self.config = config
        self.scores = dict.fromkeys(names, 0)
        self.progress = dict.fromkeys(names, 0.)
        self.events = []
        self.round = 0
        self.goal = 12
        self.contested_steps = 0

    def move(self, now, tiles):
        active = [i for i, tile in tiles.items() if tile.state == "LOCKED"]
        if not active:
            active = [12]
        # Prefer remaining outposts then a reachable perimeter; deterministic,
        # independent of current competitors' identities and score.
        choices = [i for i in self.config.stations if i in active and i != self.goal]
        if not choices:
            choices = [i for i in sorted(active) if i != self.goal] or [12]
        self.goal = choices[self.round % len(choices)]
        self.progress = dict.fromkeys(self.progress, 0.)
        self.events.append(dict(time=now, state="BEACON_MOVED", tile=self.goal, round=self.round))

    def update(self, now, observations, eliminated, tiles):
        if tiles[self.goal].state != "LOCKED":
            self.move(now, tiles)
        if now < .8:
            return
        centre = self.config.centre(self.goal)
        occupants = [n for n, o in observations.items() if n not in eliminated and o["z"] >= -.12
                     and math.hypot(o["x"]-centre[0], o["y"]-centre[1]) <= self.config.claim_radius_m]
        if len(occupants) != 1:
            self.contested_steps += len(occupants) > 1
            return
        n = occupants[0]
        if not observations[n]["upright"] or self.goal not in observations[n]["supporting_tiles"]:
            return
        self.progress[n] += self.config.dt
        if self.progress[n] >= self.config.claim_s:
            self.scores[n] += 1
            self.events.append(dict(time=now, state="BEACON_CAPTURE", tile=self.goal, body_id=n, scores=self.scores.copy(), round=self.round))
            self.round += 1
            self.move(now, tiles)
