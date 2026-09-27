"""Every-substep contact ledger and event-driven hero/benchmark scoring."""

from dataclasses import dataclass, field
import numpy as np
import mujoco


@dataclass
class Audit:
    hp: int = 3
    score: int = 0
    combo: int = 0
    events: list = field(default_factory=list)
    depths: list = field(default_factory=list)
    contacts: dict = field(default_factory=dict)
    fallen: bool = False
    fall_duration: float = 0.0
    _pending: dict = field(default_factory=dict)

    def sample(self, m, d, r, intentional_rotation=False):
        now = float(d.time)
        dt = float(m.opt.timestep)
        for i, c in enumerate(d.contact):
            b1 = m.body(m.geom_bodyid[c.geom1]).name
            b2 = m.body(m.geom_bodyid[c.geom2]).name
            if b1.startswith("duck/") == b2.startswith("duck/"):
                continue
            other = c.geom2 if b1.startswith("duck/") else c.geom1
            name = m.body(m.geom_bodyid[other]).name
            if name == "world" or name.startswith("duck/"):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] <= 1e-7:
                continue
            depth = max(0.0, -float(c.dist))
            self.depths.append(depth)
            record = self.contacts.setdefault(
                name,
                dict(
                    first=now, samples=0, normal_impulse_Ns=0.0, max_penetration_m=0.0
                ),
            )
            record["samples"] += 1
            record["normal_impulse_Ns"] += float(force[0]) * dt
            record["max_penetration_m"] = max(record["max_penetration_m"], depth)
            episode = self._pending.setdefault(
                name, dict(first=now, last=now, impulse=0.0, geom=int(other))
            )
            episode["last"] = now
            episode["impulse"] += float(force[0]) * dt
        for name, episode in list(self._pending.items()):
            # One knockdown/recovery is one injury episode for the same obstacle.
            # Raw contacts/impulses remain fully counted; other hazards stay separate.
            g = episode["geom"]
            cleared = (
                r.trunk_pos()[0] > d.geom_xpos[g, 0] + float(max(m.geom_size[g])) + 0.14
            )
            if now - episode["last"] > 0.60 and not self.fallen and cleared:
                self._commit_hit(name, episode)
                del self._pending[name]
        up = float(d.xmat[r.trunk_body_id, 8])
        self.fall_duration = (
            self.fall_duration + dt if up < 0.5 and not intentional_rotation else 0.0
        )
        if not self.fallen and self.fall_duration > 0.08:
            self.fallen = True
            self.combo = 0
            self.events.append(dict(type="FALL", t=now))

    def _commit_hit(self, name, episode):
        cost = 2 if episode["impulse"] > 0.8 else 1
        self.hp = max(0, self.hp - cost)
        self.combo = 0
        self.events.append(
            dict(
                type="IMPACT",
                t=episode["first"],
                hazard=name,
                normal_impulse_Ns=episode["impulse"],
                hp_cost=cost,
            )
        )

    def skill_result(self, maneuver, t):
        self.events.extend(maneuver.events)
        if maneuver.status != "SUCCESS":
            return
        if maneuver.name == "RECOVER":
            if self.fallen:
                self.events.append(dict(type="RECOVER", t=t))
                self.fallen = False
                self.fall_duration = 0.0
            return
        bonus = {
            "ROLL_CENTER": 250,
            "TAKE_LEFT_ROUTE": 100,
            "TAKE_RIGHT_ROUTE": 100,
            "JUMP_CENTER": 300,
        }.get(maneuver.name, 0)
        if bonus:
            self.combo += 1
            self.score += bonus
            self.events.append(
                dict(type="STUNT_SUCCESS", t=t, skill=maneuver.name, bonus=bonus)
            )

    def report(self):
        # Include a contact still active at the end; it must not evade damage.
        for name, episode in self._pending.items():
            self._commit_hit(name, episode)
        self._pending.clear()
        return dict(
            hp=self.hp,
            damage_rules=dict(
                unit="obstacle encounter until physically cleared",
                heavy_impulse_Ns=0.8,
                light_hp=1,
                heavy_hp=2,
            ),
            score=self.score,
            combo=self.combo,
            contacts=self.contacts,
            events=sorted(self.events, key=lambda e: e["t"]),
            max_penetration_m=max(self.depths, default=0.0),
            p99_penetration_m=float(np.quantile(self.depths, 0.99))
            if self.depths
            else 0.0,
            zero_contact_pass=not self.contacts,
        )


@dataclass
class PropAudit:
    """Finite-force contacts between scenery/props, independently of duck HP."""

    contacts: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    chase_seconds: float = 0.0
    longest_chase: float = 0.0

    def sample(self, m, d, r):
        dt = float(m.opt.timestep)
        boss = m.body("boss").id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, boss, velocity, 0)
        distance = float(r.trunk_pos()[0] - d.xpos[boss, 0])
        chasing = (
            0.25 < distance < 2.5 and velocity[3] > 0.10 and d.xpos[boss, 2] > 0.15
        )
        self.chase_seconds = self.chase_seconds + dt if chasing else 0.0
        self.longest_chase = max(self.longest_chase, self.chase_seconds)
        for i, c in enumerate(d.contact):
            names = [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]
            if any(n.startswith("duck/") for n in names) or names[0] == names[1]:
                continue
            if "boss" not in names:
                continue
            other = names[1] if names[0] == "boss" else names[0]
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] <= 1e-7:
                continue
            key = "boss/" + other
            if key not in self.contacts:
                self.events.append(
                    dict(
                        type="BOSS_LAND"
                        if other == "world"
                        else "BOSS_GATE_IMPACT"
                        if other == "portal"
                        else "BOSS_HIT_PROP",
                        t=float(d.time),
                        hazard=other,
                    )
                )
            rec = self.contacts.setdefault(
                key, dict(samples=0, normal_impulse_Ns=0.0, max_penetration_m=0.0)
            )
            rec["samples"] += 1
            rec["normal_impulse_Ns"] += float(force[0]) * dt
            rec["max_penetration_m"] = max(
                rec["max_penetration_m"], max(0.0, -float(c.dist))
            )
