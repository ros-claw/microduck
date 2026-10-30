"""Pre-integration contacts, positive normal impulses, debounced pair episodes."""

from array import array
import mujoco
import numpy as np

CATEGORIES = ("duck_duck", "duck_stage", "self", "receiver")


class ContactAudit:
    def __init__(self, m, ducks):
        self.owners = {}
        self.parts = {}
        for i in range(m.ngeom):
            body = m.body(int(m.geom_bodyid[i])).name
            name = body.split("/")[0]
            if name in ducks:
                self.owners[i] = name
                self.parts[i] = body.split("/")[-1]
        self.feet = {
            m.geom(n + "/" + side + "_foot_collision").id: n
            for n in ducks
            for side in ("left", "right")
        }
        self.stage, self.receiver = m.geom("stage").id, m.geom("receiver").id
        self.depths = {k: array("d") for k in CATEGORIES}
        self.maxima = {
            k: dict(depth_m=0.0, step=None, geoms=None, position_world_m=None)
            for k in CATEGORIES
        }
        self.support = {n: False for n in ducks}
        self.body_ground = {n: False for n in ducks}
        self.touching = {n: False for n in ducks}
        self.pair_active = {}
        self.episodes = []
        self.events = []
        self.force = np.zeros(6)

    def read(self, m, d, step, dt, attacks):
        # mj_step's contacts and xposes are evaluated at its pre-integration time;
        # qpos/qvel have advanced. Episode runner stores those poses separately.
        self.support = dict.fromkeys(self.support, False)
        self.body_ground = dict.fromkeys(self.support, False)
        self.touching = dict.fromkeys(self.support, False)
        pairs, full = {}, []
        for j in range(d.ncon):
            contact = d.contact[j]
            a, b = int(contact.geom1), int(contact.geom2)
            oa, ob = self.owners.get(a), self.owners.get(b)
            if not (oa or ob):
                continue
            cat = (
                "receiver"
                if self.receiver in (a, b)
                else "duck_stage"
                if self.stage in (a, b)
                else "self"
                if oa == ob
                else "duck_duck"
            )
            depth = max(0.0, -float(contact.dist))
            if depth > 0:
                self.depths[cat].append(depth)
            if depth > self.maxima[cat]["depth_m"]:
                self.maxima[cat] = dict(
                    depth_m=depth,
                    step=step,
                    geoms=[a, b],
                    position_world_m=contact.pos.tolist(),
                )
            mujoco.mj_contactForce(m, d, j, self.force)
            fn = max(0.0, float(self.force[0]))
            normal = contact.frame[:3]
            if self.stage in (a, b) and fn > 0.05:
                foot = self.feet.get(a, self.feet.get(b))
                if foot and normal[2] * (1 if b in self.feet else -1) > 0.5:
                    self.support[foot] = True
                elif not foot:
                    self.body_ground[oa or ob] = True
            if cat == "duck_duck" and fn > 0.05:
                self.touching[oa] = self.touching[ob] = True
                pair = tuple(sorted((oa, ob)))
                value = pairs.setdefault(
                    pair, [0.0, np.zeros(3), 0.0, float(contact.dist), []]
                )
                value[0] += fn * dt
                # Vector is force on the second name in the canonical pair.
                value[1] += normal * fn * dt * (1 if ob == pair[1] else -1)
                value[2] = max(value[2], fn)
                value[3] = min(value[3], float(contact.dist))
                value[4].append([self.parts[a], self.parts[b]])
            full.append(
                [
                    a,
                    b,
                    float(contact.dist),
                    *self.force.tolist(),
                    *contact.pos.tolist(),
                    *normal.tolist(),
                ]
            )
        for pair, value in pairs.items():
            if pair not in self.pair_active:
                event = dict(
                    event_id=len(self.episodes),
                    state="CONTACT",
                    actor_pair=list(pair),
                    start_step=step,
                    end_step=step,
                    summed_normal_impulse_Ns=0.0,
                    normal_impulse_on_second_Ns=[0.0, 0.0, 0.0],
                    peak_force_N=0.0,
                    min_distance_m=0.0,
                    duration_s=0.0,
                    attacks=[],
                    parts=[],
                )
                self.pair_active[pair] = event
                self.episodes.append(event)
                self.events.append(
                    dict(
                        state="CONTACT",
                        time_s=step * dt,
                        event_id=event["event_id"],
                        actors=list(pair),
                    )
                )
            e = self.pair_active[pair]
            e["end_step"] = step
            e["summed_normal_impulse_Ns"] += value[0]
            e["normal_impulse_on_second_Ns"] = (
                np.array(e["normal_impulse_on_second_Ns"]) + value[1]
            ).tolist()
            e["peak_force_N"] = max(e["peak_force_N"], value[2])
            e["min_distance_m"] = min(e["min_distance_m"], value[3])
            e["duration_s"] += dt
            for name in pair:
                if attacks.get(name) and attacks[name] not in e["attacks"]:
                    e["attacks"].append(attacks[name])
            for parts in value[4]:
                if parts not in e["parts"]:
                    e["parts"].append(parts)
        for pair, e in list(self.pair_active.items()):
            if (step - e["end_step"]) * dt > 0.04:
                del self.pair_active[pair]
        return full

    def summary(self):
        cats = {}
        for key in CATEGORIES:
            v = np.frombuffer(self.depths[key], dtype=np.float64)
            cats[key] = dict(
                penetrating_contact_samples=len(v),
                p99_penetration_m=float(np.quantile(v, 0.99)) if len(v) else 0.0,
                maximum=self.maxima[key],
            )
        populated = [
            np.frombuffer(v, dtype=np.float64) for v in self.depths.values() if len(v)
        ]
        allv = np.concatenate(populated) if populated else np.zeros(1)
        return dict(
            categories=cats,
            penetration_max_m=float(allv.max()),
            penetrating_p99_m=float(np.quantile(allv, 0.99)),
            episodes=self.episodes,
            sampling="mj_step solved pre-integration contact; post-integration qpos separately recorded",
            gate_proposal_pass=bool(
                allv.max() <= 0.003 and np.quantile(allv, 0.99) <= 0.0015
            ),
        )
