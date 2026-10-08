"""Monotone lifecycle: release is an input; falling/lost require physics evidence."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Tile:
    id: int
    eq_id: int
    qpos_adr: int
    qvel_adr: int
    state: str = "LOCKED"
    warning_at: float | None = None
    release_at: float | None = None

    def warn(self, now):
        if self.state != "LOCKED":
            raise ValueError("Only a locked tile can warn")
        self.state = "WARNING"
        self.warning_at = float(now)

    def release(self, data, now):
        if self.state != "WARNING" or now <= self.warning_at:
            raise ValueError("Release requires prior visible warning")
        data.eq_active[self.eq_id] = False
        self.state = "RELEASED"
        self.release_at = float(now)

    def update(self, data, now):
        events = []
        z = float(data.qpos[self.qpos_adr + 2])
        vz = float(data.qvel[self.qvel_adr + 2])
        if self.state == "RELEASED" and z < -0.027 and vz < -0.03:
            self.state = "FALLING"
            events.append(dict(time=now, tile=self.id, state=self.state, z=z, vz=vz))
        if self.state == "FALLING" and z < -0.15:
            self.state = "LOST"
            events.append(dict(time=now, tile=self.id, state=self.state, z=z, vz=vz))
        return events

    def public(self, now):
        return dict(
            id=self.id,
            state=self.state,
            warning_age_s=None if self.warning_at is None else now - self.warning_at,
        )


def bind_tiles(model):
    return [
        Tile(
            i,
            model.equality(f"tile_{i}_support").id,
            int(model.jnt_qposadr[model.joint(f"tile_{i}_free").id]),
            int(model.jnt_dofadr[model.joint(f"tile_{i}_free").id]),
        )
        for i in range(9)
    ]


class TileScheduler:
    """Seeded evaluator plan. Controller receives public state, never this object."""

    def __init__(self, seed, warning_s=4.0, interval_s=5.0, mode="seeded"):
        if (
            not np.isfinite([warning_s, interval_s]).all()
            or warning_s <= 0
            or interval_s < warning_s
        ):
            raise ValueError("Require finite interval >= positive warning duration")
        if mode not in ("seeded", "drop_test"):
            raise ValueError("Unknown schedule mode")
        order = (
            [1]
            if mode == "drop_test"
            else [
                1,
                *np.random.default_rng(seed)
                .permutation([0, 2, 3, 4, 5, 6, 7, 8])
                .tolist(),
            ]
        )
        self.plan = [
            dict(
                tile=i,
                warning_at=2.0 + j * interval_s,
                release_at=2.0 + j * interval_s + warning_s,
            )
            for j, i in enumerate(order)
        ]
        self.emitted = set()

    def advance(self, data, tiles, now):
        events = []
        for item in self.plan:
            tile = tiles[item["tile"]]
            for kind, key in [("WARNING", "warning_at"), ("RELEASED", "release_at")]:
                token = (tile.id, kind)
                if now + 1e-9 >= item[key] and token not in self.emitted:
                    if kind == "WARNING":
                        tile.warn(now)
                    else:
                        tile.release(data, now)
                    self.emitted.add(token)
                    events.append(dict(time=now, tile=tile.id, state=kind))
        return events
