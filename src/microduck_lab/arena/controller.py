"""Minimal warning response for DG-02, not a qualified general survivor skill."""

import math
import numpy as np

PITCH = 0.492


def centre(tile):
    return np.array([(tile // 3 - 1) * PITCH, (tile % 3 - 1) * PITCH])


def observation(duck, tiles, supports, now):
    pos = duck.trunk_pos()
    current = (
        min(supports, key=lambda i: float(np.linalg.norm(pos[:2] - centre(i))))
        if supports
        else None
    )
    return dict(
        schema="microduck.arena.observation.v1",
        sim_time_s=now,
        robot=dict(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            yaw=duck.trunk_yaw(),
            current_tile_id=current,
        ),
        visible_tiles=[tile.public(now) for tile in tiles],
    )


class WarningResponder:
    def __init__(self):
        self.target = None

    def choose(self, obs):
        robot = obs["robot"]
        tiles = {t["id"]: t["state"] for t in obs["visible_tiles"]}
        current = robot["current_tile_id"]
        if self.target is not None and tiles[self.target] != "LOCKED":
            self.target = None
        if self.target is None and current is not None and tiles[current] == "WARNING":
            adjacent = [
                i
                for i, state in tiles.items()
                if state == "LOCKED"
                and abs(i // 3 - current // 3) + abs(i % 3 - current % 3) == 1
            ]

            # Distance/yaw cost uses only current visible geometry, not future TTL.
            def cost(i):
                delta = centre(i) - np.array([robot["x"], robot["y"]])
                yaw = math.atan2(delta[1], delta[0]) - robot["yaw"]
                return float(np.linalg.norm(delta)) + 0.15 * abs(
                    math.atan2(math.sin(yaw), math.cos(yaw))
                )

            if adjacent:
                self.target = min(adjacent, key=lambda i: (cost(i), i))
        if self.target is not None:
            delta = centre(self.target) - np.array([robot["x"], robot["y"]])
            if np.linalg.norm(delta) < 0.045 and current == self.target:
                self.target = None
                return "stand", (0.0, 0.0, 0.0), "ARRIVED"
            desired = math.atan2(delta[1], delta[0])
            angle = math.atan2(
                math.sin(desired - robot["yaw"]), math.cos(desired - robot["yaw"])
            )
            return (
                "walk",
                (
                    float(np.clip(2.5 * np.linalg.norm(delta), 0.08, 0.45))
                    if abs(angle) < 0.45
                    else 0.0,
                    0.0,
                    float(np.clip(3 * angle, -1.6, 1.6)),
                ),
                "GO_TO_TILE",
            )
        return (
            "stand",
            (0.0, 0.0, 0.0),
            "NO_SAFE_OPTION"
            if current is not None and tiles[current] == "WARNING"
            else "HOLD",
        )
