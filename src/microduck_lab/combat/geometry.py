"""One physical boundary shared by the arena, observers and action masks."""

import math
import numpy as np
import mujoco


def boundary_distance(point, config):
    xy = np.asarray(point)[:2]
    if config.arena == "fixed_octagon":
        angles = np.arange(8) * math.pi / 4 + math.pi / 8
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        return float(config.radius_m * math.cos(math.pi / 8) - np.max(normals @ xy))
    if config.arena == "fixed_disc":
        return float(config.radius_m - np.linalg.norm(xy))
    return float(config.radius_m - np.max(np.abs(xy)))


def add_stage(spec, config, **kwargs):
    if config.arena == "fixed_octagon":
        vertices = [
            [
                config.radius_m * math.cos(i * math.pi / 4),
                config.radius_m * math.sin(i * math.pi / 4),
                z,
            ]
            for z in (-0.04, 0.04)
            for i in range(8)
        ]
        spec.add_mesh(
            name="fixed_octagon_mesh", uservert=np.asarray(vertices).flatten().tolist()
        )
        return spec.worldbody.add_geom(
            name="stage",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="fixed_octagon_mesh",
            pos=[0, 0, -0.04],
            **kwargs,
        )
    kind = (
        mujoco.mjtGeom.mjGEOM_BOX
        if config.arena == "lab"
        else mujoco.mjtGeom.mjGEOM_CYLINDER
    )
    size = (
        [config.radius_m, config.radius_m, 0.04]
        if config.arena == "lab"
        else [config.radius_m, 0.04, 0]
    )
    return spec.worldbody.add_geom(
        name="stage", type=kind, size=size, pos=[0, 0, -0.04], **kwargs
    )
