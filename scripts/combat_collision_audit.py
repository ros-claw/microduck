"""Audit native convex hull coverage, not an exact visual triangle-distance claim."""

from pathlib import Path
import argparse
import itertools
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from microduck_lab.combat.world import ASSETS, ROBOT, digest
from microduck_lab.combat.episode import write_json


def vertices(m, g):
    mesh = int(m.geom_dataid[g])
    start = int(m.mesh_vertadr[mesh])
    end = start + int(m.mesh_vertnum[mesh])
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, m.geom_quat[g])
    return m.mesh_vert[start:end] @ mat.reshape(3, 3).T + m.geom_pos[g]


def max_violation(points, eq):
    return np.concatenate(
        [
            np.max(points[i : i + 256] @ eq[:, :3].T + eq[:, 3], axis=1)
            for i in range(0, len(points), 256)
        ]
    )


def audit():
    m = mujoco.MjModel.from_xml_path(str(ASSETS / ROBOT))
    result = {}
    for name in ("jaw_soft", "trunk_base", "ankle_left", "ankle_right"):
        body = m.body(name).id
        visual = np.unique(
            np.concatenate(
                [
                    vertices(m, g)
                    for g in range(m.ngeom)
                    if m.geom_bodyid[g] == body and m.geom_group[g] == 2
                ]
            ),
            axis=0,
        )
        collision = [
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == body and m.geom_contype[g] & 1
        ]
        hulls = [ConvexHull(vertices(m, g)).equations for g in collision]
        violation = np.min(
            np.array([max_violation(visual, eq) for eq in hulls]), axis=0
        )
        row = dict(
            visual_vertices=len(visual),
            native_external_hulls=len(hulls),
            outside_union_fraction=float(np.mean(violation > 1e-5)),
            maximum_halfspace_gap_lower_bound_m=float(max(0.0, violation.max())),
            worst_visual_point_body_frame_m=visual[int(violation.argmax())].tolist(),
        )
        if name in ("jaw_soft", "trunk_base"):
            low, high = visual.min(0) - 0.0005, visual.max(0) + 0.0005
            corners = np.array(list(itertools.product(*zip(low, high))))
            visual_hull = ConvexHull(visual).equations
            gap = np.max(corners @ visual_hull[:, :3].T + visual_hull[:, 3], axis=1)
            row["legacy_aabb_corner_gap_to_visual_hull_lower_bound_m"] = float(
                max(0.0, gap.max())
            )
        result[name] = row
    return dict(
        schema="duckfu.coverage.v1",
        robot_sha256=digest(ASSETS / ROBOT),
        bodies=result,
        limitations=[
            "Vertex coverage and convex halfspace bounds only; not visual triangle-exact collision.",
            "Native mesh geoms use convex hulls; uncovered vertices expose missing surface coverage.",
            "No additional legacy AABB shells used by the combat world.",
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = Path(a.out)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    write_json(path, result)
    print(result)
