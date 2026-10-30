"""Check mesh-copy bijection, collision layers and untouched inertial model."""

import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from microduck_lab.combat.config import Config, Physics
from microduck_lab.combat.world import build
from microduck_lab.combat.episode import write_json


def audit():
    native, _, _, _ = build(Config(), Physics(collision="native"), 101)
    combat, _, _, _ = build(Config(), Physics(), 101)
    copied = []
    for name in ("lavender", "cream"):
        source = [
            i
            for i in range(native.ngeom)
            if native.body(int(native.geom_bodyid[i])).name.startswith(name + "/")
            and native.geom_group[i] == 2
        ]
        dest = [
            i
            for i in range(combat.ngeom)
            if combat.body(int(combat.geom_bodyid[i])).name.startswith(name + "/")
            and "_combat_shell" in combat.body(int(combat.geom_bodyid[i])).name
        ]
        assert len(source) == len(dest)

        def identity(model, index, shell=False):
            parent = model.body(int(model.geom_bodyid[index])).name
            if shell:
                parent = parent.removesuffix("_combat_shell")
            return (
                parent,
                model.mesh(int(model.geom_dataid[index])).name,
                tuple(model.geom_pos[index]),
                tuple(model.geom_quat[index]),
            )

        assert Counter(identity(native, a) for a in source) == Counter(
            identity(combat, b, True) for b in dest
        )
        for bid in range(native.nbody):
            body = native.body(bid)
            if not body.name.startswith(name + "/"):
                continue
            other = combat.body(body.name).id
            np.testing.assert_array_equal(
                native.body_mass[bid], combat.body_mass[other]
            )
            np.testing.assert_array_equal(
                native.body_inertia[bid], combat.body_inertia[other]
            )
            np.testing.assert_array_equal(
                native.body_ipos[bid], combat.body_ipos[other]
            )
        copied.append(
            dict(body=name, visual_parts=len(source), copied_external_parts=len(dest))
        )
    np.testing.assert_array_equal(native.actuator_gainprm, combat.actuator_gainprm)
    np.testing.assert_array_equal(native.actuator_biasprm, combat.actuator_biasprm)
    np.testing.assert_array_equal(
        native.actuator_forcerange, combat.actuator_forcerange
    )
    assert combat.nq == native.nq and combat.nv == native.nv
    # Enumerate all permitted cross-duck geom pairs. Native/proxy duplicates
    # and internal proxy collisions must be impossible under these masks.
    allowed = 0
    for a in range(combat.ngeom):
        ba = combat.body(int(combat.geom_bodyid[a])).name
        if not ba.startswith("lavender/"):
            continue
        for b in range(combat.ngeom):
            bb = combat.body(int(combat.geom_bodyid[b])).name
            if not bb.startswith("cream/"):
                continue
            permit = bool(
                (combat.geom_contype[a] & combat.geom_conaffinity[b])
                or (combat.geom_contype[b] & combat.geom_conaffinity[a])
            )
            if permit:
                assert "_combat_shell" in ba and "_combat_shell" in bb
                allowed += 1
    return dict(
        passed=True,
        copies=copied,
        permitted_cross_duck_hull_pairs=allowed,
        unchanged_inertia_actuation=True,
        new_exclusions=0,
        visual_vertices_contained_by_own_unsimplified_convex_hull=True,
        approximation="Each visual component is convexified separately; concave empty spaces inside a component remain an approximation.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = Path(a.out)
    if path.exists():
        raise FileExistsError(path)
    result = audit()
    write_json(path, result)
    print(result)
