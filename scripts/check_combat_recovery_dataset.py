"""Independently compare each extracted record against its original tape.

BODY object velocity is COM-centered in MuJoCo. Free-joint qvel[:3] is the
linear velocity required by the link-origin training reset API.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.combat.world import CombatBank, digest
from microduck_lab.combat.episode import write_json
from microduck_lab.sim.runtime import DuckRuntime

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--states", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
meta = json.loads(Path(a.states).with_suffix(".json").read_text())
data = np.load(a.states)
assert digest(a.states) == meta["states_sha256"]
groups = defaultdict(list)
for i, row in enumerate(meta["rows"]):
    groups[row["source_run"]].append((i, row))
errors = defaultdict(float)
maximum_com_difference = 0.0
for directory, rows in groups.items():
    directory = Path(directory)
    m = mujoco.MjModel.from_binary_path(str(directory / "scene.mjb"))
    d = mujoco.MjData(m)
    states = np.load(directory / "states.npy", mmap_mode="r")
    controls = np.load(directory / "controls.npz")["ctrl"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert digest(directory / "manifest.json") == rows[0][1]["source_manifest_sha256"]
    for i, row in rows:
        duck = DuckRuntime(
            m, d, CombatBank({}), prefix=row["actor"] + "/", name=row["actor"]
        )
        mujoco.mj_setState(m, d, states[row["source_step"]], manifest["state_spec"])
        mujoco.mj_forward(m, d)
        q, v = duck.trunk_qpos_adr, duck.trunk_qvel_adr
        angular = d.xmat[duck.trunk_body_id].reshape(3, 3) @ d.qvel[v + 3 : v + 6]
        expected = dict(
            raw_root=d.qpos[q : q + 7],
            raw_world_velocity=np.r_[d.qvel[v : v + 3], angular],
            joints=d.qpos[duck.joint_qpos_idx],
            joint_velocity=d.qvel[duck.joint_qvel_idx],
            last_action=controls[round(row["time_s"] * 50) - 1, duck.act_ids]
            - duck.default_pose,
        )
        for key, value in expected.items():
            errors[key] = max(errors[key], float(np.abs(data[key][i] - value).max()))
        com = np.empty(6)
        mujoco.mj_objectVelocity(
            m, d, mujoco.mjtObj.mjOBJ_BODY, duck.trunk_body_id, com, 0
        )
        maximum_com_difference = max(
            maximum_com_difference, float(np.linalg.norm(com[3:] - d.qvel[v : v + 3]))
        )
assert max(errors.values()) < 1e-10, errors
write_json(
    a.out,
    dict(
        passed=True,
        states=len(meta["rows"]),
        source_trajectories=len(groups),
        states_sha256=digest(a.states),
        maximum_absolute_errors=dict(errors),
        maximum_com_vs_link_velocity_difference_m_s=maximum_com_difference,
        previous_action="Exactly the preceding executed 14 servo offsets; never reset to zero",
        initial_velocity="Free-joint origin linear qvel; body-local angular qvel rotated to world",
        checker_sha256=digest(__file__),
    ),
)
print(
    json.dumps(
        dict(
            passed=True, states=len(meta["rows"]), maximum_absolute_errors=dict(errors)
        ),
        indent=2,
    )
)
