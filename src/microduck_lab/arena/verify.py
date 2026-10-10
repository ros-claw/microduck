"""DG-02 semantic input replay: constraints, public choices, support and referee."""

import gzip
import json
from pathlib import Path
import mujoco
import numpy as np
from ..sim.runtime import DuckRuntime, PolicyBank
from .episode import STATE, sha
from .tiles import bind_tiles, TileScheduler
from .controller import WarningResponder, observation
from .referee import Referee
from .survival import contacts


def verify(path):
    path = Path(path)
    audit = json.loads((path / "audit.json").read_text())
    if audit["schema"] != "microduck.arena.scheduled.v1":
        raise ValueError("Unsupported schema")
    required = {
        "scene.mjb",
        "trajectory.npz",
        "contacts.jsonl.gz",
        "decisions.jsonl.gz",
    }
    if set(audit["hashes"]) != required:
        raise ValueError("Incomplete replay manifest")
    for name in required:
        if sha(path / name) != audit["hashes"][name]:
            raise ValueError("Hash mismatch: " + name)
    if audit["mujoco_version"] != mujoco.__version__:
        raise ValueError("MuJoCo mismatch")
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    with gzip.open(path / "decisions.jsonl.gz", "rt") as f:
        decisions = [json.loads(line) for line in f]
    m = mujoco.MjModel.from_binary_path(str(path / "scene.mjb"))
    d = mujoco.MjData(m)
    expected = mujoco.MjData(m)
    mujoco.mj_setState(m, d, z["states"][0], STATE)
    duck = DuckRuntime(m, d, PolicyBank({}), prefix="cream/", name="cream")
    tiles = bind_tiles(m)
    scheduler = TileScheduler(audit["seed"], mode=audit["mode"])
    controller = WarningResponder()
    referee = Referee()
    if scheduler.plan != audit["oracle_schedule"]:
        raise ValueError("Schedule differs from seed/config")
    events = []
    support = []
    qerr = verr = ferr = 0.0
    count = 0
    terminal = False
    with gzip.open(path / "contacts.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            if i >= len(z["states"]):
                raise ValueError("Extra contact frame")
            t = i * audit["dt"]
            sample = json.loads(line)
            if sample["step"] != i or abs(sample["t"] - t) > 1e-9:
                raise ValueError("Contact timestamps invalid")
            events.extend(scheduler.advance(d, tiles, t))
            if i % round(0.02 / audit["dt"]) == 0:
                if count >= len(decisions):
                    raise ValueError("Missing policy decision")
                decision = decisions[count]
                count += 1
                obs = observation(duck, tiles, support, t)
                policy, command, intent = (
                    controller.choose(obs)
                    if audit["brain"] == "warning"
                    else ("stand", (0.0, 0.0, 0.0), "HOLD")
                )
                if terminal:
                    policy, command, intent = "stand", (0.0, 0.0, 0.0), "ELIMINATED"
                actual = dict(
                    time=t,
                    input=obs,
                    policy=policy,
                    command=list(command),
                    intent=intent,
                    target=controller.target,
                )
                if actual != decision:
                    raise ValueError(
                        "Public observation or controller decision mismatch at "
                        + str(i)
                    )
            if not np.array_equal(d.eq_active, z["eq_active"][i]):
                raise ValueError("Constraint activity differs from events")
            mujoco.mj_setState(m, expected, z["states"][i], STATE)
            qerr = max(qerr, float(np.max(np.abs(d.qpos - expected.qpos))))
            verr = max(verr, float(np.max(np.abs(d.qvel - expected.qvel))))
            d.ctrl[:] = z["ctrl"][i]
            mujoco.mj_step(m, d)
            pairs, support = contacts(m, d, tiles)
            if len(pairs) != len(sample["contacts"]):
                raise ValueError("Contact count differs")
            for actual, want in zip(pairs, sample["contacts"], strict=True):
                if actual[:2] != want[:2]:
                    raise ValueError("Contact identity differs")
                ferr = max(
                    ferr, float(np.max(np.abs(np.asarray(actual[2:]) - want[2:])))
                )
            if support != sample["supporting_tiles"]:
                raise ValueError("Support graph differs")
            events.extend(
                event for tile in tiles for event in tile.update(d, t + audit["dt"])
            )
            robot = dict(
                z=float(duck.trunk_pos()[2]),
                vz=float(duck.trunk_linvel()[2]),
                supporting_tiles=support,
                contact_step=i,
            )
            new = referee.update(t + audit["dt"], {"cream": robot})
            events.extend(new)
            terminal |= bool(new)
    if i + 1 != len(z["states"]) or count != len(decisions):
        raise ValueError("Missing/extra frames or decisions")
    if events != audit["events"]:
        raise ValueError("Tile/referee events differ")
    outcome = referee.outcome(timeout=True)
    if outcome != audit["outcome"]:
        raise ValueError("Outcome differs")
    qerr = max(qerr, float(np.max(np.abs(duck.trunk_pos() - z["root"][-1, :3]))))
    result = dict(
        frames=len(z["states"]),
        decisions=count,
        qpos_max_error=qerr,
        qvel_max_error=verr,
        contact_max_error=ferr,
        events_reproduced=len(events),
        outcome=outcome,
        passed=max(qerr, verr, ferr) < 1e-10,
    )
    if not result["passed"]:
        raise ValueError(str(result))
    (path / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
