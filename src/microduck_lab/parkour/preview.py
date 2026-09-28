"""Short-horizon model-based skill preview from the current simulation state.

This is privileged-state simulation planning, not vision or Jev inference.
Only cloned data are advanced. The live robot is never reset to a prediction.
Predictions are advisory; full-rate contact/replay gates remain independent.
"""

import copy
import time
import numpy as np
import mujoco
from .jump import CandidateJump, GapAudit
from .world import foot_support
from .skills import lane_command


def preview_jump(
    m,
    d,
    r,
    far_edge,
    horizon=2.9,
    prepare_s=0.0,
    forward_speed=0.45,
    brake_s=0.0,
    settle_s=0.0,
):
    started = time.perf_counter()
    predicted = copy.copy(d)
    runtime = copy.copy(r)
    runtime.data = predicted
    runtime.bank = copy.copy(r.bank)
    for name in ("last_action", "command", "_jv_prev", "default_pose"):
        setattr(runtime, name, getattr(r, name).copy())
    runtime.bank.mirror_run = False
    runtime.joint_vel_delay = 0
    for _ in range(round(prepare_s / 0.02)):
        runtime.active_policy = "run"
        runtime.set_command()
        runtime.command[:3] = lane_command(runtime, 0.0, forward_speed)
        runtime.step()
        mujoco.mj_step(m, predicted, 100)
    for policy, seconds in [("run", brake_s), ("stand", settle_s)]:
        for _ in range(round(seconds / 0.02)):
            runtime.active_policy = policy
            runtime.set_command()
            if policy == "run":
                runtime.command[2] = np.clip(-4 * runtime.trunk_yaw(), -2.5, 2.5)
            runtime.step()
            mujoco.mj_step(m, predicted, 100)
    launch_position = runtime.trunk_pos().copy()
    floor_id = m.geom("floor/0").id
    near_edge = float(predicted.geom_xpos[floor_id, 0] + m.geom_size[floor_id, 0])
    if not near_edge - 0.16 <= launch_position[0] <= near_edge - 0.02:
        return dict(
            passed=False,
            prepare_s=prepare_s,
            forward_speed=forward_speed,
            brake_s=brake_s,
            settle_s=settle_s,
            launch_position=launch_position.tolist(),
            status="OUTSIDE_LAUNCH_WINDOW",
            min_upright=float(predicted.xmat[r.trunk_body_id, 8]),
            max_abs_y_m=abs(float(launch_position[1])),
            final_position=launch_position.tolist(),
            predicted_duration_s=float(predicted.time - d.time),
            wall_ms=1000 * (time.perf_counter() - started),
        )
    jump = CandidateJump(float(predicted.time), far_edge, landing_tolerance=0.19)
    audit = GapAudit()
    audit.sample(m, predicted, runtime)
    min_up = 1.0
    max_side = abs(float(runtime.trunk_pos()[1]))
    for _ in range(round(horizon / 0.02)):
        jump.update(m, predicted, runtime, audit)
        runtime.step()
        # Known currently applied machinery forces are held over this short
        # gap-local forecast. No future hazard event or seed is inspected.
        mujoco.mj_step(m, predicted, 100)
        audit.sample(m, predicted, runtime)
        min_up = min(min_up, float(predicted.xmat[r.trunk_body_id, 8]))
        max_side = max(max_side, abs(float(runtime.trunk_pos()[1])))
        if jump.status != "RUNNING" or runtime.trunk_pos()[2] < -0.10:
            break
    passed = bool(
        jump.status == "SUCCESS"
        and runtime.trunk_pos()[0] > far_edge + 0.015
        and runtime.trunk_pos()[2] > 0.10
        and min_up > 0.70
        and audit.crossed
        and max_side < 0.20
        and foot_support(m, predicted, "floor/1")
    )
    return dict(
        passed=passed,
        prepare_s=prepare_s,
        forward_speed=forward_speed,
        brake_s=brake_s,
        settle_s=settle_s,
        launch_position=launch_position.tolist(),
        status=jump.status,
        min_upright=min_up,
        max_abs_y_m=max_side,
        final_position=runtime.trunk_pos().tolist(),
        predicted_duration_s=float(predicted.time - d.time),
        wall_ms=1000 * (time.perf_counter() - started),
    )


class GapSequence:
    """Execute the same motor-only approach evaluated in the preview."""

    name = "JUMP_CENTER"
    status = "RUNNING"

    def __init__(self, started, far_edge, plan, position):
        self.started = started
        self.far_edge = far_edge
        self.plan = plan
        self.launch_position = position.copy()
        self.tick = 0
        self.jump = None
        self.events = []
        self.phase = "APPROACH"

    def update(self, m, d, r, audit):
        advance = round(self.plan["prepare_s"] / 0.02)
        brake = round(self.plan["brake_s"] / 0.02)
        settle = round(self.plan["settle_s"] / 0.02)
        r.bank.mirror_run = False
        if self.tick < advance + brake + settle:
            r.joint_vel_delay = 0
            r.set_command()
            if self.tick < advance:
                self.phase = "APPROACH"
                r.active_policy = "run"
                r.command[:3] = lane_command(r, 0.0, self.plan["forward_speed"])
            elif self.tick < advance + brake:
                self.phase = "BRAKE"
                r.active_policy = "run"
                r.command[2] = np.clip(-4 * r.trunk_yaw(), -2.5, 2.5)
            else:
                self.phase = "SETTLE"
                r.active_policy = "stand"
            self.tick += 1
            return self.status
        if self.jump is None:
            self.jump = CandidateJump(
                float(d.time), self.far_edge, landing_tolerance=0.19
            )
            self.events.append(
                dict(type="SKILL_START", skill=self.name, t=float(d.time))
            )
        self.status = self.jump.update(m, d, r, audit)
        self.phase = self.jump.phase
        if self.status != "RUNNING":
            self.events.extend(self.jump.events)
        return self.status


def choose_gap_sequence(m, d, r, far_edge):
    """Bounded candidate set; stop at first acceptable physical prediction."""
    trials = []
    candidates = [
        (0.16, 0.4, 0.2),
        (0.08, 0.4, 0.2),
        (0.24, 0.4, 0.2),
        (0.08, 0.2, 0.5),
        (0.24, 0.4, 0.5),
        (0.32, 0.4, 0.2),
        (0.40, 0.4, 0.2),
        (0.0, 0.6, 0.2),
    ]
    longer = [(0.56, 0.4, 0.2), (0.64, 0.4, 0.2), (0.48, 0.4, 0.2), (0.72, 0.4, 0.2)]
    candidates = (
        longer + candidates if r.trunk_linvel()[0] < 0.2 else candidates + longer
    )
    for advance, brake, settle in candidates:
        result = preview_jump(
            m,
            d,
            r,
            far_edge,
            prepare_s=advance,
            forward_speed=0.45,
            brake_s=brake,
            settle_s=settle,
        )
        trials.append(result)
        if result["passed"]:
            return result, trials
    return None, trials


def choose_sweeper_route(m, d, r, driver, sweeper_x):
    """Compare measured-state lane routes; account for the moving finite-torque arm."""
    trials = []
    preferred = -1 if r.trunk_pos()[1] < 0 else 1
    for side, offset in [
        (preferred, 0.37),
        (-preferred, 0.37),
        (preferred, 0.30),
        (-preferred, 0.30),
    ]:
        started = time.perf_counter()
        model = copy.copy(m)
        predicted = copy.copy(d)
        runtime = copy.copy(r)
        runtime.model, runtime.data = model, predicted
        runtime.bank = copy.copy(r.bank)
        for name in ("last_action", "command", "_jv_prev", "default_pose"):
            setattr(runtime, name, getattr(r, name).copy())
        machinery = copy.deepcopy(driver)
        target = side * offset
        min_up = 1.0
        touched = False
        impulse = 0.0
        sweep_id = model.geom("sweeper/geom").id
        duck_ids = {
            g
            for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name.startswith("duck/")
        }
        force = np.zeros(6)
        for _ in range(225):
            runtime.bank.mirror_run = side < 0
            runtime.joint_vel_delay = 0
            runtime.active_policy = "run"
            runtime.set_command()
            runtime.command[:3] = sweeper_command(runtime, target)
            machinery.step(model, predicted, float(runtime.trunk_pos()[0]))
            runtime.step()
            for _ in range(100):
                mujoco.mj_step(model, predicted)
                for i, c in enumerate(predicted.contact):
                    if (c.geom1 == sweep_id and c.geom2 in duck_ids) or (
                        c.geom2 == sweep_id and c.geom1 in duck_ids
                    ):
                        mujoco.mj_contactForce(model, predicted, i, force)
                        touched |= bool(force[0] > 1e-5)
                        impulse += max(0.0, float(force[0])) * model.opt.timestep
            min_up = min(min_up, float(predicted.xmat[r.trunk_body_id, 8]))
            if (
                runtime.trunk_pos()[0] > sweeper_x + 0.4
                or min_up < 0.8
                or runtime.trunk_pos()[2] < 0.06
            ):
                break
        passed = bool(
            not touched
            and min_up > 0.9
            and runtime.trunk_pos()[0] > sweeper_x + 0.4
            and runtime.trunk_pos()[2] > 0.10
        )
        result = dict(
            target_y=target,
            passed=passed,
            contact=bool(touched),
            impulse_Ns=impulse,
            min_upright=min_up,
            final_position=runtime.trunk_pos().tolist(),
            wall_ms=1000 * (time.perf_counter() - started),
        )
        trials.append(result)
        if passed:
            return target, trials
    # Explicit fallback, not a fabricated safe prediction. The current lane is
    # preserved when neither candidate qualifies; the outcome remains audited.
    viable = [
        p
        for p in trials
        if p["min_upright"] > 0.9 and p["final_position"][0] > sweeper_x + 0.4
    ]
    if viable:
        return min(viable, key=lambda p: p["impulse_Ns"])["target_y"], trials
    return float(r.trunk_pos()[1]), trials


def sweeper_command(r, target):
    # Stay within the narrow landing platform until the wider recovery bay.
    waypoint = (
        float(np.clip(target, -0.18, 0.18)) if r.trunk_pos()[0] < 3.55 else target
    )
    return lane_command(
        r, waypoint, 0.55 if abs(r.trunk_pos()[1] - waypoint) > 0.08 else 0.7
    )
