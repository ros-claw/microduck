"""Own-state dynamic entry envelope; rejected entries retain trained braking.

These are development bounds, not a claim of new learned transition weights.
No enemy phase or action enters the contract.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class MotorContext:
    yaw_rate: float
    joint_speed_max: float
    joint_position_max: float
    last_action_max: float
    foot_normal_N: float
    finite: bool


def measure_motor_context(duck, audit):
    import mujoco

    d, m = duck.data, duck.model
    load, force = 0.0, np.zeros(6)
    for i in range(d.ncon):
        contact = d.contact[i]
        a, b = int(contact.geom1), int(contact.geom2)
        if audit.stage not in (a, b):
            continue
        foot = audit.feet.get(a, audit.feet.get(b))
        if foot != duck.prefix.rstrip("/"):
            continue
        mujoco.mj_contactForce(m, d, i, force)
        normal_z = float(contact.frame[2]) * (1 if b in audit.feet else -1)
        load += max(0.0, float(force[0])) * max(0.0, normal_z)
    # Root angular velocity is read from the physical trunk, never overwritten.
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, duck.trunk_body_id, vel, 0)
    q, qd, action = (
        d.qpos[duck.joint_qpos_idx],
        d.qvel[duck.joint_qvel_idx],
        duck.last_action,
    )
    return MotorContext(
        float(vel[2]),
        float(np.max(np.abs(qd))),
        float(np.max(np.abs(q))),
        float(np.max(np.abs(action))),
        load,
        bool(np.isfinite(np.r_[vel, q, qd, action, load]).all()),
    )


def evaluate_entry(body, context):
    if context is None:
        return dict(accepted=False, reasons=["missing_own_motor_context"])
    conditions = dict(
        finite=context.finite,
        supported=body.foot_support and context.foot_normal_N > 0.5,
        upright=body.upright >= 0.96 and not body.body_ground,
        speed=body.speed <= 0.48,
        yaw_rate=abs(context.yaw_rate) <= 2.0,
        joints=context.joint_speed_max <= 20.0 and context.joint_position_max <= 3.2,
        history=context.last_action_max <= 2.5,
        head=abs(body.neck_pitch_delta) <= 0.5 and abs(body.head_pitch_delta) <= 0.5,
        boundary=body.boundary_distance >= 0.12,
    )
    return dict(
        accepted=all(conditions.values()),
        reasons=[k for k, v in conditions.items() if not v],
        measured=dict(
            **asdict(context),
            speed_m_s=body.speed,
            upright=body.upright,
            boundary_m=body.boundary_distance,
        ),
    )


def evaluate_continuation(body, context, unsupported_s):
    """Walking has brief unloaded gait phases; entry and continuation differ.

    Bounds are development candidates, evaluated separately from numerical gates.
    No entry with missing support is permitted by this sustain contract.
    """
    if context is None:
        return dict(accepted=False, reasons=["missing_own_motor_context"])
    conditions = dict(
        finite=context.finite,
        upright=body.upright >= 0.94 and not body.body_ground,
        gait_support=unsupported_s <= 0.12,
        speed=body.speed <= 0.55,
        yaw_rate=abs(context.yaw_rate) <= 4.0,
        joints=context.joint_speed_max <= 25.0,
        history=context.last_action_max <= 2.5,
        boundary=body.boundary_distance >= 0.10,
    )
    return dict(
        accepted=all(conditions.values()),
        reasons=[k for k, v in conditions.items() if not v],
        measured=dict(
            **asdict(context), unsupported_s=unsupported_s, upright=body.upright
        ),
    )
