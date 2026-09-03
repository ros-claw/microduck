"""Core policy runtime for one duck inside a (possibly multi-duck) MuJoCo world.

Ported from pollen-robotics/microduck_rl `scripts/infer_policy.py` (Apache-2.0),
generalized to a namespaced duck inside a shared physics world:

- 61D observation contract: [gyro(3), projected_gravity(3), joint_pos_rel(14),
  joint_vel(14), last_action(14), command(13 = twist(3) + head_pose(4) + body_pose(6))]
- action = policy(obs); ctrl = DEFAULT_POSE + action * scale  (position actuators)
- policy rate 50 Hz, physics timestep 5 ms (4 substeps per policy step)

Override channels (used by higher-level skills, never by the policy itself):
- head_override: replaces the 4 neck/head ctrl targets (rope turning).
- leg_override: replaces the 10 leg ctrl targets (scripted jump maneuver).
Overrides are *actuation-level* and honest: the policy keeps running, physics
decides what happens. This mirrors the official runtime where `head_offset`
is added on top of ctrl[5:9] in legacy mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

# STAND2 reference pose (matches HOME_FRAME in microduck_constants.py).
# Actions are offsets from this pose; joint obs are relative to it.
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)

LEG_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]   # policy-order indices of leg joints
HEAD_IDX = [5, 6, 7, 8]                        # neck_pitch, head_pitch, head_yaw, head_roll

POLICY_HZ = 50
PHYSICS_DT = 0.005
SUBSTEPS = int(1.0 / POLICY_HZ / PHYSICS_DT)   # 4


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by the inverse of quat [w, x, y, z]."""
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


@dataclass
class PolicyBank:
    """Lazy ONNX policy registry. All policies share the 61D obs contract."""

    paths: dict[str, str]
    _sessions: dict[str, object] = field(default_factory=dict)

    def get(self, name: str):
        if name not in self._sessions:
            import onnxruntime as ort

            if name not in self.paths:
                raise KeyError(f"policy {name!r} not in bank ({sorted(self.paths)})")
            self._sessions[name] = ort.InferenceSession(
                self.paths[name], providers=["CPUExecutionProvider"]
            )
        return self._sessions[name]

    def infer(self, name: str, obs: np.ndarray) -> np.ndarray:
        sess = self.get(name)
        out = sess.run(
            [sess.get_outputs()[0].name],
            {sess.get_inputs()[0].name: obs.reshape(1, -1)},
        )[0]
        return out.squeeze(0).astype(np.float32)


XL330_CURRENT_LIMIT_A = 1.75   # firmware current limit
XL330_KT = 0.3660              # N·m/A (bam model xl330-m6) — torque = kt·I


def apply_current_limit(model: mujoco.MjModel, prefix: str = ""):
    """Clamp actuators to the XL330 firmware torque limit (kt × 1.75 A ≈ 0.64 Nm).

    infer_policy.py does this for deployment rehearsal — the training stack
    uses the BAM voltage model whose torque saturates the same way. Without
    the clamp, the position actuators deliver the XML's 0.96 Nm (50% more
    than reality) and policies trained with BAM transfer badly.
    """
    lim = XL330_KT * XL330_CURRENT_LIMIT_A
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if name and name.startswith(prefix) and not name.startswith(f"{prefix}rope"):
            model.actuator_forcerange[a, 0] = -lim
            model.actuator_forcerange[a, 1] = lim
            model.actuator_forcelimited[a] = 1


class DuckRuntime:
    """One duck's control interface into a shared MuJoCo world.

    All MuJoCo names are resolved with the duck's namespace prefix
    (e.g. "lavender/imu_ang_vel"), so N ducks coexist in one model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bank: PolicyBank,
        prefix: str = "",
        name: str = "duck",
        action_scale: float = 1.0,
    ):
        self.model = model
        self.data = data
        self.bank = bank
        self.prefix = prefix
        self.name = name
        self.action_scale = action_scale

        m = model
        # --- actuators (14, policy order) ---
        self.act_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{n}")
             for n in ACTUATOR_NAMES],
            dtype=int,
        )
        assert (self.act_ids >= 0).all(), f"{name}: missing actuators"
        # qpos/qvel addresses of the actuated joints (via actuator transmission)
        self.joint_qpos_idx = np.array(
            [int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in self.act_ids], dtype=int
        )
        self.joint_qvel_idx = np.array(
            [int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in self.act_ids], dtype=int
        )

        # --- sensors ---
        self.imu_ang_vel_adr = self._sensor_adr("imu_ang_vel")
        self.trunk_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}trunk_base"
        )
        assert self.trunk_body_id >= 0

        trunk_jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}trunk_base_freejoint"
        )
        self.trunk_qpos_adr = int(m.jnt_qposadr[trunk_jid])
        self.trunk_qvel_adr = int(m.jnt_dofadr[trunk_jid])

        # --- state ---
        self.default_pose = DEFAULT_POSE.copy()
        self.last_action = np.zeros(14, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)  # unified command block
        self.active_policy = "stand"

        # actuation-level overrides (policy still runs; these replace ctrl slots)
        self.head_override: np.ndarray | None = None   # absolute joint targets (4)
        self.leg_override: np.ndarray | None = None    # absolute joint targets (10)

        # BAM voltage-model drive (optional, the faithful deployment path):
        # when set, ctrl targets go through bam's XL330 M6 dynamics instead of
        # the plain position servos. Enable for explosive trained policies.
        self.bam_drive = None

        # Training-feeds-delayed-joint_vel: the velocity/standup/jump envs delay
        # joint_vel by exactly 1 policy step (delay_min_lag=delay_max_lag=1).
        # Deploying with CURRENT joint_vel is out-of-distribution for policies
        # trained that way — keep a 1-step buffer.
        self.joint_vel_delay = 0   # infer_policy.py deploys with NO obs delay
        self._jv_prev = np.zeros(14, dtype=np.float32)

    # ------------------------------------------------------------------ util
    def _sensor_adr(self, short: str) -> int:
        sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{self.prefix}{short}"
        )
        assert sid >= 0, f"{self.name}: sensor {short!r} missing"
        return int(self.model.sensor_adr[sid])

    # ------------------------------------------------------------ perception
    def base_ang_vel(self) -> np.ndarray:
        return self.data.sensordata[self.imu_ang_vel_adr:self.imu_ang_vel_adr + 3].astype(np.float32)

    def projected_gravity(self) -> np.ndarray:
        quat = self.data.xquat[self.trunk_body_id]
        return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])).astype(np.float32)

    def trunk_quat(self) -> np.ndarray:
        return self.data.xquat[self.trunk_body_id].copy()

    def trunk_pos(self) -> np.ndarray:
        return self.data.qpos[self.trunk_qpos_adr:self.trunk_qpos_adr + 3].copy()

    def trunk_yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[self.trunk_qpos_adr + 3:self.trunk_qpos_adr + 7]
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def trunk_linvel(self) -> np.ndarray:
        return self.data.qvel[self.trunk_qvel_adr:self.trunk_qvel_adr + 3].copy()

    def is_upright(self, cos_thresh: float = 0.55) -> bool:
        return -self.projected_gravity()[2] > cos_thresh

    def site_pos(self, short: str) -> np.ndarray:
        sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, f"{self.prefix}{short}"
        )
        return self.data.site_xpos[sid].copy()

    # --------------------------------------------------------------- control
    def set_command(self, twist=(0, 0, 0), head=(0, 0, 0, 0), body=(0, 0, 0, 0, 0, 0)):
        """Write the unified 13D command block fed into the policy obs."""
        self.command[:] = np.concatenate([twist, head, body]).astype(np.float32)

    def get_obs(self) -> np.ndarray:
        jp = self.data.qpos[self.joint_qpos_idx] - self.default_pose
        jv_now = self.data.qvel[self.joint_qvel_idx].astype(np.float32)
        if self.joint_vel_delay > 0:
            jv = self._jv_prev
            self._jv_prev = jv_now
        else:
            jv = jv_now
        return np.concatenate([
            self.base_ang_vel(),
            self.projected_gravity(),
            jp.astype(np.float32),
            jv.astype(np.float32),
            self.last_action,
            self.command,
        ]).astype(np.float32)

    def step(self):
        """One 50 Hz policy step: infer, apply, advance physics 4 substeps."""
        obs = self.get_obs()
        assert obs.shape == (61,), obs.shape
        action = self.bank.infer(self.active_policy, obs)
        self.last_action = action.copy()
        target = self.default_pose + action * self.action_scale
        if self.head_override is not None:
            target[HEAD_IDX] = self.head_override
        if self.leg_override is not None:
            target[LEG_IDX] = self.leg_override
        if self.bam_drive is not None:
            self.bam_drive.drive(target)   # BAM writes torques into data.ctrl
        else:
            self.data.ctrl[self.act_ids] = target


ACTUATOR_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
