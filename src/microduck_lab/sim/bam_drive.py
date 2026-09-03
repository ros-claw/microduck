"""BAM actuator drive — deploy policies against the SAME actuator physics they
were trained with (the bam XL330 M6 voltage model), inside plain MuJoCo.

Why: the training stack (mjlab) uses BAM; quasi-static policies (walk/stand)
transfer to plain position actuators fine, but explosive maneuvers (the jump)
learn to exploit BAM's exact lag/saturation and fail on stiff position servos.
testbench_sim2real.py shows bam's MujocoController drives a vanilla MuJoCo
loop — we adopt it per duck.

Per duck: convert its 14 position actuators to torque motors, zero joint
friction/damping (BAM owns friction), then MujocoController writes torques.
"""

from __future__ import annotations

import mujoco
import numpy as np

VIN = 7.4          # XL330 supply voltage (testbench default)
KP_FW = 200.0      # firmware position kp used by the BAM controller


def _load_bam_model():
    import json as _json
    from bam.actuators import actuators as bam_actuators
    from bam.model import models as bam_models, _resolve_json_path

    with open(_resolve_json_path(None, "xl330", "m6")) as f:
        params = _json.load(f)
    m = bam_models["m6"]()
    m.set_actuator(bam_actuators["xl330"]())
    m.actuator.kp = KP_FW
    m.actuator.vin = VIN
    m.load_parameters_from_dict(params)
    return m


class BamDrive:
    """Owns one duck's 14 servos through BAM voltage dynamics."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str,
                 actuator_names: list[str]):
        from bam.mujoco import MujocoController

        self.names = [f"{prefix}{n}" for n in actuator_names]
        # convert the duck's position actuators to torque motors (BAM owns
        # the voltage→torque dynamics + friction)
        kt = None
        for full in self.names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
            assert aid >= 0, full
            model.actuator_gainprm[aid, 0] = 1.0          # pure motor
            model.actuator_biasprm[aid, :] = 0.0
            jid = int(model.actuator_trnid[aid, 0])
            dof = int(model.jnt_dofadr[jid])
            model.dof_damping[dof] = 0.0
            model.dof_frictionloss[dof] = 0.0
        bam_model = _load_bam_model()
        kt = bam_model.kt.value
        R = bam_model.R.value
        fl = VIN * kt / R
        for full in self.names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
            model.actuator_forcerange[aid, 0] = -fl
            model.actuator_forcerange[aid, 1] = fl
            model.actuator_forcelimited[aid] = 1
        mujoco.mj_setConst(model, data)

        self.ctrl = MujocoController(bam_model, self.names, model, data)

    def drive(self, targets: np.ndarray):
        """targets: 14 absolute joint position goals (radians)."""
        for name, t in zip(self.names, targets):
            self.ctrl.set_q_target(name, float(t))
        self.ctrl.update()
