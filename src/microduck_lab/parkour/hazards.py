"""Finite-force machinery and freely moving props. SI units throughout."""

from dataclasses import dataclass
import math
import mujoco
import numpy as np


@dataclass(frozen=True)
class Hazard:
    name: str
    kind: str
    x: float
    y: float = 0.0
    z: float = 0.29
    half_width: float = 0.28
    speed: float = 2.2
    rotor_mass: float = 0.04
    drive_torque: float = 0.025
    phase: float = 0.0

    def add_to(self, spec):
        b = spec.worldbody.add_body(name=self.name, pos=[self.x, self.y, self.z])
        common = dict(
            name=self.name + "/geom",
            rgba=[1.0, 0.08, 0.18, 1.0],
            friction=[0.6, 0.005, 0.0001],
            solref=[0.002, 1.0],
            solimp=[0.99, 0.999, 0.0005, 0.5, 2.0],
        )
        if self.kind == "push_bar":
            # Raised pivot: the ball pushes the hanging crossbar forward/up.
            b.pos = [self.x, self.y, self.z + 0.32]
            b.add_joint(
                name=self.name + "/joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 1, 0],
                damping=0.002,
                stiffness=0.015,
            )
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, -self.half_width, -0.32, 0, self.half_width, -0.32],
                size=[0.009],
                mass=0.035,
                **common,
            )
        elif self.kind == "crate":
            b.add_freejoint(name=self.name + "/free")
            common["rgba"] = [1.0, 0.65, 0.03, 1.0]
            common["priority"] = 1
            common["friction"] = [0.25, 0.003, 0.0001]
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.065] * 3, mass=0.07, **common
            )
            eq = spec.add_equality(
                name=self.name + "/hold",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=self.name,
                name2="world",
            )
            eq.solref = [0.002, 1.0]
            spec.worldbody.add_geom(
                name=self.name + "/warning",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[self.x, self.y, 0.001],
                size=[0.10, 0.001],
                rgba=[1.0, 0.65, 0.03, 0.0],
                contype=0,
                conaffinity=0,
            )
        elif self.kind == "sweeper":
            # Spring-loaded lift lets the arm yield if a foot traps it against
            # the platform; preload balances the 40 g arm at its nominal height.
            b.add_joint(
                name=self.name + "/lift",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0, 0, 1],
                stiffness=8.0,
                springref=self.rotor_mass * 9.81 / 8.0,
                damping=2 * math.sqrt(self.rotor_mass * 8.0),
                limited=True,
                range=[-0.005, 0.15],
            )
            b.add_joint(
                name=self.name + "/joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 0, 1],
                damping=0.001,
            )
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, -self.half_width, 0, 0, self.half_width, 0],
                size=[0.014],
                mass=self.rotor_mass,
                **common,
            )
            a = spec.add_actuator(
                name=self.name + "/motor",
                target=self.name + "/joint",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
            )
            a.gainprm[0] = 1.0
            a.forcelimited = True
            a.forcerange = [-self.drive_torque, self.drive_torque]
        elif self.kind == "finish_gate":
            b.add_joint(
                name=self.name + "/joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0, 0, 1],
                limited=True,
                range=[-0.6, 0.0],
                damping=0.1,
            )
            common["rgba"] = [0.05, 0.9, 0.45, 1.0]
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.025, self.half_width, 0.30],
                mass=0.25,
                **common,
            )
            a = spec.add_actuator(
                name=self.name + "/motor",
                target=self.name + "/joint",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
            )
            a.gainprm[0] = 40.0
            a.biasprm[1] = -40.0
            a.biasprm[2] = -2.0
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.forcelimited = True
            a.forcerange = [-6.0, 6.0]
        elif self.kind == "boulder":
            b.add_freejoint(name=self.name + "/free")
            common["rgba"] = [0.65, 0.12, 1.0, 1.0]
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.25], mass=0.8, **common
            )
            for axis, q in enumerate(
                (
                    [1, 0, 0, 0],
                    [0.70710678, 0.70710678, 0, 0],
                    [0.70710678, 0, 0.70710678, 0],
                )
            ):
                b.add_geom(
                    name=self.name + f"/stripe{axis}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.2505, 0.004],
                    quat=q,
                    rgba=[0.95, 0.4, 1.0, 1.0],
                    mass=0.0,
                    contype=0,
                    conaffinity=0,
                )
        else:
            raise ValueError(self.kind)


class HazardDriver:
    def __init__(self, hazards):
        self.hazards = tuple(hazards)
        self.warnings = {}
        self.released = set()
        self.events = []

    def step(self, m, d, duck_x):
        for h in self.hazards:
            if h.kind == "crate":
                if h.name not in self.warnings and h.x - duck_x < 0.65:
                    self.warnings[h.name] = float(d.time)
                    self.events.append(
                        dict(type="HAZARD_TELEGRAPH", t=float(d.time), hazard=h.name)
                    )
                if h.name in self.warnings:
                    elapsed = d.time - self.warnings[h.name]
                    m.geom_rgba[m.geom(h.name + "/warning").id] = (
                        [1.0, 0.08, 0.18, 0.2]
                        if h.name in self.released
                        else [
                            1.0,
                            0.65,
                            0.03,
                            0.25 + 0.65 * (math.sin(6 * math.pi * elapsed) > 0),
                        ]
                    )
                if (
                    h.name in self.warnings
                    and h.name not in self.released
                    and d.time - self.warnings[h.name] >= 0.85
                ):
                    d.eq_active[m.equality(h.name + "/hold").id] = False
                    self.released.add(h.name)
                    m.geom_rgba[m.geom(h.name + "/geom").id] = [1.0, 0.08, 0.18, 1.0]
                    self.events.append(
                        dict(type="CRATE_RELEASE", t=float(d.time), hazard=h.name)
                    )
            elif h.kind == "sweeper":
                v = d.qvel[m.jnt_dofadr[m.joint(h.name + "/joint").id]]
                d.ctrl[m.actuator(h.name + "/motor").id] = np.clip(
                    0.035 * (h.speed - v), -h.drive_torque, h.drive_torque
                )
            elif h.kind == "finish_gate":
                closed = duck_x > h.x + 0.25
                d.ctrl[m.actuator(h.name + "/motor").id] = -0.6 if closed else 0.0
                if closed and h.name not in self.released:
                    self.released.add(h.name)
                    self.events.append(
                        dict(type="FINISH_GATE_CLOSE", t=float(d.time), hazard=h.name)
                    )
            elif h.kind == "boulder":
                bid = m.body(h.name).id
                adr = m.jnt_dofadr[m.joint(h.name + "/free").id]
                # A bounded physical drive force; collisions and gravity determine motion.
                d.xfrc_applied[bid, 0] = np.clip(4 * (h.speed - d.qvel[adr]), -0.5, 1.0)
