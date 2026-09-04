"""Oracle Rope Feasibility Lab — no ducks, just two ideal endpoint carriers and
an elastic-cable rope. Answers the go/no-go question from the 0904 discussion:

    Can MuJoCo form a STABLE, full-rotation rope loop (top > 27 cm, bottom
    < 2 cm, 10+ revolutions) from endpoints that stay within the Microduck
    mouth-reachable region (h ≈ 18 cm)?

Physics notes (MuJoCo, verified against the XML reference):
- The rope is `mujoco.elasticity.cable` — an inextensible 1-D elastic cable
  with independent bend/twist stiffness (NOT the hand-rolled ball-joint
  chain). This is the rope model MuJoCo intends for ropes/cables.
- Air drag/viscosity are OFF by default (option density=0, viscosity=0) — the
  earlier "air damping kills it" claim was wrong; dissipation is from joint
  damping, bend damping, and contacts.
- Endpoints are mocap carriers (non-colliding): an idealized firm grip. The
  duck-feasibility question (can a duck produce this wrench?) is answered
  separately by the wrench budget, not here.

Morphology metrics follow the 0904 doc §30: rotation angle θ, angular velocity
ω, belly radius R, plane error, and top/bottom z *near the jumper x* (not a
global max — a whip to 30 cm at one end is not the loop clearing the head).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np


# --------------------------------------------------------------------- world
def build_oracle_world(length: float = 0.62, count: int = 41, radius: float = 0.003,
                       density: float = 200.0, bend: float = 1e3, twist: float = 1e4,
                       damping: float = 1e-4, sep: float = 0.50, height: float = 0.18,
                       floor_friction: float = 0.05) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Two mocap carriers at (±sep/2, 0, height) + an elastic cable hung between.

    `density` is the rope material density (kg/m^3); rope mass ≈
    density * π r² L. For radius 3 mm and L 0.62: density 200 → ~0.35 g/m·…
    (we sweep it; the doc wants 1–8 g total).
    """
    pA = np.array([-sep / 2, 0.0, height])
    pB = np.array([sep / 2, 0.0, height])
    xml = f"""
<mujoco model="oracle_rope">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <body name="carryA" pos="{pA[0]} {pA[1]} {pA[2]}" mocap="true">
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 0 0 0.3"/>
    </body>
    <body name="carryB" pos="{pB[0]} {pB[1]} {pB[2]}" mocap="true">
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="0 0 1 0.3"/>
    </body>
    <!-- The composite cable PINS its first body to the world (measured:
         B_first has no joint). To hold BOTH ends on the carriers, the cable
         rides on a freejoint wrapper body `ropefreeA` (needs a small geom for
         mass — a massless free body falls through the connect constraint). -->
    <body name="ropefreeA" pos="{pA[0]} {pA[1]} {pA[2]}">
      <freejoint/>
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 1 0 0.0"/>
      <composite type="cable" curve="s" count="{count} 1 1" size="{length}"
                 offset="0 0 0" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{twist}"/>
          <config key="bend" value="{bend}"/>
        </plugin>
        <joint kind="main" damping="{damping}"/>
        <geom type="capsule" size="{radius}" density="{density}"
              rgba="0.9 0.6 0.1 1" friction="{floor_friction} 0.005 0.0001"
              contype="1" conaffinity="1"/>
        <skin rgba="0.9 0.6 0.1 1" inflate="0.001"/>
      </composite>
    </body>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.3 0.2 1"/>
  </worldbody>
  <equality>
    <connect body1="ropefreeA" body2="carryA" anchor="0 0 0"/>
    <connect body1="B_last" body2="carryB" anchor="0 0 0"/>
  </equality>
</mujoco>
"""
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def rope_bodies(model: mujoco.MjModel) -> list[int]:
    """Body ids of the cable segments (exclude the two carriers and world)."""
    out = []
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith("B_") and nm not in ("B_first", "B_last"):
            out.append(b)
    # include the endpoint segment bodies too (first/last are the tips)
    for nm0 in ("B_first", "B_last"):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm0)
        if b >= 0:
            out.append(b)
    return sorted(set(out))


# --------------------------------------------------------------- morphology
@dataclass
class RopeMorphology:
    """Frame-wise rope shape + rotation state, measured near the jumper x.

    The rope spans x; it rotates about the x-axis (the line between turners).
    We track the belly point (the cable midpoint region) in the y-z plane and
    unwrap its angle to get θ and ω. Radius R = distance from the x-axis line
    (at the belly's y-z) — the loop's inflation.
    """
    # history
    theta: float = 0.0          # unwrapped rotation angle of the belly (rad)
    theta_prev_raw: float = 0.0
    n_wraps: int = 0
    omega: float = 0.0          # low-passed dθ/dt (rad/s)
    belly_radius: float = 0.0   # distance of belly from the endpoint axis (m)
    top_z: float = 0.0          # max rope z near jumper x
    bottom_z: float = 0.0       # min rope z near jumper x
    plane_error: float = 0.0    # std of rope x-positions (planarity; low=planar)
    revolutions: float = 0.0    # theta / 2π

    def update(self, data: mujoco.MjData, body_ids: list[int], dt: float,
               jumper_x: float = 0.0, axis_y: float = 0.0, axis_z: float = 0.18):
        pts = np.array([data.xpos[b] for b in body_ids])   # (N,3)
        if len(pts) == 0:
            return
        # restrict to the segment near the jumper's x (the middle of the span)
        near = pts[np.abs(pts[:, 0] - jumper_x) < 0.10]
        if len(near) < 3:
            near = pts
        # belly = the lowest point near the jumper (the part that sweeps the floor)
        belly = near[np.argmin(near[:, 2])]
        # rotation of the belly about the endpoint axis (y=axis_y, z=axis_z)
        dy = belly[1] - axis_y
        dz = belly[2] - axis_z
        raw = math.atan2(dy, -dz)          # 0 = straight down, unwrap for rotation
        # unwrap (raw and theta_prev_raw are both in [-π,π], so a single
        # branch handles the wrap — a `while` here hangs forever because the
        # condition operands don't change inside the loop)
        d_ang = raw - self.theta_prev_raw
        if d_ang > math.pi:
            self.n_wraps -= 1
        elif d_ang < -math.pi:
            self.n_wraps += 1
        self.theta_prev_raw = raw
        new_theta = raw + self.n_wraps * 2 * math.pi
        dtheta = new_theta - self.theta
        self.theta = new_theta
        self.revolutions = self.theta / (2 * math.pi)
        inst_omega = dtheta / dt
        self.omega = 0.9 * self.omega + 0.1 * inst_omega
        self.belly_radius = float(math.hypot(dy, dz))
        self.top_z = float(np.max(near[:, 2]))
        self.bottom_z = float(np.min(near[:, 2]))
        self.plane_error = float(np.std(pts[:, 0]))


# --------------------------------------------------------------- trajectory
@dataclass
class EndpointTrajectory:
    """Fourier-series endpoint motion in the y-z plane + phase difference.

    Per the 0904 doc §13: don't assume circles. Each endpoint:
        y(φ) = a1 cosφ + a2 cos2φ + a3 sin2φ
        z(φ) = h + b1 sinφ + b2 sin2φ + b3 cos2φ
    Endpoint B uses φ + dphi (the phase difference Δφ_AB — a key search dim).
    """
    h: float = 0.18
    a1: float = 0.03            # primary lateral throw
    b1: float = 0.03            # primary vertical throw
    a2: float = 0.0
    a3: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    dphi: float = 0.0           # phase difference between the two endpoints

    def pos(self, phi: float, center: np.ndarray) -> np.ndarray:
        y = (self.a1 * math.cos(phi) + self.a2 * math.cos(2 * phi)
             + self.a3 * math.sin(2 * phi))
        z = (self.h + self.b1 * math.sin(phi) + self.b2 * math.sin(2 * phi)
             + self.b3 * math.cos(2 * phi))
        return center + np.array([0.0, y, z - center[2]])


class OracleDriver:
    """Drives the two endpoint carriers along an EndpointTrajectory."""

    def __init__(self, model, data, name_a="carryA", name_b="carryB"):
        self.model, self.data = model, data
        self.body_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_a)
        self.body_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_b)
        self.mocap_a = int(model.body_mocapid[self.body_a])
        self.mocap_b = int(model.body_mocapid[self.body_b])

    def place(self, phi: float, traj: EndpointTrajectory,
              center_a: np.ndarray, center_b: np.ndarray):
        self.data.mocap_pos[self.mocap_a] = traj.pos(phi, center_a)
        self.data.mocap_pos[self.mocap_b] = traj.pos(phi + traj.dphi, center_b)


class PhaseTrackingTurner:
    """Swing-up + sustain rope driver: endpoints trace circles whose phase
    TRACKS the measured rope belly angle with a +90° lead (pump energy in the
    direction of motion — like pumping a swing). The lead angle is the energy
    throttle; amplitude ramps up during spin-up.

    This is the drive that makes the elastic cable form a stable rotating loop
    (measured: sustained ~1.2 Hz, top 35 cm, bottom ~1-2 cm at L=0.82). A naive
    fixed-frequency circle drive never inflates the loop.
    """

    def __init__(self, model, data, body_ids, axis_z=0.18, name_a="carryA", name_b="carryB"):
        self.model, self.data = model, data
        self.ids = body_ids
        self.axis_z = axis_z
        self.driver = OracleDriver(model, data, name_a, name_b)
        self.wraps = 0
        self.prev_theta = None
        self.cont = 0.0

    def belly_theta(self) -> float:
        """Continuous (unwrapped) belly angle about the endpoint x-axis."""
        pts = np.array([self.data.xpos[b] for b in self.ids])
        belly = pts[np.argmin(pts[:, 2])]
        th = math.atan2(belly[1], -(belly[2] - self.axis_z))
        if self.prev_theta is None:
            self.prev_theta = th
        d = th - self.prev_theta
        if d > math.pi:
            self.wraps -= 1
        elif d < -math.pi:
            self.wraps += 1
        self.prev_theta = th
        self.cont = th + self.wraps * 2 * math.pi
        return self.cont

    def step(self, i_step: int, dt: float, cA: np.ndarray, cB: np.ndarray,
             spin_amp_max: float = 0.045, sustain_amp: float = 0.045,
             lead: float = math.pi / 2, dphi: float = 0.0):
        cont = self.belly_theta()
        # amplitude ramps up during spin-up (|cont| < 1 rev), then holds
        if abs(cont) < 2 * math.pi:
            r = min(spin_amp_max, 0.012 + i_step * 0.000004)
        else:
            r = sustain_amp
        phi = cont + lead
        traj = EndpointTrajectory(h=self.axis_z, a1=r, b1=r, dphi=dphi)
        self.driver.place(phi, traj, cA, cB)
        return cont
