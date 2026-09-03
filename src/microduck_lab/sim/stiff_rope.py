"""Mocap-ended stiff rope — a *well-behaved* rotating loop.

Physics honesty notes (declared): the rope is a composite cable with mild bend
stiffness (a real beaded/trick rope), and its two ends are carried by mocap
bodies pinned to the turners' mouths — i.e. the turners hold the rope FIRMLY
and are not dragged by it (in reality a duck would brace; modelling the grip
as rigid is the idealization, disclosed). Everything the JUMPER interacts with
is fully physical: real rope bodies, real contacts, real tripping.

Why this exists: a free floppy 0.5 g cable driven by 2 cm mouth circles is
chaotic — the loop under-inflates, migrates, and coils (see
docs/physics-notes.md). A slightly stiff rope with firm end grips forms the
clean planar loop that real skipping ropes form.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def build_stiff_rope_spec(length: float, count: int, radius: float, density: float,
                          color, pA, pB, bend: float = 1e3) -> mujoco.MjSpec:
    """A composite cable (elasticity plugin = mild bend stiffness) spawned as a
    straight horizontal rope from pA to pB along +X. Ends: `rope/A` and
    `rope/B` mocap carrier bodies; the cable's first/last composite bodies are
    welded to them (connect equality, zero-gap at spawn)."""
    r, g, b, a = color
    xml = f"""
<mujoco model="rope">
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <body name="A" pos="{pA[0]} {pA[1]} {pA[2]}" mocap="true">
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 0 0 0.3"/>
    </body>
    <body name="B" pos="{pB[0]} {pB[1]} {pB[2]}" mocap="true">
      <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="0 0 1 0.3"/>
    </body>
    <composite type="cable" curve="s" count="{count} 1 1" size="{length}"
               offset="{pA[0]} {pA[1]} {pA[2]}" initial="none">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="1e4"/>
        <config key="bend" value="{bend}"/>
      </plugin>
      <joint kind="main" damping="0.0005"/>
      <geom type="capsule" size="{radius}" density="{density}"
            rgba="{r} {g} {b} {a}" friction="0.3 0.02 0.001" contype="1" conaffinity="1"/>
      <skin rgba="{r} {g} {b} {a}" inflate="0.0015"/>
    </composite>
  </worldbody>
  <equality>
    <connect body1="B_first" body2="A" anchor="0 0 0"/>
    <connect body1="B_last" body2="B" anchor="0 0 0"/>
  </equality>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


class StiffRope:
    """Drives the two mocap rope-end carriers along synchronized circles.

    Works on the composed world's custom serial-chain rope (rope/seg_*) with
    mocap carriers (rope/carryA, rope/carryB) connected to the rope ends.
    """

    def __init__(self, model, data, name_a="rope/carryA", name_b="rope/carryB"):
        self.model, self.data = model, data
        self.body_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_a)
        self.body_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_b)
        self.mocap_a = int(model.body_mocapid[self.body_a])
        self.mocap_b = int(model.body_mocapid[self.body_b])
        assert self.mocap_a >= 0 and self.mocap_b >= 0, "carriers must be mocap"
        self.phase = 0.0
        # rope segment bodies (the custom chain)
        self.rope_body_ids = [b for b in range(model.nbody)
                              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("rope/seg")]

    def drive(self, phase: float, radius: float, center_a: np.ndarray, center_b: np.ndarray,
              freq: float, dt: float):
        """Move both mocap ends along synchronized circles in the Y-Z plane."""
        self.phase = (phase + 2 * math.pi * freq * dt) % (2 * math.pi)
        for mid, c in ((self.mocap_a, center_a), (self.mocap_b, center_b)):
            self.data.mocap_pos[mid] = c + np.array([0.0, radius * math.cos(phase),
                                                     radius * math.sin(phase)])
        return self.phase

    def belly(self):
        pts = np.array([self.data.xpos[b] for b in self.rope_body_ids])
        c = pts[[0, -1]].mean(axis=0)
        mid = pts[len(pts) // 2]
        ang = math.atan2(mid[1] - c[1], -(mid[2] - c[2]))
        return ang, float(np.hypot(mid[1] - c[1], mid[2] - c[2])), pts
