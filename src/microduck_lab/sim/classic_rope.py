"""Classic (overhead) rope skipping world — the elastic-cable rope forms a real
rotating loop that goes OVER the jumper's head and grazes the floor, with three
ducks in the scene. The rope ends ride on mocap carriers (the declared "firm
grip" idealization — the turner ducks' whole-body drive is the RL program, see
the 0904 doc); the rope's rotation physics, the jumper's contacts, and the
trips are all real.

The elastic cable (bend stiffness) is load-bearing: the floppy serial chain
never inflates into a loop (measured — CR-02), the cable with mild bend
stiffness does. CPU MuJoCo only (the Warp GPU backend rejects body plugins).
"""

from __future__ import annotations

import math
import pathlib

import mujoco
import numpy as np

from .composer import DuckSpec, DUCK_COLORS, _yaw_quat, _tint_duck, _add_handle
from ..sim.runtime import PolicyBank, DuckRuntime

from ..paths import ASSET_ROOT as _ROOT
ROBOT = _ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"


def build_classic_world(
    robot_xml=str(ROBOT),
    ducks: list[DuckSpec] | None = None,
    rope_length: float = 0.86,
    rope_density: float = 400.0,
    rope_bend: float = 1e3,
    rope_radius: float = 0.003,
    turner_sep: float = 0.50,
    carrier_height: float = 0.18,
    playground: bool = False,
    timestep: float = 0.005,
    rope_kind: str = "cable",       # "cable" (elastic plugin) | "chain" (ball-joint
                                    # serial chain — numerically tame, sustains a
                                    # SLOW ~1 Hz loop; measured the cable's floor
                                    # is ~2.8 Hz and chaos-fragile)
    connect_to: str = "carriers",   # "carriers" (mocap idealization) | "handles"
                                    # (rope ends ride the turners' handle bodies —
                                    # the ducks' bodies really drive the rope)
    rope_contacts: str | None = None,
    legacy_rope_offset: bool = False,
    connect_timeconst: float = 0.04,
    rope_floor_timeconst: float | None = None,
    rope_joint_type: str = "hinge",
    jumper_y: float = -.10,
    rope_initial_phase: float = 0.,
    asset_colors: bool = False,
    presentation: str = "lab",
):
    """3 ducks + an elastic-cable rope on mocap carriers, one MuJoCo world.

    Returns (model, data, info) with info = {rope_body_ids, mocap_a, mocap_b,
    carrier_centers}.
    """
    if rope_contacts not in (None, "off", "floor", "jumper", "full"):
        raise ValueError("rope_contacts must be off, floor, jumper, full, or None")
    if presentation not in ('lab', 'studio'):
        raise ValueError('presentation must be lab or studio')
    if rope_radius <= 0 or rope_density <= 0 or rope_length <= 0:
        raise ValueError("rope radius, density and length must be positive")
    if rope_joint_type not in ("hinge", "ball") or (rope_joint_type == "ball" and rope_kind != "triple"):
        raise ValueError("rope_joint_type supports hinge or ball on the triple geometry")
    if rope_contacts is not None and rope_kind != "triple":
        raise ValueError("explicit rope_contacts currently supports only the triple rope")
    if connect_timeconst < 2 * timestep:
        raise ValueError("connect_timeconst must be at least two physics steps")
    if ducks is None:
        ducks = [
            DuckSpec("lavender", (-turner_sep / 2, 0.0), yaw=0.0, color=DUCK_COLORS["lavender"]),
            DuckSpec("cream", (turner_sep / 2, 0.0), yaw=math.pi, color=DUCK_COLORS["cream"]),
            DuckSpec("sky", (0.0, jumper_y), yaw=math.pi / 2, color=DUCK_COLORS["sky"]),
        ]
    cA = np.array([-turner_sep / 2, 0.0, carrier_height])
    cB = np.array([turner_sep / 2, 0.0, carrier_height])

    spec = mujoco.MjSpec()
    spec.option.timestep = timestep      # 0.005 for duck servos; the cable holds up (CG solver, 100 iters)
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.iterations = 100
    spec.option.ls_iterations = 20
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1088
    # lighting + a visible floor (the rope and ducks must read on video)
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    light = spec.worldbody.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    spec.add_texture(name="gp", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.36, 0.55, 0.36], rgb2=[0.30, 0.47, 0.30],
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.85, 0.85, 0.85],
                     width=300, height=300)
    gmat = spec.add_material(name="gp")
    gmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "gp"
    gmat.texuniform = True
    gmat.texrepeat = [4, 4]
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    floor.material = "gp"
    if presentation == 'studio':
        # Render-only settings. No extra bodies, collision shapes or contacts.
        spec.visual.global_.offwidth = 2048
        spec.visual.global_.offheight = 2048
        spec.visual.quality.shadowsize = 4096
        spec.visual.quality.offsamples = 4
        spec.visual.headlight.diffuse = [.25, .25, .25]
        spec.visual.headlight.ambient = [.18, .18, .18]
        spec.visual.headlight.specular = [.05, .05, .05]
        spec.add_texture(name='studio_sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[.035,.055,.08], rgb2=[.16,.20,.24], width=512, height=3072)
        studio_floor = spec.add_material(name='studio_floor', rgba=[.18,.21,.25,1],
            specular=.12, shininess=.2, reflectance=0.)
        floor.material = studio_floor.name
        light.pos = [0,0,3.5]
        light.dir = [0,0,-1]
        light.diffuse = [.65,.61,.55]
        light.ambient = [.08,.08,.08]
        light.specular = [.18,.18,.18]
        for name,pos,direction,diffuse in (
            ('fill',[1,-.3,1.5],[-.6,.2,-1],[.25,.32,.40]),
            ('rim',[0,1,1.4],[0,-.6,-1],[.45,.48,.52])):
            extra = spec.worldbody.add_light(name=name,pos=pos,dir=direction)
            extra.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            extra.diffuse = diffuse
            extra.castshadow = False
    # contact bits: rope(2,1) ⟂ floor(1,7) collide; jumper(4, 5→7) ghosts the
    # rope until the skip starts (a duck IN the sweep during spin-up bleeds the
    # build — measured), while jumper⟂floor always collide
    floor.conaffinity = 7

    # ducks — the TURNERS get a visible handle welded into the beak; the rope
    # end rides the carrier right at the handle tip, so it reads as the duck
    # holding + turning the rope (the duck is a stable anchor — measured that
    # head-circling at the rope rate knocks a stander down)
    turner_set = set()
    if ducks:
        turner_set = {ducks[0].name, ducks[1].name}
    for d in ducks:
        child = mujoco.MjSpec.from_file(robot_xml)
        if d.color and not asset_colors:
            _tint_duck(child, d.color)
        if d.name in turner_set:
            _add_handle(child)
        frame = spec.worldbody.add_frame(pos=[d.pos[0], d.pos[1], 0.0], quat=_yaw_quat(d.yaw))
        spec.attach(child, prefix=f"{d.name}/", frame=frame)

    # rope carriers (mocap, non-colliding) — only in the carriers idealization
    for cname, p in ((("ropeA", cA), ("ropeB", cB)) if connect_to == "carriers" else ()):
        cb = spec.worldbody.add_body(name=cname, pos=[float(p[0]), float(p[1]), float(p[2])])
        cb.mocap = True
        cb.add_geom(name=cname + "_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004],
                    rgba=[1, 0, 0, 0.0], contype=0, conaffinity=0)

    if connect_to == "handles":
        # the rope's ends ride the turners' handle bodies. The anchors are the
        # MEASURED handle-body rest origins (tip site is 4 cm further out along
        # the stick; the connect anchors at the body origin = the beak base).
        # Measured on the compiled world: (±0.164, 0, 0.228).
        # the chain spawns from cA extending +x: anchor_a must be the LEFT
        # turner (lavender, -x) so the rope reaches anchor_b (cream, +x) in
        # its rest pose (the reverse yanks the whole span at t=0 — measured).
        # anchor at the handle TIPS (z 0.27) — pins at the lower handle-body
        # origin (0.228) do NOT spin the rope up (measured: floor graze during
        # buildup kills it); the tips (0.27) clear it.
        cA = np.array([-0.138, 0.0, 0.270])    # lavender handle tip
        cB = np.array([0.138, 0.0, 0.270])     # cream handle tip
    if rope_kind == "triple":
        # Hinge-TRIPLE chain (bend y/z + axial twist with armature) — the exact
        # rope the CR-05 rope-turner policy trains on. The twist DOF is
        # load-bearing: a two-ended rope twists one turn per loop turn, and
        # hinge-PAIR chains (no twist) cannot rotate as a loop (measured).
        # Spawned SAGGING (parabola between the two anchors) — a straight spawn
        # with L >> span yanks the far connect at t=0 (measured).
        n_seg = 24
        seg = rope_length / n_seg
        span = float(np.linalg.norm(cB - cA))

        def _rope_end(sag):
            pos = np.zeros(2)
            ang = 0.0
            prev = None
            for i in range(n_seg):
                u = (i + 0.5) / n_seg
                alpha = np.arctan2((4 * sag / span) * (2 * u - 1), 1.0)
                ang = alpha if prev is None else ang + alpha - prev
                prev = alpha
                pos += seg * np.array([np.cos(ang), np.sin(ang)])
            return pos

        lo, hi = 0.01, rope_length
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _rope_end(mid)[0] > span:
                lo = mid
            else:
                hi = mid
        sag = 0.5 * (lo + hi)
        # per-segment relative y-pitch quats (MuJoCo quat w,x,y,z; pitch about
        # -y by alpha maps +x to (cos a, 0, sin a)-style descent)
        parts = ['<mujoco model="rope"><worldbody>',
                 f'<body name="ropewrap" pos="{cA[0]} {cA[1]} {cA[2]}" '
                 f'quat="{math.cos(rope_initial_phase/2)} {math.sin(rope_initial_phase/2)} 0 0"><freejoint/>',
                 '<geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 1 0 0"/>']
        prev_alpha = 0.0
        for i in range(n_seg):
            u = (i + 0.5) / n_seg
            alpha = float(np.arctan2((4 * sag / span) * (2 * u - 1), 1.0))
            rel = alpha - prev_alpha
            prev_alpha = alpha
            # rotation about +y by -rel: quat (cos(rel/2), 0, -sin(rel/2)... ) —
            # descending tangent (alpha<0 early) must pitch the segment DOWN:
            # R_y(theta) maps +x to (cosθ, 0, -sinθ); want -sinθ = -|drop| → θ=alpha... 
            qw = math.cos(alpha / 2 - (0 if i == 0 else 0))
            # rel rotation about y by (alpha_i - alpha_{i-1}) with sign so the
            # chain sags: measured by the FK above; quat about +y of angle rel
            qw, qy = math.cos(rel / 2), -math.sin(rel / 2)
            joints = (f'<joint name="rball{i}" type="ball" damping="0.0002"/>'
                      if rope_joint_type == "ball" else
                      f'<joint name="rty{i}" type="hinge" axis="0 1 0" damping="0.0002"/>'
                      f'<joint name="rtz{i}" type="hinge" axis="0 0 1" damping="0.0002"/>'
                      f'<joint name="rtx{i}" type="hinge" axis="1 0 0" damping="0.0002" armature="1e-5"/>')
            parts.append(
                f'<body name="seg_{i}" pos="{seg if i or legacy_rope_offset else 0:.5f} 0 0" quat="{qw:.6f} 0 {qy:.6f} 0">'
                + joints +
                f'<geom name="rope_s{i}" type="capsule" size="{rope_radius}" fromto="0 0 0 {seg:.5f} 0 0" '
                # CONTACTLESS: the rope-turner policy trained with a contactless
                # rope (Warp stability); floor contact changes the yank dynamics
                # the policy's balance was hardened against — deploy as trained.
                f'density="{rope_density}" rgba="0.9 0.55 0.1 1" '
                f'friction="0.01 0.005 0.0001" contype="0" conaffinity="0" '
                f'solimp="0.6 0.8 0.001" solref="0.03 1.0"/>')
        parts.append('</body>' * (n_seg + 1))
        parts.append('</worldbody></mujoco>')
        rope_spec = mujoco.MjSpec.from_string("".join(parts))
        rope_frame = spec.worldbody.add_frame(pos=[0, 0, 0])
        spec.attach(rope_spec, prefix="rope/", frame=rope_frame)
        last_body = f"rope/seg_{n_seg - 1}"
    elif rope_kind == "chain":
        # Serial ball-joint chain (a real floppy rope — NO elastic plugin).
        # Measured: sustains a slow ~1 Hz floor-grazing overhead loop and
        # survives going live, where the elastic cable is chaos-fragile and
        # only sustains ≥2.8 Hz. The slow rate is also the hop-friendly one.
        n_seg = 40
        seg = rope_length / n_seg
        parts = ['<mujoco model="rope"><worldbody>',
                 f'<body name="ropewrap" pos="{cA[0]} {cA[1]} {cA[2]}"><freejoint/>',
                 '<geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 1 0 0"/>']
        for i in range(n_seg):
            parts.append(
                f'<body name="seg_{i}" pos="{seg:.5f} 0 0">'
                f'<joint name="rj{i}" type="ball" damping="0.0002" stiffness="0" springref="0"/>'
                # damping 0.001: the 3.1 Hz whip diverged the solver (QACC NaN,
                # measured twice); this damps the high-frequency whip
                f'<geom name="rope_s{i}" type="capsule" size="{rope_radius}" fromto="0 0 0 {seg:.5f} 0 0" '
                f'density="{rope_density}" rgba="0.9 0.55 0.1 1" '
                f'friction="0.01 0.005 0.0001" contype="2" conaffinity="1" '
                f'solimp="0.6 0.8 0.001" solref="0.03 1.0"/>')
        parts.append('</body>' * (n_seg + 1))
        parts.append('</worldbody></mujoco>')
        rope_spec = mujoco.MjSpec.from_string("".join(parts))
        rope_frame = spec.worldbody.add_frame(pos=[0, 0, 0])
        spec.attach(rope_spec, prefix="rope/", frame=rope_frame)
        last_body = f"rope/seg_{n_seg - 1}"
    else:
        # elastic cable: first body is world-welded, so ride it on a massed
        # freejoint wrapper (see oracle_rope.py note)
        cable_xml = f"""
        <mujoco model="rope">
          <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
          <worldbody>
            <body name="ropewrap" pos="{cA[0]} {cA[1]} {cA[2]}">
              <freejoint/>
              <geom type="sphere" size="0.004" contype="0" conaffinity="0" rgba="1 1 0 0"/>
              <composite type="cable" curve="s" count="41 1 1" size="{rope_length}"
                         offset="0 0 0" initial="none">
                <plugin plugin="mujoco.elasticity.cable">
                  <config key="twist" value="1e4"/><config key="bend" value="{rope_bend}"/>
                </plugin>
                <joint kind="main" damping="0.0001"/>
                <geom type="capsule" size="{rope_radius}" density="{rope_density}"
                      rgba="0.9 0.55 0.1 1" friction="0.05 0.005 0.0001" contype="1" conaffinity="1"/>
                <skin rgba="0.9 0.55 0.1 1" inflate="0.002"/>
              </composite>
            </body>
          </worldbody>
        </mujoco>
        """
        rope_spec = mujoco.MjSpec.from_string(cable_xml)
        rope_frame = spec.worldbody.add_frame(pos=[0, 0, 0])
        spec.attach(rope_spec, prefix="rope/", frame=rope_frame)
        last_body = "rope/B_last"

    # connect: wrapper→A-side anchor, rope last body→B-side anchor
    if connect_to == "handles":
        # anchor_a = LEFT turner (lavender at -x), anchor_b = RIGHT (cream) —
        # matches the chain's +x spawn direction (see above).
        anchor_a, anchor_b = "lavender/handle", "cream/handle"
    else:
        anchor_a, anchor_b = "ropeA", "ropeB"
    for n1, n2 in (("rope/ropewrap", anchor_a), (last_body, anchor_b)):
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_CONNECT
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = n1
        eq.name2 = n2
        for k in range(3):
            eq.data[k] = 0.0
        if connect_to == "handles":
            # SOFT: matches the training env's far-end connect (solref 0.04);
            # a hard connect yanks the beak harder than the policy was
            # hardened against (measured: turners fall instantly on CPU).
            eq.solref = [connect_timeconst, 1.0]
            eq.solimp = [0.8, 0.95, 0.01, 0.0, 2.0]

    # The turner ducks brace the rope they're holding — exclude rope↔turner
    # contact pairs (declared: they hold it, it doesn't knock them over). Only
    # the JUMPER's rope contact is physically scored (real trips).
    duck_names = [d.name for d in ducks]
    turner_names = sorted(set(duck_names[:2]))   # first two = turners
    # collect body names from the spec (rope segments + turner duck bodies)
    rope_body_names = sorted({(b.name or "") for b in spec.bodies
                              if (b.name or "").startswith("rope/")})
    turner_body_names = sorted({(b.name or "") for b in spec.bodies
                                if any((b.name or "").startswith(t + "/") for t in turner_names)})
    for rb in rope_body_names:
        for tb in turner_body_names:
            ex = spec.add_exclude()
            ex.bodyname1 = rb
            ex.bodyname2 = tb

    # A dedicated bit preserves the robot's existing collision groups. Set
    # this before compilation so body-level broadphase masks agree with geoms.
    if rope_contacts is not None:
        for geom in spec.geoms:
            name = geom.name or ""
            if name.startswith("rope/"):
                geom.contype = 8 if "rope_s" in name and rope_contacts != "off" else 0
                geom.conaffinity = 0
            elif name == "floor" and rope_contacts in ("floor", "full"):
                geom.conaffinity |= 8
            elif (geom.parent.name or "").startswith("sky/") and rope_contacts in ("jumper", "full"):
                if geom.contype or geom.conaffinity:
                    geom.conaffinity |= 8

    if rope_floor_timeconst is not None:
        if rope_contacts not in ("floor", "full") or rope_floor_timeconst < 2*timestep:
            raise ValueError("rope_floor_timeconst requires floor contacts and at least two physics steps")
        # Explicit pairs stiffen only rope/floor; duck contacts keep their
        # original response. Raising rope geom priority also stiffens impacts
        # against the duck, which proved unstable during spin-up.
        for geom in spec.geoms:
            if (geom.name or "").startswith("rope/rope_s"):
                spec.add_pair(geomname1=geom.name, geomname2="floor", condim=3,
                    friction=[.01,.01,.005,.0001,.0001],
                    solref=[rope_floor_timeconst,1.], solimp=[.95,.99,.001,.5,2.])
    model = spec.compile()
    # MuJoCo's compiler AUTO-COMPUTES the connect anchor from the rest pose:
    # the cable rests 0.86 m along +x from ropewrap, so B_last's auto-anchor
    # lands 0.34 m past carrier B and the rope end trails free (measured:
    # B_last at x=0.58 instead of 0.25). Zero the anchors — both ends ride
    # exactly on their carriers.
    model.eq_data[:, :] = 0.0
    if connect_to == "handles":
        # ...then re-anchor precisely: the rope ends ride the handle TIP sites
        # (body2 anchor = the tip's local position in the handle frame), and
        # the rope's last-segment TIP (body1 anchor = +seg along the segment).
        for eq_i in range(model.neq):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[eq_i]) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[eq_i]) or ""
            if "/handle" in n2:
                tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n2.replace("/handle", "/handle_tip"))
                if tip_sid >= 0:
                    model.eq_data[eq_i, 3:6] = model.site_pos[tip_sid]
            if n1.startswith("rope/seg_"):
                model.eq_data[eq_i, 0:3] = [rope_length / 24, 0.0, 0.0]
    data = mujoco.MjData(model)

    rope_body_ids = [b for b in range(model.nbody)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("rope/B_", "rope/seg_"))]
    if connect_to == "carriers":
        mocap_a = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ropeA")])
        mocap_b = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ropeB")])
    else:
        mocap_a = mocap_b = -1
    info = dict(rope_body_ids=rope_body_ids, mocap_a=mocap_a, mocap_b=mocap_b,
                carrier_centers=(cA, cB), turner_names=tuple(turner_names))
    return model, data, info


# ─────────────────────────────────────────────────────────────────────────────
class RopeHoldDrive:
    """Two-stage rope drive for the classic skip (measured physics):

    Stage 0 (lift spin-up): carriers start 0.16 m HIGH and lower over 12–20 s.
    A two-pinned rope piled on the floor can NEVER spin up (measured: the
    floor pile kills the swing); lifted, the belly swings free, the loop
    inflates (~8 Hz), then lowering the carriers deepens the loop until the
    belly grazes the floor. Real turners start a long rope exactly this way.

    Stage 1 (hold): the drive phase advances at `om_drive` (starting from the
    measured rope rate, ramped slowly toward `settle_target_hz`) plus a phase
    servo toward `cont + 90°` (kp). Pure forcing collapses the loop; pure
    tracking runs away to 4–8 Hz; the mix holds a floor-grazing loop at
    ~2.8–3.4 Hz (measured: settle ≈ target + ~1.5 Hz).

    The elastic cable cannot sustain an inflated loop below ~2.8 Hz
    (centrifugal support at a 0.18 m axis) — the jumper's hop rate is trained
    to match the rope, not vice versa.
    """

    def __init__(self, axis_z: float, settle_target_hz: float = 1.3,
                 tau_s: float = 6.0, kp: float = 4.0,
                 r_spin: float = 0.05, r_hold: float = 0.05,
                 lift: float = 0.16, lower_duration: float = 8.0,
                 hold_mode: bool = True):
        # hold_mode=False (the CHAIN): never ramp down — pure phase-tracking
        # forever. The chain self-selects ~1.05 Hz and the rate-hold machinery
        # (om_drive forcing + phase servo + amplitude servo) KILLS it
        # (measured). The cable needs the hold; the chain just tracks.
        self.hold_mode = hold_mode
        self.axis_z = axis_z
        self.settle_target = 2 * math.pi * settle_target_hz
        # external rate target (rad/s): the demo sets this to the DUCK's
        # measured hop rate once hopping — the rope follows the jumper, like
        # real turners. Clipped to the loop's sustainable band.
        self.rate_target = self.settle_target
        self.tau = tau_s
        self.kp = kp
        self.r_spin = r_spin
        self.r_hold = r_hold
        self.lift = lift
        self.lower_duration = lower_duration
        self.prev_th = None
        self.wraps = 0
        self.cont = 0.0
        self.omega = 0.0
        self.phys_i = 0
        self.om_drive = None       # None = still spinning up
        self.phi = 0.0
        self.timing_corr = 0.0     # outer PLL: pass-vs-apex timing offset (rad)
        self._corr_applied = 0.0   # slew-limited version actually applied to phi
                                     # (a timing_corr jump teleports the carriers
                                     # and yanks the loop dead — measured)
        self.t = 0.0
        self.t_inflated = None
        # swing-up is chaotic (any numeric perturbation flips the outcome —
        # measured: same code holds at 2.9 Hz in one harness, stalls in
        # another). On stall, RETRY: re-lift the carriers, re-spin with a
        # slightly different amplitude (the attempt jitter breaks the tie).
        self.attempt = 0
        self.stall_t = 0.0
        self.retries = 0
        self.track_xy = np.zeros(2)   # carrier drift-follow (rope stays over
                                        # a drifting jumper; set by the demo)

    def reset_spinup(self):
        """Re-enter spin-up after a stall (rope state is reset by the caller —
        restore the rope dofs to their rest pose, leave the ducks alone)."""
        self.om_drive = None
        self.stall_t = 0.0
        self.attempt += 1
        self.retries += 1
        self.t_inflated = None
        self.prev_th = None
        self.wraps = 0
        self.cont = 0.0
        self.omega = 0.0

    @staticmethod
    def _wrap_pi(x: float) -> float:
        return (x + math.pi) % (2 * math.pi) - math.pi

    def carrier_z(self) -> float:
        """Current carrier height: LIFTED during spin-up (a rope piled on the
        floor can never spin up — measured — so the carriers start high and the
        belly swings free, exactly like real turners starting a long rope).
        The lower runs on a FIXED schedule after inflation: the falling axis is
        a parametric pump that helps push the swing over the top."""
        if self.t_inflated is None:
            return self.axis_z + self.lift
        f = min(1.0, max(0.0, (self.t - self.t_inflated - 2.0) / self.lower_duration))
        return self.axis_z + self.lift * (1.0 - f)

    def step(self, data, ids, ma, mb, cA0, cB0, dt: float = 0.001):
        self.t += dt
        cz = self.carrier_z()
        cA = np.array([cA0[0] + self.track_xy[0], cA0[1] + self.track_xy[1], cz])
        cB = np.array([cB0[0] + self.track_xy[0], cB0[1] + self.track_xy[1], cz])
        # anti-whip: the 3.1 Hz chain occasionally diverges via a joint-velocity
        # explosion (QACC NaN — measured). Clip the rope's joint velocities to
        # twice the loop's max rate; the loop never reaches it, the whip does.
        if self.rope_vadr is not None:
            data.qvel[self.rope_vadr] = np.clip(data.qvel[self.rope_vadr], -40.0, 40.0)
        # The belly angle is measured about the FIXED final axis (not the
        # riding carrier height) — this is the exact reference the proven
        # spin-up (oracle + lift tests) used; don't "fix" it.
        pts = np.array([data.xpos[b] for b in ids])
        belly = pts[np.argmin(pts[:, 2])]
        th = math.atan2(belly[1] - self.track_xy[1], -(belly[2] - self.axis_z))
        if self.prev_th is None:
            self.prev_th = th
        dth = th - self.prev_th
        if dth > math.pi:
            self.wraps -= 1
        elif dth < -math.pi:
            self.wraps += 1
        self.prev_th = th
        new_cont = th + self.wraps * 2 * math.pi
        inst_w = (new_cont - self.cont) / dt
        self.cont = new_cont
        self.omega = 0.995 * self.omega + 0.005 * inst_w

        if self.om_drive is None:
            # proven chain spin-up: a LINEAR resonant swing (a driven
            # oscillator — deterministic, unlike the chaotic tracking catch),
            # amplitude ramping, UNTIL the swing visibly builds; then hand over
            # to phase-tracking which wraps the swing into a rotating loop.
            # A time-gated seed deadlocks when the swing hasn't built yet
            # (measured); gate on the swing amplitude instead.
            if abs(self.cont) < 1.5:
                A = min(0.05 + 0.005 * self.attempt, 0.01 + self.t * 0.008)
                yA = A * math.sin(2 * math.pi * 1.2 * self.t)
                # linear drive writes the carriers directly (no circle phase)
                data.mocap_pos[ma] = cA + np.array([0, yA, 0.0])
                data.mocap_pos[mb] = cB + np.array([0, yA, 0.0])
                self.phys_i += 1
                return
            r = min(self.r_spin * (1 + 0.04 * self.attempt), 0.012 + self.phys_i * 0.00002)
            self.phi = self.cont + math.pi / 2
            if abs(self.cont) > 4 * math.pi and abs(self.omega) > 3:
                if self.hold_mode:
                    self.om_drive = abs(self.omega)   # lock in the measured rate
                if self.t_inflated is None:
                    self.t_inflated = self.t
        else:
            # amplitude servo: hold the loop's belly radius at the axis height
            # (floor-graze). A stall/contact transient shrinks the loop and it
            # never re-inflates on phase-tracking alone (measured) — so pump
            # harder when the belly rides high, ease off when it digs.
            belly_r = math.hypot(belly[1], belly[2] - self.axis_z)
            r = float(np.clip(self.r_hold * (1 + 2.0 * (self.axis_z - belly_r)),
                              0.6 * self.r_hold, 1.6 * self.r_hold))
            self.om_drive += (self.rate_target - self.om_drive) * dt / self.tau
            err = self._wrap_pi(self.cont + math.copysign(math.pi / 2, self.omega) - self.phi)
            self.phi += math.copysign(self.om_drive, self.omega) * dt + self.kp * err * dt
            # stall watchdog: lost rotation OR the loop shrank (belly rides
            # high — after a grazing death the rope keeps swinging small and
            # om never quite hits the |om|<1 gate — measured)
            belly_r = math.hypot(belly[1] - self.track_xy[1], belly[2] - self.axis_z)
            if abs(self.omega) < 1.0 or (self.carrier_z() <= self.axis_z + 0.01
                                         and belly_r < 0.6 * self.axis_z):
                self.stall_t += dt
            else:
                self.stall_t = 0.0

        self._corr_applied += float(np.clip(self.timing_corr - self._corr_applied,
                                            -0.5 * dt, 0.5 * dt))
        phi = self.phi + self._corr_applied
        yA = r * math.cos(phi)
        zA = r * math.sin(phi)
        data.mocap_pos[ma] = cA + np.array([0, yA, zA])
        data.mocap_pos[mb] = cB + np.array([0, yA, zA])
        self.phys_i += 1

    @property
    def inflated(self) -> bool:
        return self.t_inflated is not None

    @property
    def stalled(self) -> bool:
        """Hold-phase stall → the caller should reset the rope dofs and call
        reset_spinup()."""
        return self.om_drive is not None and self.stall_t > 2.0


# ─────────────────────────────────────────────────────────────────────────────
class ChainRopeDrive:
    """Minimal drive for the serial-chain rope — the EXACT recipe measured to
    hold a ~1.05 Hz floor-grazing overhead loop live for 70+ s:

    - carriers start LIFTED (+0.16 m) and lower over t=12–20 s
    - phase-tracking circles (phi = belly_angle + 90°), amplitude ramp
      0.012→0.05 over ~2 s — the tracking self-excites the swing into rotation
    - no seed, no rate control, no watchdog: the chain's floor-graze contact
      governs the rate once live; extra machinery only breaks it (measured).

    The outer timing PLL (pass↔apex) writes `timing_corr`, applied slew-limited.
    """

    def __init__(self, axis_z: float, lift: float = 0.16,
                 lower_t0: float = 12.0, lower_t1: float = 26.0,
                 r_max: float = 0.05):
        self.axis_z = axis_z
        self.lift = lift
        self.lower_t0 = lower_t0
        self.lower_t1 = lower_t1
        self.r_max = r_max
        self.prev_th = None
        self.wraps = 0
        self.cont = 0.0
        self.omega = 0.0
        self.t = 0.0
        self.phys_i = 0
        self.timing_corr = 0.0
        self._corr_applied = 0.0
        self.track_xy = np.zeros(2)
        self.t_inflated = None
        self.stall_t = 0.0
        self.retries = 0
        self.attempt = 0
        self.lower_t0_eff = lower_t0   # shifts on each retry

    def reset_spinup(self):
        """Re-spin after a stall: swing-up is chaotic — the caller resets the
        rope dofs to rest; the lower schedule re-arms with a fresh window and
        the amplitude ramp jitters per attempt."""
        self.retries += 1
        self.attempt += 1
        self.t_inflated = None
        self.stall_t = 0.0
        self.prev_th = None
        self.wraps = 0
        self.cont = 0.0
        self.omega = 0.0
        self.lower_t0_eff = self.t + 2.0     # re-lift, lower again 12 s later
        self.r_max = 0.05 * (1 + 0.05 * self.attempt)

    def carrier_z(self) -> float:
        f = min(1.0, max(0.0, (self.t - self.lower_t0_eff) / (self.lower_t1 - self.lower_t0)))
        return self.axis_z + self.lift * (1.0 - f)

    @property
    def inflated(self) -> bool:
        if self.t_inflated is None and abs(self.cont) > 4 * math.pi and abs(self.omega) > 2:
            self.t_inflated = self.t
        return self.t_inflated is not None

    @property
    def stalled(self) -> bool:
        return self.stall_t > 2.0

    def step(self, data, ids, ma, mb, cA0, cB0, dt: float = 0.001):
        self.t += dt
        cz = self.carrier_z()
        cA = np.array([cA0[0] + self.track_xy[0], cA0[1] + self.track_xy[1], cz])
        cB = np.array([cB0[0] + self.track_xy[0], cB0[1] + self.track_xy[1], cz])
        # anti-whip: the 3.1 Hz chain occasionally diverges via a joint-velocity
        # explosion (QACC NaN — measured). Clip the rope's joint velocities to
        # twice the loop's max rate; the loop never reaches it, the whip does.
        if self.rope_vadr is not None:
            data.qvel[self.rope_vadr] = np.clip(data.qvel[self.rope_vadr], -40.0, 40.0)
        pts = np.array([data.xpos[b] for b in ids])
        belly = pts[np.argmin(pts[:, 2])]
        th = math.atan2(belly[1] - self.track_xy[1], -(belly[2] - self.axis_z))
        if self.prev_th is None:
            self.prev_th = th
        dth = th - self.prev_th
        if dth > math.pi:
            self.wraps -= 1
        elif dth < -math.pi:
            self.wraps += 1
        self.prev_th = th
        new_cont = th + self.wraps * 2 * math.pi
        inst_w = (new_cont - self.cont) / dt
        self.cont = new_cont
        self.omega = 0.995 * self.omega + 0.005 * inst_w

        # stall watchdog: after inflation, losing the rotation (or never
        # re-catching after the lower) → the caller resets + re-spins
        if self.t_inflated is not None or self.t > 18.0:
            belly_r = math.hypot(belly[1] - self.track_xy[1], belly[2] - self.axis_z)
            weak = abs(self.omega) < 1.5 or (self.carrier_z() <= self.axis_z + 0.01
                                             and belly_r < 0.5 * self.axis_z)
            self.stall_t = self.stall_t + dt if weak else 0.0
        r = min(self.r_max, 0.012 + self.phys_i * 0.00002)
        self._corr_applied += float(np.clip(self.timing_corr - self._corr_applied,
                                            -0.5 * dt, 0.5 * dt))
        phi = self.cont + math.pi / 2 + self._corr_applied
        yA = r * math.cos(phi)
        zA = r * math.sin(phi)
        data.mocap_pos[ma] = cA + np.array([0, yA, zA])
        data.mocap_pos[mb] = cB + np.array([0, yA, zA])
        self.phys_i += 1


# ─────────────────────────────────────────────────────────────────────────────
class ChainForcedDrive:
    """Deterministic rope drive for the heavy serial chain (the money-shot
    drive). Measured on the density-1000 chain: a linear resonant seed wraps
    the rope into rotation, then FORCED circles at the target rate hold a
    grazing overhead loop indefinitely (pass_hz == drive rate, top 0.33 m,
    bottom ~0) — no chaos, no runaway, direct phase control for the PLL.

    Phase-tracking (self-excited) drives were measured chaos-fragile and
    rate-uncontrollable; this drive trades "emergent" for RELIABLE: the rate
    is exactly the duck's trained hop rate, and `phase_off` shifts every pass
    by exactly itself (1:1) for the timing PLL.
    """

    def __init__(self, axis_z: float, rate_hz: float = 3.1,
                 lift: float = 0.16, lower_t0: float = 12.0, lower_t1: float = 26.0,
                 r_drive: float = 0.05, seed_hz: float = 1.2):
        self.axis_z = axis_z
        self.rate_hz = rate_hz
        self.lift = lift
        self.lower_t0 = lower_t0
        self.lower_t1 = lower_t1
        self.r_drive = r_drive
        self.seed_hz = seed_hz
        self.t = 0.0
        self.phys_i = 0
        self.phase = 0.0            # drive phase (advanced at rate_hz)
        self.phase_off = 0.0        # PLL offset (pass↔apex), slew-limited
        self._off_applied = 0.0
        self.track_xy = np.zeros(2)
        self.prev_th = None
        self.wraps = 0
        self.cont = 0.0
        self.omega = 0.0
        self.t_inflated = None
        self.rope_vadr = None    # set by the caller (rope dof addresses)

    def carrier_z(self) -> float:
        f = min(1.0, max(0.0, (self.t - self.lower_t0) / (self.lower_t1 - self.lower_t0)))
        return self.axis_z + self.lift * (1.0 - f)

    @property
    def inflated(self) -> bool:
        if self.t_inflated is None and abs(self.cont) > 2 * math.pi and abs(self.omega) > 2:
            self.t_inflated = self.t
        return self.t_inflated is not None

    def step(self, data, ids, ma, mb, cA0, cB0, dt: float = 0.001):
        self.t += dt
        cz = self.carrier_z()
        cA = np.array([cA0[0] + self.track_xy[0], cA0[1] + self.track_xy[1], cz])
        cB = np.array([cB0[0] + self.track_xy[0], cB0[1] + self.track_xy[1], cz])
        # anti-whip: the 3.1 Hz chain occasionally diverges via a joint-velocity
        # explosion (QACC NaN — measured). Clip the rope's joint velocities to
        # twice the loop's max rate; the loop never reaches it, the whip does.
        if self.rope_vadr is not None:
            data.qvel[self.rope_vadr] = np.clip(data.qvel[self.rope_vadr], -40.0, 40.0)
        # measure the belly angle (for the inflated flag + monitoring only)
        pts = np.array([data.xpos[b] for b in ids])
        belly = pts[np.argmin(pts[:, 2])]
        th = math.atan2(belly[1] - self.track_xy[1], -(belly[2] - self.axis_z))
        if self.prev_th is None:
            self.prev_th = th
        dth = th - self.prev_th
        if dth > math.pi:
            self.wraps -= 1
        elif dth < -math.pi:
            self.wraps += 1
        self.prev_th = th
        new_cont = th + self.wraps * 2 * math.pi
        self.omega = 0.995 * self.omega + 0.005 * (new_cont - self.cont) / dt
        self.cont = new_cont

        if not self.inflated:
            # linear resonant seed: wraps the rope into rotation reliably
            A = min(0.06, 0.012 + self.t * 0.008)
            yA = A * math.sin(2 * math.pi * self.seed_hz * self.t)
            zA = 0.0
            self.phase = 2 * math.pi * self.rate_hz * self.t  # keep the clock fresh
        else:
            # forced circles at the duck's rate + the PLL phase offset
            self.phase += 2 * math.pi * self.rate_hz * dt
            self._off_applied += float(np.clip(self.phase_off - self._off_applied,
                                               -0.5 * dt, 0.5 * dt))
            phi = self.phase + self._off_applied
            yA = self.r_drive * math.cos(phi)
            zA = self.r_drive * math.sin(phi)
        data.mocap_pos[ma] = cA + np.array([0, yA, zA])
        data.mocap_pos[mb] = cB + np.array([0, yA, zA])
        self.phys_i += 1
