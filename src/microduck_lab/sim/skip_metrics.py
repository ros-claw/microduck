"""Geometric and contact-based scoring for the three-duck, x-span rope rig.

No timing-only counter is used. One opportunity is a full signed revolution
between successive upper crossings of the same material segment of the rope.
Both foot center lines must be crossed below their actual mesh soles, the rope
must also cross overhead, and a clean supported landing must follow.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


class SkipCycles:
    def __init__(self, warmup=9.0, clearance=.001, landing_s=.05, attachment_limit=.01):
        self.warmup = warmup
        self.clearance = clearance
        self.landing_s = landing_s
        self.attachment_limit = attachment_limit
        self.previous_angle = None
        self.angle = 0.0
        self.last_upper = None
        self.last_upper_angle = 0.0
        self.previous_t = None
        self.cycles = []
        self.startup = False
        self.state = self._new_state()

    @staticmethod
    def _new_state():
        return dict(foot_hits={}, overhead=False, faults=set(), landing_duration=0., landed=False,
                    max_floor_penetration=0.)

    def update(self, *, t, phase, crossings, supported, upright, body_contact,
               rope_contact, attachment_gap, collisions_enabled, turners_upright=True,
               floor_penetration=0.):
        dt = 0.0 if self.previous_t is None else t - self.previous_t
        self.previous_t = t
        if self.previous_angle is None:
            self.angle = phase
            self.previous_angle = phase
            return
        before = self.angle
        self.angle += (phase - self.previous_angle + math.pi) % (2*math.pi) - math.pi
        self.previous_angle = phase
        upper = math.floor((before + math.pi)/(2*math.pi)) != math.floor((self.angle + math.pi)/(2*math.pi))
        if upper:
            complete = self.last_upper is not None and abs(self.angle-self.last_upper_angle) > 1.5*math.pi
            if complete and t <= self.warmup:
                self.startup = True
            if complete and self.last_upper >= self.warmup:
                s = self.state
                reasons = set(s['faults'])
                if len(s['foot_hits']) != 2:
                    reasons.add('did_not_clear_both_feet')
                elif max(s['foot_hits'].values())-min(s['foot_hits'].values()) > .12:
                    reasons.add('foot_passes_not_together')
                if not s['overhead']:
                    reasons.add('no_overhead_crossing')
                if not s['landed']:
                    reasons.add('no_clean_landing')
                self.cycles.append(dict(start=self.last_upper, end=t,
                                        clean=not reasons, reasons=sorted(reasons),
                                        max_floor_penetration_m=s['max_floor_penetration']))
            self.last_upper = t
            self.last_upper_angle = self.angle
            self.state = self._new_state()
        s = self.state
        s['max_floor_penetration'] = max(s['max_floor_penetration'],floor_penetration)
        if not upright:
            s['faults'].add('tilted')
        if not turners_upright:
            s['faults'].add('fallen_turner')
        if body_contact:
            s['faults'].add('body_ground_contact')
        if rope_contact:
            s['faults'].add('rope_duck_contact')
        if attachment_gap > self.attachment_limit:
            s['faults'].add('loose_attachment')
        if not collisions_enabled:
            s['faults'].add('collisions_disabled')
        if floor_penetration > .002:
            s['faults'].add('rope_below_floor')
        for foot, gap, overhead in crossings:
            if overhead:
                s['overhead'] = True
            elif gap >= self.clearance:
                s['foot_hits'][foot] = t
            else:
                s['faults'].add('insufficient_sole_clearance')
        if len(s['foot_hits']) == 2 and supported and upright and not body_contact and not rope_contact:
            s['landing_duration'] += dt
            if s['landing_duration'] >= self.landing_s:
                s['landed'] = True
        else:
            s['landing_duration'] = 0.

    def result(self):
        clean = sum(c['clean'] for c in self.cycles)
        count = len(self.cycles)
        return dict(full_revolutions=count, clean_skips=clean,
                    physical_success_rate=clean/count if count else None,
                    startup_before_warmup=self.startup,
                    passed=bool(self.startup and count >= 20 and clean/count >= .8),
                    criteria=dict(warmup_s=self.warmup, minimum_cycles=20,
                                  sole_clearance_m=self.clearance, landing_support_s=self.landing_s,
                                  attachment_limit_m=self.attachment_limit, min_upright_cos=.8,
                                  max_rope_floor_penetration_m=.002),
                    cycles=self.cycles)


class PhysicalSkipAudit:
    """Observer for run_honest_classic_skip; inspects every physics step."""
    def __init__(self):
        self.scorer = SkipCycles()
        self.previous_points = [None, None]
        self.last_underfoot_crossing = None
        self.crossing_samples = []
        self.low_crossings = [None,None]

    def setup(self, m, d, info):
        self.m = m
        names = [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
        bodies = [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '' for b in m.geom_bodyid]
        self.rope = np.array([n.startswith('rope/rope_s') for n in names])
        self.sky = np.array([b.startswith('sky/') for b in bodies])
        self.rope_ids = np.flatnonzero(self.rope)
        self.feet = [names.index('sky/left_foot_collision'), names.index('sky/right_foot_collision')]
        self.floor = names.index('floor')
        self.trunk = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'sky/trunk_base')
        self.turner_trunks = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n+'/trunk_base') for n in ('lavender','cream')]
        self.handles = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n+'/handle_tip') for n in ('lavender','cream')]
        self.vertices = {}
        for g in np.flatnonzero(self.sky & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                self.vertices[g] = np.array(m.mesh_vert[start:start+count], dtype=float)
        self.enabled = all(bool((m.geom_contype[r] & m.geom_conaffinity[g]) or
                                (m.geom_contype[g] & m.geom_conaffinity[r]))
                           for r in self.rope_ids for g in self.vertices)
        self.enabled &= all(bool(m.geom_contype[r] & m.geom_conaffinity[self.floor]) for r in self.rope_ids)

    def __call__(self, m, d, info, hopping):
        if not all(np.isfinite(v).all() for v in (d.qpos,d.qvel,d.qacc)) or np.any(d.warning.number):
            raise RuntimeError(f'invalid physics after t={self.scorer.previous_t}; reported t={d.time}')
        bounds = {}
        head_top = -np.inf
        for g, vertices in self.vertices.items():
            rotation = d.geom_xmat[g].reshape(3,3)
            if g in self.feet:
                world = vertices @ rotation.T + d.geom_xpos[g]
                bounds[g] = (world.min(axis=0), world.max(axis=0))
                top = bounds[g][1][2]
            else:
                top = np.max(vertices @ rotation[2])+d.geom_xpos[g,2]
            head_top = max(head_top,top)
        g = self.rope_ids
        half = d.geom_xmat[g].reshape(-1,3,3)[:,:,2] * m.geom_size[g,1,None]
        a, b = d.geom_xpos[g]-half, d.geom_xpos[g]+half
        penetration = max(0., float(d.geom_xpos[self.floor,2] -
            np.min(np.minimum(a[:,2],b[:,2])-m.geom_size[g,0])))
        center = d.geom_xpos[g[len(g)//2]]
        axis = d.site_xpos[self.handles].mean(axis=0)
        phase = math.atan2(center[1]-axis[1], axis[2]-center[2])
        crossings = []
        for i, foot in enumerate(self.feet):
            lo, hi = bounds[foot]
            target = (lo+hi)/2
            denom = b[:,0]-a[:,0]
            fractions = np.divide(target[0]-a[:,0],denom,out=np.full(len(g),np.inf),where=np.abs(denom)>1e-8)
            candidates = np.flatnonzero((fractions >= 0) & (fractions <= 1))
            current = None
            if len(candidates):
                points = a[candidates]+fractions[candidates,None]*(b[candidates]-a[candidates])
                k = int(np.argmin(np.abs(points[:,1]-target[1])))
                seg, point = int(candidates[k]), points[k]
                current = (seg,point.copy(),float(point[1]-target[1]))
                prev = self.previous_points[i]
                if (prev is not None and abs(seg-prev[0]) <= 1 and
                        np.linalg.norm(point-prev[1]) < .03 and prev[2]*current[2] < 0):
                    r = float(m.geom_size[g[seg],0])
                    overhead = bool(point[2]-r > head_top)
                    crossings.append((i,float(lo[2]-point[2]-r),overhead))
                    self.crossing_samples.append(dict(t=float(d.time),foot=i,
                        sole_z=float(lo[2]),rope_z=float(point[2]),overhead=overhead,
                        foot_y=float(target[1]),axis_y=float(axis[1]),axis_z=float(axis[2])))
                    if point[2] < axis[2]:
                        self.low_crossings[i] = float(d.time)
                        if all(v is not None for v in self.low_crossings) and abs(self.low_crossings[0]-self.low_crossings[1]) < .12:
                            self.last_underfoot_crossing = sum(self.low_crossings)/2
            self.previous_points[i] = current
        supported = body_contact = rope_contact = False
        for c in d.contact:
            if c.efc_address < 0:
                continue
            x,y = int(c.geom1),int(c.geom2)
            if (self.rope[x] and self.sky[y]) or (self.rope[y] and self.sky[x]):
                rope_contact = True
            if self.floor in (x,y):
                other = y if x == self.floor else x
                if other in self.feet:
                    supported = True
                elif self.sky[other]:
                    body_contact = True
        gap = 0.
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT:
                x,y = m.eq_obj1id[e],m.eq_obj2id[e]
                pa = d.xpos[x]+d.xmat[x].reshape(3,3)@m.eq_data[e,:3]
                pb = d.xpos[y]+d.xmat[y].reshape(3,3)@m.eq_data[e,3:6]
                gap = max(gap,float(np.linalg.norm(pa-pb)))
        self.scorer.update(t=float(d.time),phase=phase,crossings=crossings,
                           supported=supported,upright=bool(d.xmat[self.trunk,8]>.8),
                           body_contact=body_contact,rope_contact=rope_contact,
                           attachment_gap=gap,collisions_enabled=self.enabled,
                           turners_upright=bool(np.all(d.xmat[self.turner_trunks,8]>.8)),
                           floor_penetration=penetration)
