"""Read-only contact chronology and formation measurements for failure analysis."""
from itertools import combinations
import mujoco
import numpy as np


class FormationDiagnostics:
    def __init__(self):
        self.events = []
        self.samples = []
        self.last_events = {}
        self.initial_bounds = {}

    def setup(self, m, d, info):
        self.names = ('lavender', 'cream', 'sky', 'graphite')
        self.names = tuple(n for n in self.names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n+'/trunk_base') >= 0)
        self.owners = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or '').split('/')[0] for b in m.geom_bodyid]
        self.geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or
            f'{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))}#geom{g}' for g in range(m.ngeom)]
        self.rope_ids = info['rope_body_ids']
        self.minimum_separation = {'/'.join(p): float('inf') for p in combinations(self.names, 2)}

    def physics(self, m, d):
        for contact_index, contact in enumerate(d.contact):
            if contact.efc_address < 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            owners = (self.owners[a], self.owners[b])
            # Keep robot/robot and rope/robot contacts; ordinary soles on floor are not failures.
            if owners[0] == owners[1] or not all(n in self.names or n == 'rope' for n in owners):
                continue
            key = '/'.join(sorted(owners))
            if d.time - self.last_events.get(key, -1) < .10:
                continue
            self.last_events[key] = float(d.time)
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, contact_index, force)
            self.events.append(dict(t=float(d.time), pair=key,
                geoms=[self.geom_names[a], self.geom_names[b]], penetration_m=max(0., -float(contact.dist)),
                normal_force_n=float(force[0])))

    def sample(self, m, d, rt):
        if not self.initial_bounds:
            for name in self.names:
                points = []
                for g, owner in enumerate(self.owners):
                    if owner != name or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                        continue
                    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh = m.geom_dataid[g]; start = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
                        points.extend(m.mesh_vert[start:start+count] @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g])
                    else:
                        points.extend([d.geom_xpos[g]-m.geom_rbound[g],d.geom_xpos[g]+m.geom_rbound[g]])
                points = np.array(points)
                self.initial_bounds[name] = dict(min=points.min(0).tolist(), max=points.max(0).tolist())
        positions = {n: rt[n].trunk_pos().copy() for n in self.names}
        for pair in combinations(self.names, 2):
            key = '/'.join(pair)
            self.minimum_separation[key] = min(self.minimum_separation[key], float(np.linalg.norm(positions[pair[0]][:2]-positions[pair[1]][:2])))
        self.samples.append(dict(t=float(d.time), robots={n: dict(position=positions[n].tolist(),
            upright=float(d.body(n+'/trunk_base').xmat[8]), phase=float(rt[n].hop_phase_source()) if n in ('sky','graphite') else None,
            command=rt[n].command[:3].tolist()) for n in self.names},
            turn_hz=float(rt['lavender'].turn_frequency), phase_offset_s=float(rt['lavender'].phase_offset_s),
            rope_midpoint=d.xpos[self.rope_ids[len(self.rope_ids)//2]].tolist()))

    def result(self):
        return dict(initial_collision_bounds=self.initial_bounds,minimum_root_xy_separation_m=self.minimum_separation,
                    contact_events=self.events,samples=self.samples)
