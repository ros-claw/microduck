"""Real-gap long-jump candidate. Not a validated skill until CPU gap batteries pass.

50 Hz PD deployment contract, 0.64 Nm current limit, native collision meshes.
The two terrain boxes have no supporting plane between them. Initial development
uses a 15 cm gap; no mid-episode root pose or velocity interventions are used.
"""
from copy import deepcopy
from dataclasses import dataclass
import mujoco
import numpy as np
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainOutput, TerrainGeometry
from .microduck_jump_env_cfg import make_microduck_jump_env_cfg, MicroduckJumpRlCfg
from . import mdp as microduck_mdp
from mjlab_microduck.robot.microduck_constants import get_standup_spec


@dataclass(kw_only=True)
class ParkourGapTerrainCfg(SubTerrainCfg):
    gap: float = .15
    edge: float = .10

    def function(self,difficulty,spec,rng):
        del difficulty,rng
        body=spec.body('terrain');ox,oy=self.size[0]/2,self.size[1]/2
        geometries=[]
        for lo,hi in [(0,ox+self.edge),(ox+self.edge+self.gap,self.size[0])]:
            g=body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=[(lo+hi)/2,oy,-.04],size=[(hi-lo)/2,.40,.04],friction=[1,.005,.0001])
            geometries.append(TerrainGeometry(geom=g,color=(.08,.10,.15,1)))
        return TerrainOutput(origin=np.array([ox,oy,0.]),geometries=geometries)


def parkour_robot_spec():
    spec=get_standup_spec()
    for actuator in spec.actuators:
        actuator.forcerange=[-.3660*1.75,.3660*1.75];actuator.forcelimited=True
    return spec


def make_microduck_parkour_env_cfg(play=False):
    cfg=make_microduck_jump_env_cfg(play=play,pd_actuators=True)
    cfg.scene.entities['robot']=deepcopy(cfg.scene.entities['robot'])
    cfg.scene.entities['robot'].spec_fn=parkour_robot_spec
    cfg.scene.entities['robot'].init_state.pos=(0.,0.,.12)
    cfg.scene.entities['robot'].collisions=()  # Preserve the actual deployment XML.
    cfg.sim.mujoco.timestep=.002
    cfg.decimation=10
    cfg.episode_length_s=2.0
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(size=(3.,2.),num_rows=1,num_cols=1,
        sub_terrains={'gap':ParkourGapTerrainCfg()},border_width=0.,color_scheme='none')
    cfg.events['reset_base'].params['pose_range']={'x':(-.015,.015),'y':(-.005,.005),'z':(0.,.005),'yaw':(-.035,.035)}
    cfg.events['reset_base'].params['velocity_range']={'x':(0.,.15)}
    cfg.rewards.pop('jump_stay_in_place')
    cfg.rewards.pop('jump_landing_composite')
    cfg.rewards['jump_apex_progress'].weight=2.
    cfg.rewards['jump_apex_progress'].func=microduck_mdp.parkour_apex
    cfg.rewards['jump_apex_progress'].params['target_height']=.06
    cfg.rewards['parkour_progress']=RewardTermCfg(func=microduck_mdp.parkour_progress,weight=4.)
    cfg.rewards['parkour_landing']=RewardTermCfg(func=microduck_mdp.parkour_landing,weight=8.)
    cfg.rewards['parkour_sideways']=RewardTermCfg(func=microduck_mdp.parkour_sideways,weight=5.)
    cfg.terminations['fell_into_gap']=TerminationTermCfg(func=microduck_mdp.parkour_below_platform)
    cfg.rewards['action_rate_l2'].weight=-.005
    cfg.curriculum['action_rate_weight'].params['weight_stages']=[
        {'step':0,'weight':-.005},{'step':500*24,'weight':-.02},{'step':1000*24,'weight':-.05}]
    # Introduce smoothness after discovery; this experiment
    # changes the task, not its 61-D observation or action order.
    return cfg


MicroduckParkourRlCfg=deepcopy(MicroduckJumpRlCfg)
MicroduckParkourRlCfg.experiment_name='microduck_parkour'
MicroduckParkourRlCfg.run_name='real_gap_v0'
MicroduckParkourRlCfg.max_iterations=1500
MicroduckParkourRlCfg.save_interval=100
