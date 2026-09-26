"""Current observed geometry only: no world seed, future sequence or motor phase."""
import math
import mujoco
import numpy as np
LANES=(-.48,0.,.48)

def observe(m,d,duck,chunks,active_skill,target_lane):
    pos=duck.trunk_pos();hazards=[]
    for c in chunks:
        bid=int(m.body(c.name).id);p=d.xpos[bid];g=int(m.geom(c.name+'_g').id);gp=d.geom_xpos[g]
        if not (-.8<gp[0]-pos[0]<1.8):continue
        velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
        radius=.31 if c.kind in ('sweeper','pendulum') else (.30 if c.kind=='moving_wall' else float(m.geom_rbound[g]))
        hazards.append(dict(id=c.name,kind=c.kind,distance_m=float(gp[0]-pos[0]),lateral_m=float(gp[1]),height_m=float(gp[2]),half_width_m=radius,velocity_m_s=velocity[3:].tolist()))
    hazards.sort(key=lambda h:h['distance_m'])
    supported=False
    for ct in d.contact:
        if ct.efc_address<0:continue
        geoms=[int(ct.geom1),int(ct.geom2)]
        if any((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '')=='floor' for g in geoms):
            bodies=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '' for g in geoms]
            if any('duck/ankle_' in b for b in bodies):supported=True
    state=dict(robot=dict(x_m=float(pos[0]),y_m=float(pos[1]),speed_m_s=float(np.linalg.norm(duck.trunk_linvel()[:2])),yaw_rad=float(duck.trunk_yaw()),upright=float(d.xmat[duck.trunk_body_id,8]),airborne=not supported,active_skill=active_skill,target_lane=target_lane),hazards=hazards,mission=dict(objective='reach_finish_without_contact',style='prefer a verified roll when safe; otherwise bypass',finish_x_m=8.,elapsed_s=float(d.time),time_limit_s=60.),skills=dict(run=dict(speed_m_s=.27,braking_distance_m=.25),roulade=dict(duration_s=2.,travel_m=.59,requires_standing_start=True,measured_min_bar_center_height_m=.285,validated_start_distance_m=[.31,.42]),jump=dict(available=False,reason='Only in-place hopping measured; parkour crossing not validated'),lane_change=dict(method='turn and walk',approx_duration_s=3.)))

    state['lanes']={name:dict(y_m=LANES[i],clear_next_1_3m=lane_clear(state,i)) for i,name in enumerate(('right','center','left'))}
    return state

def lane_clear(state,lane,horizon=1.3):
    return all(not(-.15<h['distance_m']<horizon and abs(h['lateral_m']-LANES[lane])<h['half_width_m']+.12 and not(h['kind']=='gate' and h['height_m']>.58)) for h in state['hazards'])

def candidates(state,locked=False,roll_ready=True):
    if locked:return ['CONTINUE']
    r=state['robot']
    if r['upright']<.8:return ['BRAKE']
    # State-safe candidate filtering does not claim full reachability proof.
    result=['BRAKE'];lane=r['target_lane']
    if lane_clear(state,lane,.22+.8*r['speed_m_s']):result.append('KEEP_RUNNING')
    if lane<2 and lane_clear(state,lane+1) and not r['airborne']:result.append('DODGE_LEFT')
    if lane>0 and lane_clear(state,lane-1) and not r['airborne']:result.append('DODGE_RIGHT')
    for h in state['hazards']:
        if h['kind']=='high_bar' and 0<h['distance_m']<.40 and abs(r['y_m']-h['lateral_m'])<h['half_width_m']+.12 and 'KEEP_RUNNING' in result:
            result.remove('KEEP_RUNNING')
        if h['kind']=='high_bar' and .30<h['distance_m']<.42 and abs(r['y_m']-h['lateral_m'])<.06 and abs(r['yaw_rad'])<.20 and r['speed_m_s']<.045 and not r['airborne'] and roll_ready and h['height_m']>=.285:
            result.append('ROULADE');break
    return result
