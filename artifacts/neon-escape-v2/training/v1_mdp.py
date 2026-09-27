# ── Parkour: real gap development experiment ─────────────────────────────────
def parkour_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-time forward frontier, capped beyond the gap; no walking annuity."""
    robot = env.scene['robot']
    xy = robot.data.root_link_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2]
    x = torch.nan_to_num(xy[:, 0], nan=0.).clamp(0., .32)
    if not hasattr(env, '_parkour_frontier'):
        env._parkour_frontier = x.clone()
    fresh = env.episode_length_buf <= 1
    env._parkour_frontier[fresh] = x[fresh]
    delta = (x-env._parkour_frontier).clamp(0., .5*env.step_dt)
    env._parkour_frontier = torch.maximum(env._parkour_frontier,x)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0., 1.)
    return delta/env.step_dt * upright


def parkour_landing(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pay only supported upright landings beyond the far edge, after real air.

    Both sole sites must clear the ground while upright. The CPU deployment
    battery additionally audits all sole vertices and the entire contact stream.
    """
    robot=env.scene['robot']
    if not hasattr(env,'_parkour_foot_ids'):
        env._parkour_foot_ids=robot.find_sites(['left_foot','right_foot'],preserve_order=True)[0]
        env._parkour_air=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    feet=robot.data.site_pos_w[:,env._parkour_foot_ids]-env.scene.terrain.env_origins[:,None,:]
    upright=-robot.data.projected_gravity_b[:,2]
    fresh=env.episode_length_buf<=1
    env._parkour_air[fresh]=False
    grounded = _sensor_any_contact(env, 'feet_ground_contact')
    env._parkour_air |= (feet[:,:,2].min(dim=1).values>.035)&(upright>.8)&(~grounded)
    beyond=(feet[:,:,0].min(dim=1).values>.26)
    low=(feet[:,:,2].min(dim=1).values<.02)
    xy=robot.data.root_link_pos_w[:,:2]-env.scene.terrain.env_origins[:,:2]
    speed=robot.data.root_link_lin_vel_w.norm(dim=-1)
    return (env._parkour_air & beyond & low & grounded & (upright>.95)).float()*torch.exp(-xy[:,1].square()/.01)*torch.exp(-speed.square()/.25)


def parkour_sideways(env: ManagerBasedRlEnv) -> torch.Tensor:
    xy=env.scene['robot'].data.root_link_pos_w[:,:2]-env.scene.terrain.env_origins[:,:2]
    return -torch.nan_to_num(xy[:,1].square(),nan=0.).clamp(max=1.)


def parkour_below_platform(env: ManagerBasedRlEnv) -> torch.Tensor:
    z=env.scene['robot'].data.root_link_pos_w[:,2]-env.scene.terrain.env_origins[:,2]
    return z < -.12


def parkour_apex(env: ManagerBasedRlEnv, target_height=.06, max_paid_rate=.5) -> torch.Tensor:
    """Upright, unsupported sole-height progress; falling cannot earn apex credit."""
    robot=env.scene['robot']
    ids=robot.find_sites(['left_foot','right_foot'],preserve_order=True)[0]
    z=robot.data.site_pos_w[:,ids,2].min(dim=1).values-env.scene.terrain.env_origins[:,2]-.015
    valid=(-robot.data.projected_gravity_b[:,2]>.8)&(~_sensor_any_contact(env,'feet_ground_contact'))
    height=torch.where(valid,z.clamp(0.,target_height),torch.zeros_like(z))
    if not hasattr(env,'_parkour_apex_frontier'):
        env._parkour_apex_frontier=torch.zeros_like(z)
    env._parkour_apex_frontier[env.episode_length_buf<=1]=0.
    delta=(height-env._parkour_apex_frontier).clamp(0.,max_paid_rate*env.step_dt)
    env._parkour_apex_frontier=torch.maximum(env._parkour_apex_frontier,height)
    return delta/(env.step_dt*target_height)
