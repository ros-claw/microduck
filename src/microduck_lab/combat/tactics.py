"""Public-state tactics and committed motor maneuvers for continuous matches.

The selector can be rules or a frozen learned categorical policy. It is called
only at a legal maneuver boundary. Recovery/boundary preemption is explicit.
"""

import math
import numpy as np

from .maneuver import MotorCommand
from .observations import relative
from .recovery import RecoveryGuard

ACTIONS = ("APPROACH", "ORBIT_LEFT", "ORBIT_RIGHT", "DASH_BUMP", "EVADE_LEFT", "EVADE_RIGHT", "BRACE", "CENTER", "RETREAT")
FEATURES = 32


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def local(vec, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c*vec[0]+s*vec[1], -s*vec[0]+c*vec[1]])


def features(me, target, others, radius, cooldown):
    delta, dist, angle, closing = relative(me, target)
    nearby = sorted(others, key=lambda pair: (np.linalg.norm(np.array(pair[0].xy)-me.xy), pair[0].name))
    extras = []
    for b, allied in nearby[:2]:
        extras.extend([*np.clip(local(np.array(b.xy)-me.xy, me.yaw)/radius, -2, 2), b.upright, float(allied)])
    extras += [0.] * (8-len(extras))
    result = [*np.array(me.xy)/radius, *local(me.velocity, me.yaw)/.5,
              math.sin(me.yaw), math.cos(me.yaw), me.upright, float(me.foot_support), me.boundary_distance/radius,
              *local(delta, me.yaw)/radius, *local(np.array(target.velocity)-me.velocity, me.yaw)/.5,
              math.sin(angle), math.cos(angle), target.upright, float(target.foot_support), target.boundary_distance/radius,
              dist/radius, closing/.5, *extras, min(1., cooldown/2), float(me.touching_opponent),
              math.sin(target.yaw-me.yaw), math.cos(target.yaw-me.yaw)]
    result = np.asarray(result, dtype=np.float32)
    if result.shape != (FEATURES,) or not np.isfinite(result).all():
        raise ValueError("Invalid public tactical observation")
    return np.clip(result, -4, 4)


def choose_target(name, public, teams, alive, previous=None):
    enemies = [n for n in alive if teams[n] != teams[name]]
    if not enemies:
        return None
    allies = sorted(n for n in alive if teams[n] == teams[name])
    # A deterministic minimum-distance assignment from one same-tick snapshot.
    # It transmits only a target ID; neither teammate reads enemy internals.
    if len(allies) == len(enemies) == 2:
        enemies.sort()
        def cost(order):
            return sum(np.linalg.norm(np.array(public[a].xy)-public[b].xy) for a,b in zip(allies,order))
        order = enemies if cost(enemies) <= cost(enemies[::-1]) else enemies[::-1]
        return order[allies.index(name)]
    return min(enemies, key=lambda n: (np.linalg.norm(np.array(public[n].xy)-public[name].xy), n))


class RuleSelector:
    def __init__(self, style):
        self.style = style

    def select(self, obs, mask, context):
        dist, closing, angle, boundary, side = context
        if boundary < .18 and mask[7]:
            action = 7
        elif self.style in ("counter", "defensive") and dist < .44 and closing > .12 and mask[4 if side > 0 else 5]:
            action = 4 if side > 0 else 5
        elif mask[3] and (self.style != "defensive" or closing < .03):
            action = 3
        elif self.style == "defensive" and dist < .3:
            action = 6
        elif not mask[0] and mask[8]:
            action = 8
        elif not mask[0] and mask[1 if side > 0 else 2]:
            action = 1 if side > 0 else 2
        elif self.style == "orbit" and dist < .44 and mask[1 if side > 0 else 2]:
            action = 1 if side > 0 else 2
        elif abs(angle) > .6 and dist < .27 and mask[8]:
            action = 8
        else:
            action = 0
        if not mask[action]:
            action = 6
        return action, dict(log_probability=0., value=0., selector="rule:"+self.style)


class Pilot:
    def __init__(self, name, team, teams, config, selector=None):
        self.name, self.team, self.teams, self.c = name, team, teams, config
        self.selector = selector or RuleSelector(config.red_style if team == "RED" else config.blue_style)
        self.guard = RecoveryGuard(name)
        self.events, self.trace = [], []
        self.phase, self.skill = "READY", "approach_orbit"
        self.action, self.target = None, None
        self.since, self.phase_since = 0., 0.
        self.origin = np.zeros(2)
        self.aim, self.serial = 0., 0
        self.attack_id = None
        self.last_attack_end = -10.
        self.contact_since = None
        self.decision = None
        self.last_selector_time = -1.
        self.pending_reward = 0.

    @property
    def active_attack(self):
        return self.attack_id if self.action == 3 and self.phase == "ACTIVE" else None

    def event(self, now, state, **kw):
        self.events.append(dict(time_s=now, state=state, actor=self.name, **kw))

    def finish(self, now, reason="COMPLETE", final=False):
        if self.decision is not None:
            self.decision.update(duration_s=max(.02, now-self.decision["time_s"]),
                                 reward=self.pending_reward, end_reason=reason, terminal=final)
            self.trace.append(self.decision)
            self.decision, self.pending_reward = None, 0.
        if self.action == 3:
            self.last_attack_end = now
            self.event(now, "ATTACK_WINDOW_ENDED", attack_id=self.attack_id, target=self.target, reason=reason)
        self.action, self.phase, self.skill = None, "READY", "approach_orbit"

    def legal(self, me, other, now):
        _, dist, angle, closing = relative(me, other)
        margin = me.boundary_distance
        mask = np.ones(len(ACTIONS), dtype=bool)
        mask[3] = (.20 < dist < self.c.dash_start_distance_m and abs(angle) < .24
                   and now-self.last_attack_end > .8 and me.upright > .94 and me.foot_support
                   and not me.touching_opponent)
        mask[0] = dist >= .23 and not me.touching_opponent
        for action, side in ((1,1), (2,-1), (4,1), (5,-1)):
            delta = np.array([-math.sin(me.yaw), math.cos(me.yaw)]) * side * .19
            mask[action] = np.linalg.norm(np.array(me.xy)+delta) < self.c.radius_m-.06
        mask[4:6] &= dist < .55
        backward = np.array(me.xy)-np.array([math.cos(me.yaw), math.sin(me.yaw)])*.15
        mask[8] = dist < .5 and np.linalg.norm(backward) < self.c.radius_m-.06
        return mask, (dist, closing, angle, margin, 1 if me.xy[1] <= 0 else -1)

    def navigate(self, me, point, speed=.30):
        delta = np.array(point)-me.xy
        distance = float(np.linalg.norm(delta))
        if distance < .04:
            return MotorCommand("tactics", "walk")
        velocity = local(delta/max(distance, 1e-9)*speed, me.yaw)
        angle = wrap(math.atan2(delta[1], delta[0])-me.yaw)
        return MotorCommand("tactics", "walk", (*velocity, float(np.clip(2.5*angle,-1.,1.))))

    def choose(self, now, public, alive):
        me = public[self.name]
        recovery = self.guard.choose(now, me)
        if recovery is not None:
            if self.action is not None:
                self.finish(now, "RECOVERY_PREEMPTION")
            self.phase, self.skill = "STABILIZE", "recover_guard"
            return MotorCommand("recovery", recovery)
        if now < .8:
            return MotorCommand("settle")
        if self.target not in alive or (self.action is None and now-self.last_selector_time >= 1.2):
            target = choose_target(self.name, public, self.teams, alive, self.target)
            if target != self.target:
                self.event(now, "TARGET_CHANGED", previous=self.target, target=target)
                self.target = target
        if self.target is None:
            return MotorCommand("no_live_opponent")
        other = public[self.target]
        _, dist, angle, closing = relative(me, other)
        # A safety maneuver steers inward using current geometry, not a hidden
        # root correction. It can fail, and the referee still sees real falls.
        if me.boundary_distance < .07 and self.action != 7:
            self.finish(now, "BOUNDARY_PREEMPTION")
            self.action, self.since, self.phase_since = 7, now, now
            self.origin, self.phase, self.skill = np.array(me.xy), "ACTIVE", "center"
            self.event(now, "BOUNDARY_ESCAPE")
        if self.action is None:
            mask, context = self.legal(me, other, now)
            others = [(b,self.teams[n] == self.team) for n,b in public.items() if n not in (self.name,self.target) and n in alive]
            obs = features(me, other, others, self.c.radius_m, now-self.last_attack_end)
            action, info = self.selector.select(obs, mask, context)
            if type(action) is not int or not 0 <= action < len(ACTIONS) or not mask[action]:
                raise ValueError("Tactical selector emitted an illegal action")
            self.action, self.since, self.phase_since = action, now, now
            self.origin, self.aim = np.array(me.xy), me.yaw+angle
            self.last_selector_time = now
            self.phase = "WINDUP" if action == 3 else "ACTIVE"
            self.skill = ("approach_orbit", "approach_orbit", "approach_orbit", "dash_bump", "evade_step", "evade_step", "brace_rebound", "center", "retreat")[action]
            self.decision = dict(time_s=now, actor=self.name, target=self.target, action=action,
                                 observation=obs.tolist(), mask=mask.tolist(), **info)
            if action == 3:
                self.serial += 1
                self.attack_id, self.contact_since = f"{self.name}:{self.serial}", None
            self.event(now, "MANEUVER", skill=self.skill, phase=self.phase, target=self.target,
                       attack_id=self.attack_id if action == 3 else None, action=ACTIONS[action])
        elapsed, phase_elapsed = now-self.since, now-self.phase_since
        moved = float(np.linalg.norm(np.array(me.xy)-self.origin))
        action = self.action
        if action == 3:
            head = (self.c.dash_neck_rad, self.c.dash_head_rad, 0., 0.)
            if self.phase == "WINDUP":
                if phase_elapsed >= .12-1e-9:
                    self.phase, self.phase_since = "ACTIVE", now
                    self.origin = np.array(me.xy)
                    self.event(now, "ATTACK_LAUNCHED", attack_id=self.attack_id, target=self.target)
                return MotorCommand("maneuver", "walk", head=head)
            if self.phase == "ACTIVE":
                if me.touching_opponent and self.contact_since is None:
                    self.contact_since = now
                contacted = self.contact_since is not None and now-self.contact_since >= self.c.dash_contact_min_s and closing < .05
                if phase_elapsed >= self.c.dash_active_s or moved > .27 or contacted:
                    self.phase, self.phase_since = "RECOVERY", now
                    self.event(now, "ATTACK_ACTIVE_ENDED", attack_id=self.attack_id, target=self.target)
                    return MotorCommand("maneuver", "walk")
                motor = "dash_candidate" if self.c.candidate_motor_path else "walk"
                return MotorCommand("maneuver", motor, (self.c.dash_speed, 0., float(np.clip(2*wrap(self.aim-me.yaw),-.5,.5))), head)
            if phase_elapsed >= .3 and (dist > .33 or phase_elapsed >= 1.0) and me.upright > .90:
                self.finish(now)
                return MotorCommand("maneuver", "walk")
            # The dash-only candidate is never used for reverse/side commands.
            return MotorCommand("maneuver", "walk", (-.30 if dist < .33 and phase_elapsed > .16 else 0., 0., 0.))
        if action in (4,5):
            side = 1 if action == 4 else -1
            if self.phase == "ACTIVE" and (elapsed >= 1.15 or moved >= self.c.evade_distance_m):
                self.phase, self.phase_since = "RECOVERY", now
                self.event(now, "EVADE_BRAKE", target=self.target)
            if self.phase == "ACTIVE":
                return MotorCommand("maneuver", "walk", (self.c.evade_forward_speed, side*self.c.evade_speed, 0.))
            if phase_elapsed >= .24:
                self.finish(now)
            return MotorCommand("maneuver", "walk")
        duration = {0:.40, 1:.55, 2:.55, 6:.50, 7:.50, 8:.50}[action]
        if elapsed >= duration-1e-9:
            self.finish(now)
            return MotorCommand("tactics", "walk")
        if action == 7:
            return self.navigate(me, (0.,0.))
        if action == 6:
            return MotorCommand("maneuver", "stand")
        if action == 8:
            return MotorCommand("maneuver", "walk", (-.30,0.,0.))
        turn = float(np.clip(2.8*angle,-1.,1.))
        if action in (1,2):
            return MotorCommand("tactics", "walk", (.15 if dist>.4 else -.15, .30 if action == 1 else -.30, turn))
        # A visible ally blocking the lane prompts actual sideways movement;
        # friendly collisions remain active throughout this attempt.
        allies = [public[n] for n in alive if n != self.name and self.teams[n] == self.team]
        if allies and min(np.linalg.norm(np.array(b.xy)-me.xy) for b in allies) < .22:
            near = min(allies, key=lambda b: np.linalg.norm(np.array(b.xy)-me.xy))
            direction = local(np.array(me.xy)-near.xy, me.yaw)
            return MotorCommand("tactics", "walk", (.12, .3 if direction[1]>=0 else -.3, turn))
        speed = .28 if dist > .29 or abs(angle) > .24 else 0.
        return MotorCommand("tactics", "walk", (speed,0.,turn))
