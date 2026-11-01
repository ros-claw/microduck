"""V2 transparent rule combinations on unchanged learned 61D/14D motors.

This is bounded command composition, not newly learned transition weights.
The only opponent input is immutable public Body data.
"""

import math
import numpy as np
from .tactics import Pilot, ACTIONS, features, wrap, choose_target
from .observations import relative
from .maneuver import MotorCommand
from .transition_contracts import evaluate_entry, evaluate_continuation
from .parameterized_actions import ManeuverRequest
from .team_intents import flank_waypoint, escape_waypoint
from .command_envelope import facing_command, flank_ready
from .geometry import boundary_distance

V2_ACTIONS = ACTIONS + ("PROBE_BUMP", "CONTACT_EXIT")


class V2Pilot(Pilot):
    def __init__(self, name, team, teams, config, planner=None):
        super().__init__(name, team, teams, config)
        self.planner = planner
        self.motor_context = None
        self.role = "DUEL"
        self.style = config.red_v2_style if team == "RED" else config.blue_v2_style
        self.last_family = None
        self.contact_origin = None
        self.touch_start = None
        self.handoff_checks = []
        self.pending_counter = False
        self.side = 1
        self.unsupported_since = None
        self.observed_drive = False
        self.evade_min_distance = float("inf")
        self.last_visible_windup = -10.0
        self.entry_opportunity_logged = False
        self.executed_assignment = None

    def navigate(self, me, point, speed=0.30):
        delta = np.array(point) - me.xy
        distance = float(np.linalg.norm(delta))
        angle = wrap(math.atan2(delta[1], delta[0]) - me.yaw)
        return MotorCommand("tactics", "walk", facing_command(distance, angle))

    @property
    def active_attack(self):
        return (
            self.attack_id if self.action in (3, 9) and self.phase == "ACTIVE" else None
        )

    def finish(self, now, reason="COMPLETE", final=False):
        if self.action == 9:
            self.last_attack_end = now
            self.event(
                now,
                "ATTACK_WINDOW_ENDED",
                attack_id=self.attack_id,
                target=self.target,
                reason=reason,
            )
        if self.action in (3, 9):
            self.last_family = "committed_drive" if self.action == 3 else "probe_bump"
        super().finish(now, reason, final)

    def legal(self, me, other, now):
        base, context = super().legal(me, other, now)
        dist, _, angle, _, _, _ = context
        probe = (
            0.18 < dist < 0.27
            and abs(angle) < 0.32
            and now - self.last_attack_end > 0.40
            and me.upright > 0.96
            and me.foot_support
            and not me.body_ground
        )
        exit_ok = me.boundary_distance > 0.10 and me.upright > 0.9 and me.foot_support
        return np.r_[base, probe, exit_ok], context

    def begin(self, now, action, public):
        me, target = public[self.name], public[self.target]
        mask, context = self.legal(me, target, now)
        if not mask[action]:
            raise ValueError("Illegal V2 action at its actual entry state")
        self.action, self.since, self.phase_since = action, now, now
        self.origin, self.aim = np.array(me.xy), me.yaw + context[2]
        self.phase = "WINDUP" if action in (3, 9) else "ACTIVE"
        if action in (4, 5):
            self.observed_drive = False
            self.evade_min_distance = context[0]
            self.pending_counter = False
            self.entry_opportunity_logged = False
        self.brake.reset()
        self.skill = {
            0: "approach_orbit",
            1: "approach_orbit",
            2: "approach_orbit",
            3: "committed_drive",
            4: "cut_counter",
            5: "cut_counter",
            6: "brace_rebound",
            7: "center",
            8: "retreat",
            9: "probe_bump",
            10: "contact_exit",
        }[action]
        family = {
            3: "committed_drive",
            4: "cut_counter",
            5: "cut_counter",
            9: "probe_bump",
            10: "contact_exit",
            7: "center",
            6: "brace",
        }.get(action, "approach")
        request = ManeuverRequest(self.name, self.target, family, self.side, self.role)
        others = [
            (b, self.teams[n] == self.team)
            for n, b in public.items()
            if n not in (self.name, self.target)
        ]
        obs = features(me, target, others, self.c.radius_m, now - self.last_attack_end)
        self.decision = dict(
            time_s=now,
            actor=self.name,
            target=self.target,
            action=action,
            observation=obs.tolist(),
            mask=mask.tolist(),
            selector="rule_v2:" + self.style,
            log_probability=0.0,
            value=0.0,
            request=request.__dict__,
            observation_contract="v2-public-rule-context; not a V1 neural actor",
        )
        if action in (3, 9):
            self.serial += 1
            self.attack_id, self.contact_since = f"{self.name}:{self.serial}", None
        self.event(
            now,
            "MANEUVER",
            skill=self.skill,
            phase=self.phase,
            target=self.target,
            action=V2_ACTIONS[action],
            attack_id=self.attack_id if action in (3, 9) else None,
            role=self.role,
        )

    def choose(self, now, public, alive):
        me = public[self.name]
        recovery = self.guard.choose(now, me)
        if recovery is not None:
            if self.action is not None:
                self.finish(now, "RECOVERY_PREEMPTION")
            self.phase, self.skill = "STABILIZE", "recover_guard"
            return MotorCommand("recovery", recovery)
        if now < 0.8:
            return MotorCommand("settle")
        if self.c.action_course != "none" and self.serial > 0 and self.action is None:
            self.phase, self.skill = "READY", "completed_action_course"
            return MotorCommand("course_complete", "stand")
        assignment = self.planner.assignment(self.name) if self.planner else None
        candidate = (
            assignment.target
            if assignment
            else choose_target(self.name, public, self.teams, alive, self.target)
        )
        if candidate is None:
            return MotorCommand("no_live_opponent")
        safe_retarget = self.action is None or self.action in (0, 1, 2, 6, 7, 8, 10)
        if self.c.controller_revision == 2 and assignment:
            # A role is part of the maneuver request. Do not display a fresh
            # PEEL assignment while an old attack/evade is still committed.
            if safe_retarget or self.target not in alive:
                if assignment != self.executed_assignment:
                    self.event(
                        now,
                        "TEAM_ASSIGNMENT_ADOPTED",
                        target=assignment.target,
                        role=assignment.role,
                        intent=assignment.intent,
                        protected=assignment.protected,
                        plan_since_s=assignment.since_s,
                    )
                self.executed_assignment = assignment
            else:
                assignment = self.executed_assignment
        if self.target not in alive or (candidate != self.target and safe_retarget):
            previous = self.target
            if self.action is not None:
                self.finish(
                    now,
                    "TEAM_RETARGET" if self.target in alive else "TARGET_ELIMINATED",
                )
            self.target = candidate
            self.event(
                now,
                "TARGET_CHANGED",
                previous=previous,
                target=candidate,
                intent=assignment.intent if assignment else "DUEL",
            )
        if assignment:
            self.role = assignment.role
        other = public[self.target]
        _, dist, angle, closing = relative(me, other)
        closing_est = self.motion.update(now, me, other)
        head_cue = (
            other.neck_pitch_delta < -0.22
            and other.head_pitch_delta < -0.02
            and other.upright > 0.9
        )
        enemy_facing = abs(relative(other, me)[2])
        enemy_toward_us = -float(
            np.array(other.velocity) @ ((np.array(other.xy) - me.xy) / max(1e-9, dist))
        )
        threat = dist < 0.42 and (
            enemy_toward_us > 0.30 or (head_cue and enemy_facing < 0.50)
        )
        partners = [
            b
            for n, b in public.items()
            if n != self.name and n in alive and self.teams[n] == self.team
        ]
        ready_flank = bool(partners and flank_ready(me, other, partners[0]))
        if me.touching_opponent:
            if self.touch_start is None:
                self.touch_start, self.contact_origin = now, np.array(other.xy)
        else:
            self.touch_start, self.contact_origin = None, None
        stalled = (
            self.touch_start is not None
            and now - self.touch_start > self.c.exit_contact_after_s
            and np.linalg.norm(np.array(other.xy) - self.contact_origin) < 0.035
        )
        if self.action in (0, 1, 2, 6) and now - self.since > 0.12 and threat:
            self.finish(now, "PUBLIC_THREAT_REPLAN")
        if me.boundary_distance < 0.075 and self.action != 7:
            if self.action is not None:
                self.finish(now, "BOUNDARY_PREEMPTION")
            self.begin(now, 7, public)
        if self.action is None:
            mask, _ = self.legal(me, other, now)
            self.side = 1 if me.xy[1] <= 0 else -1
            if self.c.controller_revision == 2 and partners:
                lateral = np.array([-math.sin(me.yaw), math.cos(me.yaw)])
                # Body-local escape sides, scored in the real octagon. Global
                # y alone chose different behavior after rotating a match.
                self.side = max(
                    (1, -1),
                    key=lambda side: (
                        boundary_distance(
                            np.array(me.xy) + side * 0.22 * lateral, self.c
                        )
                        - sum(
                            max(
                                0.0,
                                0.28
                                - np.linalg.norm(
                                    np.array(b.xy)
                                    - (np.array(me.xy) + side * 0.18 * lateral)
                                ),
                            )
                            for b in partners
                        )
                    ),
                )
            if not mask[4 if self.side > 0 else 5]:
                self.side *= -1
            evade = 4 if self.side > 0 else 5
            if self.c.action_course != "none":
                requested = 9 if self.c.action_course == "probe_bump" else 3
                action = (
                    requested
                    if mask[requested]
                    else 0
                    if mask[0]
                    else 10
                    if mask[10]
                    else 6
                )
            elif assignment and self.role == "ESCAPE":
                action = 7
            elif (
                self.style != "pressure"
                and threat
                and mask[evade]
                and not (self.c.controller_revision == 2 and self.role == "PEEL")
            ):
                action = evade
            elif stalled and mask[10]:
                action = 10
            elif self.role == "FLANK" and not ready_flank:
                action = 1 if self.side > 0 else 2
            elif (
                mask[9]
                and (self.style != "pressure" or not mask[3])
                and self.last_family != "probe_bump"
            ):
                action = 9
            elif mask[3]:
                action = 3
            elif mask[9]:
                action = 9
            elif not mask[0] and mask[10] and abs(angle) > 0.4:
                action = 10
            elif me.touching_opponent and mask[10]:
                action = 10
            elif mask[0]:
                action = 0
            elif mask[8]:
                action = 8
            else:
                action = 6
            if not mask[action]:
                action = 6
            self.begin(now, action, public)
        action, elapsed = self.action, now - self.since
        phase_elapsed = now - self.phase_since
        moved = float(np.linalg.norm(np.array(me.xy) - self.origin))
        if action in (3, 9):
            probe = action == 9
            head = (
                (-0.30, -0.15, 0.0, 0.0)
                if probe
                else (self.c.dash_neck_rad, self.c.dash_head_rad, 0.0, 0.0)
            )
            windup = self.c.probe_windup_s if probe else self.c.dash_windup_s
            active = self.c.probe_active_s if probe else self.c.dash_active_s
            travel = self.c.probe_travel_m if probe else 0.27
            if self.phase == "WINDUP":
                if phase_elapsed >= windup - 1e-9:
                    self.phase, self.phase_since, self.origin = (
                        "ACTIVE",
                        now,
                        np.array(me.xy),
                    )
                    self.event(
                        now,
                        "ATTACK_LAUNCHED",
                        attack_id=self.attack_id,
                        target=self.target,
                    )
                return MotorCommand("maneuver", "walk", head=head)
            if self.phase == "ACTIVE":
                if me.touching_opponent and self.contact_since is None:
                    self.contact_since = now
                contact_done = (
                    self.contact_since is not None
                    and now - self.contact_since >= (0.16 if probe else 0.35)
                    and closing < 0.05
                )
                if phase_elapsed >= active - 1e-9 or moved >= travel or contact_done:
                    self.phase, self.phase_since = "RECOVERY", now
                    self.event(
                        now,
                        "ATTACK_ACTIVE_ENDED",
                        attack_id=self.attack_id,
                        target=self.target,
                    )
                    self.brake.reset()
                    return MotorCommand("maneuver", "dash_candidate")
                return MotorCommand(
                    "maneuver",
                    "dash_candidate",
                    (
                        self.c.dash_speed,
                        0.0,
                        float(np.clip(2 * wrap(self.aim - me.yaw), -0.5, 0.5)),
                    ),
                    head,
                )
            if not self.brake.ready(now, me):
                return MotorCommand("maneuver", "dash_candidate")
            self.finish(now)
            return MotorCommand("maneuver", "walk")
        if action in (4, 5):
            self.side = 1 if action == 4 else -1
            self.evade_min_distance = min(self.evade_min_distance, dist)
            forward_motion = float(
                np.array(other.velocity)
                @ np.array([math.cos(other.yaw), math.sin(other.yaw)])
            )
            if head_cue:
                self.last_visible_windup = now
            self.observed_drive |= (
                now - self.last_visible_windup < 0.65
                and forward_motion > 0.18
                and dist < 0.48
            )
            passed = (
                self.observed_drive
                and closing_est < 0.04
                and dist > self.evade_min_distance + 0.015
            )
            if self.phase == "ACTIVE":
                eligible = elapsed >= 0.34 and moved >= 0.055 and passed
                if eligible and not self.entry_opportunity_logged:
                    reference_check = evaluate_entry(me, self.motor_context)
                    if reference_check["accepted"]:
                        self.entry_opportunity_logged = True
                        self.event(
                            now,
                            "CANDIDATE_HANDOFF_AVAILABLE",
                            target=self.target,
                            contract=reference_check,
                            target_distance_m=dist,
                            target_angle_rad=angle,
                            dynamic_enabled=self.c.dynamic_handoff,
                        )
                if self.c.dynamic_handoff and eligible:
                    check = evaluate_entry(me, self.motor_context)
                    self.handoff_checks.append(dict(time_s=now, **check))
                    if check["accepted"]:
                        self.phase, self.phase_since = "PIVOT", now
                        self.unsupported_since = None
                        self.pending_counter = True
                        self.event(
                            now,
                            "DYNAMIC_HANDOFF",
                            source="EVADE",
                            successor="PIVOT_PROBE",
                            target=self.target,
                            contract=check,
                        )
                    else:
                        # Wait for the next loaded stance if the only missing
                        # entry prerequisite is momentary gait support.
                        if check["reasons"] != ["supported"]:
                            self.phase, self.phase_since = "RECOVERY", now
                            self.brake.reset()
                        self.event(
                            now, "HANDOFF_REJECTED", target=self.target, contract=check
                        )
                if self.phase == "ACTIVE" and (
                    (
                        elapsed >= 0.6
                        and dist > 0.31
                        and closing_est < 0.02
                        and not (
                            self.c.dynamic_handoff
                            and now - self.last_visible_windup < 0.65
                            and not self.observed_drive
                            and elapsed < 1.2
                        )
                    )
                    or elapsed >= 2.0
                    or moved >= 0.30
                ):
                    self.phase, self.phase_since = "RECOVERY", now
                    self.brake.reset()
                    self.event(now, "EVADE_BRAKE", target=self.target)
                if self.phase == "ACTIVE":
                    return MotorCommand(
                        "maneuver",
                        "evade_candidate",
                        (0.0, self.side * self.c.evade_speed, 0.0),
                    )
            if self.phase == "PIVOT":
                phase_elapsed = now - self.phase_since
                if me.foot_support:
                    self.unsupported_since = None
                elif self.unsupported_since is None:
                    self.unsupported_since = now
                unsupported = (
                    0.0
                    if self.unsupported_since is None
                    else now - self.unsupported_since
                )
                check = evaluate_continuation(me, self.motor_context, unsupported)
                if not check["accepted"]:
                    self.phase, self.phase_since = "RECOVERY", now
                    self.brake.reset()
                    self.event(now, "HANDOFF_ABORT", target=self.target, contract=check)
                    return MotorCommand("maneuver", "walk")
                mask, _ = self.legal(me, other, now)
                # Public geometry, not hidden enemy cooldown, determines a
                # potential counter. A real miss/contact chain is audited later.
                enemy_toward_us = -float(
                    np.array(other.velocity)
                    @ ((np.array(other.xy) - me.xy) / max(1e-9, dist))
                )
                counter_entry = evaluate_entry(me, self.motor_context)
                if (
                    mask[9]
                    and self.pending_counter
                    and enemy_toward_us < 0.08
                    and (self.c.controller_revision == 1 or counter_entry["accepted"])
                ):
                    self.finish(now, "DYNAMIC_HANDOFF")
                    self.begin(now, 9, public)
                    self.event(
                        now,
                        "COUNTER_ATTEMPT",
                        target=self.target,
                        source="PUBLIC_THREAT_EVADE",
                    )
                    return MotorCommand("maneuver", "walk")
                if phase_elapsed >= 2.0:
                    self.finish(now, "PIVOT_WINDOW_EXPIRED")
                    return MotorCommand("maneuver", "walk")
                # The qualified walking motor includes reverse + side + turn.
                # Move toward the public target while facing it, instead of
                # spending the whole opportunity rotating on the spot.
                return MotorCommand(
                    "maneuver",
                    "walk",
                    facing_command(dist, angle),
                    (-0.18, -0.08, 0.0, 0.0),
                )
            if not self.brake.ready(now, me):
                return MotorCommand("maneuver", "evade_candidate")
            self.finish(now)
            return MotorCommand("maneuver", "walk")
        duration = 0.8 if action == 10 else 0.4
        if elapsed >= duration:
            self.finish(now)
            return MotorCommand("tactics", "walk")
        if action == 7:
            if self.c.controller_revision == 2 and self.role == "ESCAPE":
                return self.navigate(
                    me, escape_waypoint(me, other, public, self.c.radius_m)
                )
            return self.navigate(me, (0.0, 0.0))
        if action == 10:
            return MotorCommand(
                "maneuver", "walk", (-0.15, self.side * 0.30, float(self.side))
            )
        if action == 8:
            return MotorCommand("maneuver", "walk", (-0.30, 0.0, 0.0))
        if action == 6:
            return MotorCommand("maneuver", "stand")
        if assignment and self.role == "RESET":
            return MotorCommand("tactics", "walk", (-0.30, 0.0, 0.0))
        if assignment and self.role == "FLANK" and not ready_flank:
            partners = [
                b
                for n, b in public.items()
                if n != self.name and n in alive and self.teams[n] == self.team
            ]
            if partners:
                point = flank_waypoint(me, other, partners[0], self.c.radius_m)
                return self.navigate(me, point, 0.30)
        turn = float(np.clip(2.8 * angle, -1.0, 1.0))
        if action in (1, 2):
            return MotorCommand(
                "tactics", "walk", (0.12, 0.26 if action == 1 else -0.26, turn)
            )
        # Use the same learned walking command envelope as the qualified V1.
        return MotorCommand("tactics", "walk", facing_command(dist, angle))
