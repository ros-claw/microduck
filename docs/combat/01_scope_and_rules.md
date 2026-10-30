# CF-00–03 scope and candidate rules

The current deliverable is a **duel development lab**, using a disclosed,
one-exchange coach and a reactive public-state defender. It is not self-play,
a complete fighting game or four independent model agents.

`duel_lab.yaml` has a large fixed flat stage. `duel_v1.yaml` uses a fixed disc
(radius 0.7 m) and visible lower receiver. Neither contains collapsing tiles,
traps, trainer pushes or root assistance. Only two players are implemented;
team/learned/Jev tactical modes fail rather than silently falling back.

Attack commitment fixes heading at launch. Dodge uses observed distance and
relative closing speed, never the coach's future commands/phase. Windup,
active drive, braking, separation and stable support are recorded. Maximum
durations are safety bounds, not proof that the body has completed a skill.
A counter is presently a dodge followed by facing and another dash.

`CONTACT` means a measured contact. `CONTACTED_ATTACK` links an active attack
to it; **it is not a qualified effective strike**. Ground support and prolonged
contact are not scored as enemy damage. No health bar exists. Misses are
reported only for completed attack windows; interrupted actions are retained
as interrupted, not silently counted as completed misses or successes.

`sparring_v1`: after the opponent is down, the attacker retreats using its own
walking policy. Contacts remain enabled. This is not proof of recovery under
continued attack. `contact_stress_v1` omits that retreat, but recovery in that
mode has not passed a battery.

Referee candidate: no stage support plus trunk below −0.15 m held 0.12 s is
ring-out; a normal airborne step is not elimination. DOWN requires tilt and
non-foot stage contact. Recovery requires upright >0.94, real stage foot
support, speed <0.16 m/s held 0.3 s. Public down-count is 10 s. Re-labelling a
roll does not clear it. Simultaneous losses are handled before assigning a
winner; timeout is a draw. A physics fault is no-contest. Eliminated bodies
remain physical, with their last motor targets held.

These are development rules, not calibrated competition standards. No winner
is selected by film logic. Human viewing gate: **PENDING_HUMAN_REVIEW**.
