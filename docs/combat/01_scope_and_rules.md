# Duck Fu scope and rules

The full competition implements **1v1 and red/blue 2v2** on a fixed octagonal platform. `team_ringout_v1` uses real ring-out and a public ten-second down count. There are no collapsing tiles, traps, health points, external root pushes, scripted winners or eliminated-body deletion.

All robots decide from the same 50 Hz public snapshot before the shared physical clock advances. Each has one motor owner and its own continuous previous-action history. Team identity affects targeting and event attribution, never body mass, torque, friction, collision immunity or the referee.

Maneuvers commit to their launch direction. A physical low-head wind-up precedes the active bump; a trained zero-command stage brakes the body before switching motor skills. Motion/history and actual visible head pose can trigger a defensive replan while navigating. The opponent's hidden controller phase, actions and future seed are not observations. A public recovery guard preempts a maneuver when actual balance is lost. The unqualified upstream rolling recovery is disabled in the final profile.

`CONTACT` records actual positive-force contact. `CONTACTED_ATTACK` links contact to a particular active attack; it alone does not prove useful response. `VALID_STRIKE` additionally requires the measured directional impulse and observed target displacement or imbalance while the attacker remains upright. It is an observed response event, not a damage estimate. Friendly contacts remain physical but cannot earn an enemy-strike reward. Each attack is credited at most once. Misses and interrupted actions are retained separately.

Ring-out requires no stage foot support and trunk z below −0.15 m, held for 0.12 s. An ordinary airborne step is not elimination. The down count begins from actual tilt/body contact. Recovery requires upright >0.94, real foot support and horizontal speed <0.16 m/s held for 0.3 s. Changing the motor name does not clear the count. Bodies that are eliminated keep their last control targets and remain in the shared simulation.

A team wins only when the other team has no active participants. Simultaneous elimination is a draw. Timeout is a draw; a remaining numerical fault is `NO_CONTEST_PHYSICS`. All conditions are evaluated symmetrically. The video director cannot change an outcome.

The historical one-exchange `sparring_v1` lab remains available for paired motor experiments. A disclosed coach retreats after knocking its target down. This lab is not used as evidence of recovery under continued attack; interference stress and autonomous matches are reported separately.

[Current methods](README.md) · [Qualification and limitations](RESULTS.md) · [Historical first-round scope](EARLY_LAB.md)
