# Physics contract and collision coverage

CPU profile: Euler integration at 0.25 ms, 50 Hz motors (**80 physical substeps**),
Newton solver with 80 iterations, libccd convex contacts, contact `solref`
0.5 ms / damping ratio 1, explicit ±0.6405 Nm actuator force limits. Native
mass, inertia, position-servo gains, joint damping and materials are preserved.
Exact values and exclusions are exported in every run's `model_inventory.json`.

## Why the collision representation changed

The native head's visual-vertex coverage was relatively good (about 0.48%
outside its collision-hull union), but the torso had substantial uncovered
geometry: a convex-halfspace gap lower bound reached **59.7 mm** for one
visual vertex. This is a geometric coverage finding, **not measured dynamic
penetration**. Conversely, the old broad torso AABB has corners at least
43.4 mm beyond the visual convex hull; the head AABB reaches 36.2 mm.

Combat therefore uses a copy of **each of 70 visual mesh components per duck**
as a massless, unsimplified convex collision component for duck–duck contact.
Each component is convexified separately. These are still convex
approximations, not visual-triangle exact or deformable-shell collisions.
Concave gaps inside an individual component can still be filled by its hull.

Collision layers avoid duplicate force responses:

| Layer | Contacts |
| --- | --- |
| Stage/receiver type 1 | native ground-contact geoms only |
| Duck A native type 1→4, type 2→8 | original self contacts; type 4 also ground |
| Duck B native type 1→16, type 2→32 | original self contacts; type 16 also ground |
| A visual hull copies type 64 | B visual hull copies type 128 only |
| B visual hull copies type 128 | A visual hull copies type 64 only |

Native–native cross-duck and native–proxy duplicate contacts are disabled by
these declared layers; visual proxy self/ground contacts are excluded by the
same masks. The original self and ground responses are retained. No new
body-exclusion pairs or runtime collision changes are added. The structural
audit verifies geometry-copy bijection and exact inertia/actuation equality.

[Native/AABB coverage](../../artifacts/combat/collision-coverage-native.json)
· [New shells and mask audit](../../artifacts/combat/visual-parts-coverage.json).

## Contact evidence

Every physical step records or aggregates solved contact force, geom IDs,
position and normal at the pre-integration state. Post-step position is
separate. `mj_contactForce` is in the contact frame; world normal impulse is
transformed using that frame, with direction tested against two separating
free bodies. Pair episodes debounce brief separation; attacks are separately
identified so contact chatter is not scored repeatedly.

Contacts and proprioceptive sensor readings follow MuJoCo's last solved
stage (at most one physics step old); free-joint positions/velocities come
from the current integration state. All actors use the same snapshot and
declared contact age. This is ideal simulation state, not onboard perception.

Depth statistics use **only penetrating samples** for P99 and distinguish
duck–duck, duck–stage, self and receiver contacts. The initial proposed gate is
max ≤3 mm and penetrating P99 ≤1.5 mm. Passing that gate alone does not prove
visual accuracy, attack effectiveness or hardware validity.

A 0.30 s local impact replay at dt and dt/2 preserved contact and had about
3.8% normal-impulse difference and 0.484 mm largest final-position difference.
It is a fixed-input local numerical comparison, not whole-game equivalence.
[Actual result](../../artifacts/combat/impact-convergence.json).

No runtime root pose/velocity edits, external force, mass/friction/gain changes
or torque enlargement are used. Reset and read-only recorded-state rendering
are explicitly separate boundaries.
