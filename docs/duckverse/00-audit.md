# DG-00 audit / 现状审计

Audit date: 2026-10-08. All executions in this repository are SIMULATION.

## Locked sources and local reality

| Source | Version inspected | Actual reuse |
| --- | --- | --- |
| Delivery before this change | ae121efa3a50d6cf67e211866cb372f53888eba0 | Native-colour assets, runtime, recording patterns; existing demos untouched |
| Pollen microduck_rl lock | 5946fd9cdbc58956424420153e51975af3b30d77 | robot_allcollisions.xml and meshes |
| Local training checkout | 18052aaf7d98942921a8d57319f40d3692f96bcd | Robot XML has no diff against lock; custom training changes elsewhere preserved |
| Pollen microduck runtime | 9f7eaad1008fffd90ef871a33a18aecd066b51a9 | alpha_stand and alpha_walking ONNX |
| ROSClaw public HEAD inspected | 21838614bb14c39599b8acef731b2b64dad3b92a | Read-only integration audit; no arena integration yet |
| e-URDF-Zoo public HEAD inspected | 779b1adaf4a8d3da3e3d286b4bdffa0179ab9ad1 | Recursive tree contains no Microduck entry; Body registration remains a prerequisite |

The local `/workspace/rosclaw` directory is a collection of checkouts, not itself a repository. A local core checkout at `rosclaw/rosclaw_test/rosclaw` is 2c210fcec98bfdf25bbaf222fd77dab8da2857df; it is not treated as public HEAD. Its ToolDescriptorV2 schema distinguishes PHYSICAL_ACTION, SIMULATION and model_callable. No live core call was made; native game-session/task/receipt integration is unverified. No pretend Body manifest or successful chat receipt is supplied.

DG-01 runtime reuses `DuckSpec`, the composer namespace/attach approach, `DuckRuntime`, `PolicyBank`, `apply_current_limit`. It does not invoke the rope composer, whose floor would be invalid here. One model, one data, one robot. Four-robot independence remains DG-03 work.

## Capability matrix

| Capability | Evidence scope | Status |
| --- | --- | --- |
| Native materials | Original MJCF attached without tint | Implemented |
| Stand, neighbour walk, seam crossing | Current DG-01 physical runs | See measured report |
| Recovery/roll/jump | Previous parkour tests and releases | Not qualified on this arena |
| Four ducks / duck–duck collisions | Previous Circus assembly exists | Not qualified on collapsing tiles |
| Jev tactics | Historical Neon Escape live captures | Not connected here |
| ROSClaw chat / Practice / Darwin | Read-only source/interface audit | Not connected here |

## Physics and asset provenance

Python 3.13, MuJoCo 3.12.0, CPU ONNX inference. Actor contract: 61D observation, 14D action, 50 Hz. Servo torque limited to ±0.6405 Nm. Policy/XML digests, runtime versions, and actual performance are in [the evidence report](LAST_DUCK_STANDING.md). Geometry/model package comes from upstream; no silent migration to a new robot XML. Arena collision stiffness is explicitly changed in the new builder only.

Code Apache-2.0; upstream Pollen Robotics model license remains CC BY-SA-NC. Existing attribution in [THIRD_PARTY](../../THIRD_PARTY.md) is preserved.

## Risks and gaps

| Risk | Treatment / next gate |
| --- | --- |
| Soft weld/support sinking | Measure load drift; improve weld impedance without moving tile pose |
| Impact penetration | Per-step forces and signed distances; dt sweep and failure records |
| Low-speed walk stalls | Calibrate motor command; no root assistance |
| Tile gap traps foot | Measure actual seam crossing before increasing gap |
| Stale tactical choices / future schedule leaks | Later public-observation boundary and legality filter |
| Four-duck computation / collisions | DG-03 profile; no extrapolated performance claim |
| Body/Simulation app missing | DG-06 adapter/registration required before native-agent claims |

This audit makes no robot behavior changes outside the new arena. Existing frozen V1 reference is rechecked before publication.
