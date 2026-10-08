# Contact experiments and failures / 失败记录

Final qualification is separated from all development probes in `artifacts/duckverse-game/development/`. Failed or incomplete runs are not hidden or counted as final passes. Development audits identify the parameters actually used; they are not all complete, frozen replay packages.

| Development case | Observed problem | Response |
| --- | --- | --- |
| Four-robot seed 101, 1 ms integration | 4.512 mm maximum contact depth; draw | Refined integration; a draw remained a valid result |
| Same seed, 0.5 ms | 2.976 mm | Continued crowded-contact testing rather than assuming one seed qualified everything |
| Native contact seed 105 | Foot interaction reached 6.498 mm | 0.25 ms integration and stiffer contact parameters |
| Pre-envelope two-robot battery | 9/10 quality passes; seed 508 reached 17.092 mm at tile–jaw contact | Native visual head and collision coverage disagreed; added external conservative bounds |
| Pre-envelope four-robot battery | Seed 605 reached 13.904 mm head–trunk self contact; seed 608 also failed | Kept native self collision, changed calibration and tested the convex collision path |
| First envelope prototype | World body accidentally included in own-body exclusions | Corrected exclusion enumeration to native body IDs 1..nbody; final tests explicitly forbid world exclusions |
| Corrected envelopes, native collision path | Seed 105 reached 8.003 mm; seed 301 reached 9.321 mm | Compared legacy libccd on the same known failure cases |
| Final libccd probes 508/105/301/605/101 | All ≤1.647 mm | Froze the source, then reran full 10/12 qualification and 32 new heldouts |

An attempted multi-CCD enum setting was unsupported in MuJoCo 3.12.0 and never produced a valid physical rollout. It is not counted as an experiment result. Some prototype imports ran while source was being edited; their source stamps are diagnostic metadata, not sufficient for strict frozen replay. The pre-envelope source snapshot is retained as `development/native-head-source.tar.gz`.

Seeds 10000+ were briefly used by an incomplete earlier holdout attempt before the final collision envelope calibration. They are development data. Final heldout seeds **20000–20031** were unused when the final physics/controller source was frozen. No parameter or actor-source change followed that battery. The subsequent selected captures use exactly the frozen source and repeat heldout seeds only for filming, without tuning the actors.

Final contact quality is a declared numerical threshold, not a claim of zero visual penetration. Native convex geometry plus conservative extra boxes is a modeling approximation. All final matches retain native self contacts and duck–duck/duck–tile/receiver contacts. Solver warnings, external forces, motor bounds and per-category worst contacts remain auditable.

Native ROSClaw initially rejected an MCP input schema without `additionalProperties:false`; the adapter now declares and validates a strict schema. A stale capability snapshot was refreshed through the supported native discovery tool. Neither failure was solved by bypassing admission. Only the subsequent real completed actions are advertised as successful native runs.

中文：失败实验单独保存，不计入最终通过率；旧的未冻结原型不能当成严格重放证据。最重要的改进是补足头部和躯干外部碰撞覆盖、纠正误排除世界接触、重新标定拥挤落地求解，并在冻结后使用全新的 32 个种子验证。最终仍是凸碰撞、保守包围盒和软接触近似，不宣传“零穿模”或真机效果。

A final GitHub clone and fresh upstream bootstrap exposed two historical hop tests whose scorer ignored `MICRODUCK_ROOT` and assumed sibling training checkouts. The r1 release fixes asset lookup in that scorer and its stand-policy test, without changing any captured actor/physics source. Fresh-environment validation then passes **105 tests**, and seed 101 strictly regenerates with zero state/control error using NumPy 2.5.3 and ONNX Runtime 1.30.0 (the original capture used 2.5.2 / 1.29.0). Original and final fresh reports remain distinct.
