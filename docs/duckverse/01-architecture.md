# DG-00 architecture / 架构

Current narrow dependency path:

```mermaid
flowchart LR
    CLI[DG-01 experiment CLI] --> Arena[arena.world / free tiles + native duck]
    Arena --> Runtime[Existing DuckRuntime + PolicyBank]
    Runtime --> Servo[14 torque-limited servos / 50 Hz]
    Servo --> Physics[One MuJoCo model and data]
    CLI --> Weld[Warning events / eq_active release]
    Weld --> Physics
    Physics --> Record[Full integration state and per-step contacts]
    Record --> Replay[Input integration replay]
    Replay --> Video[Read-only single-shot renderer]
```

Current code is a calibration experiment: one fixed neighbour target, fixed public warning times, two support releases. It does not contain a survival planner, winner, multiplayer or ROSClaw adapter. The controller reads current pose and issues motor-policy commands. Tile release changes equality activity only.

Future layers, introduced behind separate measured gates: schedule/referee → 2/4 duck assembly → public observation and baseline survivor → real ROSClaw SIMULATION app → optional Jev and Practice lineage → cinematic director. No empty modules for future capabilities are added. Existing single-duck Neon Escape world is not repurposed.

Native Agent is a task-level caller, not the motor controller. A future adapter must use current core descriptors/action boundaries, link real Body instances, task IDs and terminal receipts. DG-01 local run IDs are not ROSClaw mission IDs.
