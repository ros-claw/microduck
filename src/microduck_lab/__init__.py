"""Microduck Lab — Teach ducks with words, not programs.

ROSClaw × Pollen Robotics Microduck showcase:
three ducks learn to skip a long rope together, closed loop.

    Chat → Act → Fail → Practice → Adapt → Darwin → Champion → Perform

Layered architecture (three timescales):
    Agent        ~1 Hz    goals, roles, rhythm negotiation, evolution
    Skills       5-20 Hz  rope_turn / jump / skip closed loops
    Policies     50 Hz    official ONNX policies → 14D actions → MuJoCo
"""

__version__ = "0.1.0"
