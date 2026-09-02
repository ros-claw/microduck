"""Practice recording: the factual execution layer (ROSClaw 'Practice').

Every episode writes one JSONL record with full lineage:
world config, skill params, seed, verdicts, metrics. Memory/Auto/Darwin
consume these — no Practice, no learning.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PracticeRecord:
    skill_id: str
    skill_version: str
    params: dict
    seed: int
    success: bool
    metrics: dict
    failure: str | None = None        # taxonomy: trip / no_rotation / fall / timeout
    coach_assist: bool = True          # declared training aid, never hidden
    duration_s: float = 0.0
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class PracticeLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: PracticeRecord):
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
