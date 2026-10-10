#!/usr/bin/env python3
"""Import an actual recorded SIM match into ROSClaw's existing Practice API.

This is a retrospective evidence import, never presented as live hardware
telemetry. Requires the candidate ROSClaw core and a genuine native receipt.
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, UTC, timedelta
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.event import RuntimeEvent
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.ids import generate_episode_id
from rosclaw.practice.storage.layout import generate_practice_id

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--native-evidence", required=True)
p.add_argument("--audit", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
evidence = json.loads(Path(a.native_evidence).read_text())
raw = Path(a.audit).read_bytes()
audit = json.loads(raw)
if hashlib.sha256(raw).hexdigest() != evidence["game_audit_sha256"]:
    raise ValueError("Audit does not match native execution evidence")
actions = evidence["actions"]
if (
    len(actions) != 1
    or actions[0]["state"] != "COMPLETED"
    or not actions[0]["receipt_id"]
):
    raise ValueError("A completed native action receipt is required")
practice = generate_practice_id()
episode = generate_episode_id()
session = "sess_" + practice
base = datetime.now(UTC)
task = evidence["task"]["task_id"]
mission = evidence["mission_id"]
metadata = {
    "trace_id": actions[0]["receipt_id"],
    "task_id": task,
    "robot_type": "microduck",
    "session_metadata": {
        "body_id": "microduck-lavender",
        "mission_id": mission,
        "receipt_id": actions[0]["receipt_id"],
        "run_id": evidence["game_run_id"],
        "recording_mode": "recorded_simulation_import",
        "evidence_domain": "simulation",
        "sim_clock_mapping": "import UTC base + recorded simulation seconds; not original wall-clock acquisition",
        "policy_id": "pollen-native-stand-walk",
        "policy_sha256": audit["policy_sha256"],
        "source_audit_sha256": evidence["game_audit_sha256"],
        "imported_at": base.isoformat(),
        "usable_for_real_execution": False,
    },
}
bus = RuntimeBus()
recorder = PracticeRecorder(
    bus, data_root=a.data_root, publish_to_event_bus=False, auto_start_on_skill=False
)
recorder.initialize()
recorder.start()


def publish(typ, payload, t=0, md=None):
    bus.publish(
        RuntimeEvent(
            type=typ,
            source="sandbox",
            robot="microduck",
            body_id="microduck-lavender",
            timestamp=base + timedelta(seconds=t),
            payload=payload,
            metadata=md or {"trace_id": actions[0]["receipt_id"], "task_id": task},
        )
    )


publish(
    "practice.start",
    {
        "practice_id": practice,
        "session_id": session,
        "episode_id": episode,
        "robot_id": "microduck-lavender",
        "robot_type": "microduck",
        "task_id": task,
        "task_name": "Last Duck Standing / recorded SIM import",
        "skill_id": "microduck.start_game",
        "sources": {"sandbox": True},
    },
    md=metadata,
)
publish(
    "duckverse.native_receipt",
    {
        "mission_id": mission,
        "task_id": task,
        "receipt_id": actions[0]["receipt_id"],
        "grant_id": actions[0]["grant_id"],
        "audit_sha256": evidence["game_audit_sha256"],
        "recording_mode": "recorded_simulation_import",
    },
)
for e in audit["events"]:
    publish(
        "duckverse." + e["state"].lower(),
        {
            "recorded_sim_time": e["time"],
            "actual_event": e,
            "evidence_domain": "simulation",
        },
        e["time"],
    )
publish(
    "duckverse.result",
    {
        "outcome": audit["outcome"],
        "policy_sha256": audit["policy_sha256"],
        "capture_hashes": audit["hashes"],
        "penetration_max_m": audit["penetration_max_m"],
    },
    audit["duration"],
)
publish(
    "practice.stop",
    {
        "outcome": "SUCCESS",
        "duration_ms": audit["duration"] * 1000,
        "reward": None,
        "failure_labels": [],
    },
    audit["duration"],
)
recorder.stop()
result = {
    "practice_id": practice,
    "session_id": session,
    "episode_id": episode,
    "data_root": str(Path(a.data_root).resolve()),
    "mission_id": mission,
    "task_id": task,
    "receipt_id": actions[0]["receipt_id"],
    "game_run_id": evidence["game_run_id"],
    "policy_sha256": audit["policy_sha256"],
    "recording_mode": "recorded_simulation_import",
    "evidence_domain": "simulation",
    "outcome_semantics": "SUCCESS means a completed recorded import; individual ducks can lose",
    "new_training": False,
    "policy_promotion": False,
    "darwin_used": False,
}
Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
