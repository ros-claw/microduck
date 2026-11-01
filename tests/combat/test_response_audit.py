"""Temporal attribution negatives: no future impulse, no unbounded late credit."""

import gzip
import importlib.util
import json
from pathlib import Path


def make_run(tmp_path, moved_ticks):
    manifest = dict(duration_s=0.52, physics=dict(dt=0.001))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    events = [
        dict(
            state="ATTACK_LAUNCHED", actor="a", target="b", attack_id="a:1", time_s=0.02
        ),
        dict(state="ATTACK_ACTIVE_ENDED", actor="a", attack_id="a:1", time_s=0.12),
    ]
    (tmp_path / "runtime_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    episode = dict(
        actor_pair=["a", "b"],
        attacks=["a:1"],
        start_step=40,
        end_step=120,
        normal_impulse_on_second_Ns=[0.03, 0.0, 0.0],
        summed_normal_impulse_Ns=0.03,
    )
    (tmp_path / "contacts_summary.json").write_text(
        json.dumps(dict(episodes=[episode]))
    )
    with gzip.open(tmp_path / "tactical_decisions.jsonl.gz", "wt") as f:
        for tick in range(26):
            for actor in ["a", "b"]:
                body = dict(
                    xy=[0.0, 0.0]
                    if actor == "a"
                    else [0.2 + (0.012 if tick in moved_ticks else 0.0), 0.0],
                    velocity=[0.0, 0.0],
                    upright=1.0,
                )
                f.write(
                    json.dumps(dict(time_s=tick / 50, actor=actor, observation=body))
                    + "\n"
                )
    path = Path(__file__).parents[2] / "scripts/measure_combat_response.py"
    spec = importlib.util.spec_from_file_location("response_audit_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.measure(tmp_path)


def test_future_episode_impulse_cannot_qualify_an_earlier_transient(tmp_path):
    assert not make_run(tmp_path, {3, 4})["observed_responses"]


def test_small_physical_response_after_active_is_distinguished(tmp_path):
    result = make_run(tmp_path, {7, 8})["observed_responses"]
    assert len(result) == 1 and result[0]["after_active"]
    assert result[0]["response"]["time_s"] == 0.14


def test_displacement_beyond_frozen_lag_window_is_not_credited(tmp_path):
    assert not make_run(tmp_path, {23, 24})["observed_responses"]
