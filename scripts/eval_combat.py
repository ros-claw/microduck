"""Frozen motor-variant validation, not a self-play/game win-rate benchmark."""

import argparse
from dataclasses import dataclass, replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import math
import numpy as np
from combat_probe import trial
from microduck_lab.combat.config import load, strict_load
from microduck_lab.combat.episode import write_json


@dataclass(frozen=True)
class Protocol:
    schema: str = "duckfu.motor-validation.v1"
    baseline_config: str = "configs/combat/duel_lab.yaml"
    candidate_config: str = "configs/combat/duel_candidate.yaml"
    initial_states: int = 40
    seed_base: int = 1001
    effect_threshold_m: float = 0.01
    stable_upright: float = 0.94
    stable_speed_m_s: float = 0.16
    stable_hold_s: float = 0.3
    duration_s: float = 8.0

    def __post_init__(self):
        if (
            self.schema != "duckfu.motor-validation.v1"
            or not 1 <= self.initial_states <= 40
        ):
            raise ValueError("Unsupported motor protocol")
        if (
            self.effect_threshold_m != 0.01
            or self.stable_upright != 0.94
            or self.stable_speed_m_s != 0.16
            or self.stable_hold_s != 0.3
        ):
            raise ValueError("Development measurement thresholds are frozen")
        if not 8.0 <= self.duration_s <= 12.0:
            raise ValueError("Invalid bounded motor qualification duration")


def measure(row, protocol):
    path = Path(row["run"])
    v = np.load(path / "controls.npz")["body"]
    ai = ["lavender", "cream"].index(row["actor"])
    di = 1 - ai
    effect = 0.0
    if row["first_attack_s"] is not None:
        start = round(row["first_attack_s"] * 50)
        end = min(len(v), round(((row["first_attack_end_s"] or 8.0) + 1.0) * 50))
        effect = float(
            np.max(np.linalg.norm(v[start:end, di, :2] - v[start, di, :2], axis=1))
        )
    last = v[-round(protocol.stable_hold_s * 50) :, ai]
    stable = bool(
        np.all(
            (last[:, 3] > protocol.stable_upright)
            & (last[:, 4] < protocol.stable_speed_m_s)
            & (last[:, 6] > 0.5)
        )
    )
    row["target_effect_m"] = effect
    row["actor_final_stable"] = stable
    defender_last = v[-round(protocol.stable_hold_s * 50) :, di]
    row["defender_final_stable"] = bool(
        np.all(
            (defender_last[:, 3] > protocol.stable_upright)
            & (defender_last[:, 4] < protocol.stable_speed_m_s)
            & (defender_last[:, 6] > 0.5)
        )
    )
    row["first_attack_avoided"] = bool(
        row["first_attack_s"] is not None and not row["first_exchange_contact"]
    )
    row["stable_evade_pass"] = (
        row["first_attack_avoided"] and row["defender_final_stable"]
    )
    row["attack_effect_candidate_pass"] = bool(
        row["config"]["defense"] == "stand"
        and row["first_exchange_contact"]
        and effect >= protocol.effect_threshold_m
        and stable
    )
    row["physics_proposal_pass"] = bool(
        row["physical"]["penetration_max_m"] <= 0.003
        and row["physical"]["penetrating_p99_m"] <= 0.0015
        and not any(row["physical"]["solver_warnings"])
        and row["physical"]["finite"]
        and row["physical"]["clock_ok"]
        and row["physical"]["external_forces_zero"]
        and row["physical"]["model_unchanged"]
        and row["physical"]["torque_max_Nm"] <= 0.6405001
    )
    return row


def wilson(k, n):
    z = 1.95996398454
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.jobs <= 4:
        p.error("Use 1–4 CPU workers")
    protocol = strict_load(a.protocol, Protocol)
    base, physics = load(protocol.baseline_config)
    candidate, candidate_physics = load(protocol.candidate_config)
    if (
        candidate_physics != physics
        or not candidate.candidate_motor_path
        or base.candidate_motor_path
    ):
        raise ValueError(
            "Compare matched physical profiles with an explicit motor candidate"
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    groups = []
    rng = np.random.default_rng(protocol.seed_base)
    for i in range(protocol.initial_states):
        common = dict(
            separation_m=float(rng.uniform(0.43, 0.58)),
            lateral_offset_m=float(rng.uniform(-0.025, 0.025)),
            heading_rad=(i % 4) * math.pi / 2,
            swap=bool(i % 2),
            evade_sign=1 if i % 2 else -1,
            duration_s=protocol.duration_s,
            dash_speed=0.4,
        )
        for group, config in [
            ("baseline_stand", base),
            ("candidate_stand", candidate),
            ("candidate_evade", candidate),
            ("candidate_counter", candidate),
        ]:
            cfg = replace(config, **common, defense=group.split("_")[1])
            jobs.append(
                (
                    out / f"{i:02d}-{group}",
                    cfg,
                    physics,
                    protocol.seed_base + i,
                    False,
                    None,
                )
            )
            groups.append(group)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = [measure(r, protocol) for r in pool.map(trial, jobs)]
    summaries = {}
    for group in sorted(set(groups)):
        selected = [r for r, g in zip(rows, groups, strict=True) if g == group]
        n = len(selected)
        effect = sum(r["attack_effect_candidate_pass"] for r in selected)
        standing = group.endswith("_stand")
        summaries[group] = dict(
            runs=n,
            first_exchange_contact=sum(r["first_exchange_contact"] for r in selected),
            effect_and_stable=effect if standing else None,
            effect_and_stable_wilson95=wilson(effect, n) if standing else None,
            median_target_effect_m=float(
                np.median([r["target_effect_m"] for r in selected])
            )
            if standing
            else None,
            nonstanding_motion_is_self_actuated=not standing,
            first_attack_avoided=sum(r["first_attack_avoided"] for r in selected),
            stable_evades=sum(r["stable_evade_pass"] for r in selected),
            defender_final_stable=sum(r["defender_final_stable"] for r in selected),
            physics_pass=sum(r["physics_proposal_pass"] for r in selected),
            counter_contact=sum(
                any(
                    e["state"] == "CONTACTED_ATTACK" and e["actor"] == r["defender"]
                    for e in r["events"]
                )
                for r in selected
            ),
        )
    result = dict(
        schema=protocol.schema,
        protocol=__import__("dataclasses").asdict(protocol),
        summaries=summaries,
        rows=rows,
        paired_initial_states=protocol.initial_states,
        groups_per_initial_state=4,
        no_game_win_rate_claim=True,
        human_gate="PENDING_HUMAN_REVIEW",
        note="Motor skill variants include their head-command adapter; this is not a policy-only causal ablation.",
    )
    write_json(out / "report.json", result)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
