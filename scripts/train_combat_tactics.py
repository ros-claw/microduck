"""Bounded CPU high-level categorical PPO with a frozen opponent per episode.

Native motor weights/physics/referee are immutable. This includes an explicitly
labelled rule-trajectory distillation warm start, then actual online self-play.
Every maneuver keeps its real duration for semi-MDP GAE and discounting.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import subprocess

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from microduck_lab.combat.config import strict_load
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.episode import write_json
from microduck_lab.combat.learning import export_tactics, semi_mdp_gae
from microduck_lab.combat.tactics import FEATURES, ACTIONS
from microduck_lab.combat.world import digest
from microduck_lab.combat.world import ROOT


class TacticalActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.policy, self.value = nn.Linear(64, len(ACTIONS)), nn.Linear(64, 1)

    def forward(self, obs):
        x = self.trunk(obs)
        return self.policy(x), self.value(x).squeeze(-1)


def collect(job):
    directory, cfg, seed, stochastic = job
    # Torch uses Python 3.12; every rollout uses the locked delivery runtime
    # (Python 3.13/MuJoCo 3.12), not training's MuJoCo 3.10 by accident.
    config_path = directory.with_suffix(".job.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, cfg.dict())
    command = [
        str(ROOT.parent / "rosclaw-microduck/.venv/bin/python"),
        str(ROOT / "scripts/combat_tactics_worker.py"),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--out",
        str(directory),
    ]
    if stochastic:
        command.append("--training")
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    if process.returncode:
        raise RuntimeError(f"Locked CPU rollout failed: {process.stderr[-2000:]}")
    result = json.loads((directory / "manifest.json").read_text())
    transitions = json.loads((directory / "transitions.json").read_text())
    events = [
        json.loads(s)
        for s in (directory / "runtime_events.jsonl").read_text().splitlines()
    ]
    red_trajectories = [
        transitions[n] for n, s in result["spawns"].items() if s["team"] == "RED"
    ]
    return dict(
        directory=str(directory),
        seed=seed,
        transitions=transitions["lavender"],
        trajectories=red_trajectories,
        result=result["result"],
        physical=result["physical"],
        strikes=sum(e["state"] == "VALID_STRIKE" for e in events),
        opponent_path=cfg.opponent_path,
        opponent_sha256=cfg.opponent_sha256,
        opponent_style=cfg.blue_style,
    )


def tensors(rows):
    return (
        torch.tensor(np.array([r["observation"] for r in rows]), dtype=torch.float32),
        torch.tensor(np.array([r["mask"] for r in rows]), dtype=torch.bool),
        torch.tensor([r["action"] for r in rows], dtype=torch.long),
    )


def distill_rules(model, games):
    rows = [r for g in games for trajectory in g["trajectories"] for r in trajectory]
    obs, mask, actions = tensors(rows)
    counts = torch.bincount(actions, minlength=len(ACTIONS)).clamp_min(1)
    weights = (len(rows) / counts.float()).sqrt()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(80):
        logits, _ = model(obs)
        loss = (
            nn.functional.cross_entropy(
                logits.masked_fill(~mask, -1e9), actions, reduction="none"
            )
            * weights[actions]
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return dict(
        teacher="public-state frozen rule trajectories; actor distillation only",
        samples=len(rows),
        action_counts=counts.tolist(),
        loss=float(loss.detach()),
    )


def ppo_update(model, optimizer, games):
    rows, advantages, returns = [], [], []
    for game in games:
        for trajectory in game["trajectories"]:
            if not trajectory:
                continue
            a, r = semi_mdp_gae(trajectory)
            rows.extend(trajectory)
            advantages.extend(a)
            returns.extend(r)
    obs, mask, action = tensors(rows)
    oldlog = torch.tensor([r["log_probability"] for r in rows])
    advantage, target = (
        torch.tensor(np.array(advantages)),
        torch.tensor(np.array(returns)),
    )
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    losses = []
    for _ in range(6):
        for batch in torch.randperm(len(rows)).split(256):
            logits, value = model(obs[batch])
            distribution = Categorical(logits=logits.masked_fill(~mask[batch], -1e9))
            logp = distribution.log_prob(action[batch])
            ratio = torch.exp(logp - oldlog[batch])
            actor_loss = -torch.minimum(
                ratio * advantage[batch], ratio.clamp(0.8, 1.2) * advantage[batch]
            ).mean()
            value_loss = (value - target[batch]).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + 0.5 * value_loss - 0.015 * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(
                [
                    float(actor_loss.detach()),
                    float(value_loss.detach()),
                    float(entropy.detach()),
                ]
            )
    return dict(
        transitions=len(rows),
        mean_losses=np.mean(losses, axis=0).tolist(),
        mean_return=float(
            np.mean([sum(r["reward"] for r in g["transitions"]) for g in games])
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=12)
    p.add_argument("--games-per-update", type=int, default=16)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=100001)
    p.add_argument("--resume-checkpoint", default="")
    a = p.parse_args()
    if not (
        1 <= a.updates <= 30 and 4 <= a.games_per_update <= 32 and 1 <= a.jobs <= 4
    ):
        p.error("Bounded local CPU experiment required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = strict_load(a.config, MatchConfig)
    if (
        cfg.tactics_path
        or cfg.opponent_path
        or (cfg.players == 4 and not a.resume_checkpoint)
    ):
        p.error(
            "Start from qualified motors; 2v2 resumes the qualified 1v1 tactical checkpoint"
        )
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    model = TacticalActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    started = time.monotonic()
    write_json(
        out / "budget.json",
        dict(
            updates=a.updates,
            games_per_update=a.games_per_update,
            jobs=a.jobs,
            maximum_wall_hours=2,
            gpu_count=0,
            maximum_checkpoints=2 * (a.updates + 1),
            cloud=False,
            seed_base=a.seed,
            motors_frozen=True,
            rules_frozen=True,
            initialization="rule distillation then on-policy PPO",
        ),
    )
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        if a.resume_checkpoint:
            checkpoint = torch.load(
                a.resume_checkpoint, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            warm = dict(
                initialization="resume 1v1 actor/critic/optimizer for parameter-shared team PPO",
                checkpoint_sha256=digest(a.resume_checkpoint),
            )
        else:
            warm_jobs = [
                (
                    out / f"teacher-{i:02d}",
                    replace(
                        cfg,
                        red_style=("pressure", "counter", "orbit")[i % 3],
                        blue_style=("pressure", "counter", "defensive")[i % 3],
                        swap=bool(i % 2),
                        heading_rad=(i % 4) * math.pi / 2,
                    ),
                    a.seed + i,
                    False,
                )
                for i in range(12)
            ]
            teacher = list(pool.map(collect, warm_jobs))
            warm = distill_rules(model, teacher)
        write_json(out / "distillation.json", warm)
        history = []
        for update in range(a.updates + 1):
            policy = out / f"tactics-{update:03d}.npz"
            export = export_tactics(model, policy)
            history.append((str(policy.resolve()), export["sha256"]))
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    update=update,
                    architecture="34-64tanh-64tanh-(categorical9,value1)",
                    config=cfg.dict(),
                    normalization="fixed physical scales",
                ),
                out / f"checkpoint-{update:03d}.pt",
            )
            write_json(out / f"export-{update:03d}.json", export)
            if update == a.updates:
                break
            rng = np.random.default_rng(a.seed + 10000 + update)
            jobs = []
            opponents = []
            for i in range(a.games_per_update):
                draw = float(rng.random())
                kind = (
                    "rule" if draw < 0.25 else ("latest" if draw < 0.5 else "history")
                )
                opponent = (
                    history[-1]
                    if kind == "latest"
                    else history[int(rng.integers(len(history)))]
                )
                common = dict(
                    tactics_path=str(policy.resolve()),
                    tactics_sha256=export["sha256"],
                    swap=bool(i % 2),
                    heading_rad=(i % 4) * math.pi / 2,
                    blue_style=("pressure", "counter", "orbit", "defensive")[i % 4],
                )
                if kind != "rule":
                    common.update(
                        opponent_path=opponent[0], opponent_sha256=opponent[1]
                    )
                candidate = replace(cfg, **common)
                jobs.append(
                    (
                        out / f"update-{update:03d}" / f"game-{i:02d}",
                        candidate,
                        a.seed + 1000 + update * 100 + i,
                        True,
                    )
                )
                opponents.append(
                    dict(
                        kind=kind,
                        sampling_probability=0.25
                        if kind in ("rule", "latest")
                        else 0.5,
                        sha256=common.get("opponent_sha256"),
                        frozen_entire_episode=True,
                    )
                )
            games = list(pool.map(collect, jobs))
            if any(
                not g["physical"]["finite"]
                or any(g["physical"]["solver_warnings"])
                or g["physical"]["penetration_max_m"] > 0.003
                or g["physical"]["penetrating_p99_m"] > 0.0015
                or not g["physical"]["external_forces_zero"]
                or not g["physical"]["model_unchanged"]
                for g in games
            ):
                raise RuntimeError("Numerical fault; stop before PPO update")
            metrics = ppo_update(model, optimizer, games)
            metrics.update(
                update=update,
                wall_s=time.monotonic() - started,
                opponents=opponents,
                results=[g["result"] for g in games],
                strikes=sum(g["strikes"] for g in games),
                actor_sha256=export["sha256"],
            )
            write_json(out / f"update-{update:03d}.json", metrics)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in metrics.items()
                        if k not in ("opponents", "results")
                    }
                ),
                flush=True,
            )
            if time.monotonic() - started > 7200:
                raise TimeoutError("Bounded tactics training exceeded two hours")
    write_json(
        out / "complete.json",
        dict(
            completed=True,
            policy_path=history[-1][0],
            sha256=history[-1][1],
            updates=a.updates,
            warm_start=warm,
            wall_s=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    main()
