"""Freeze the actual local assets, source ancestry and available checkpoints."""

import argparse
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import yaml
from microduck_lab.combat.world import ROOT, ASSETS, ROBOT, POLICIES, digest
from microduck_lab.combat.episode import source_hashes, write_json


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main(out):
    path = Path(out)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    training = ASSETS / "microduck_rl"
    meshes = training / "src/mjlab_microduck/robot/microduck/assets"
    checkpoints = [
        p for p in (training / "logs").rglob("*.pt") if p.stat().st_size > 100000
    ]
    manifest = [
        dict(path=str(p.relative_to(training)), bytes=p.stat().st_size)
        for p in sorted(checkpoints)
    ]
    checkpoint_path = ROOT / "training/combat/checkpoints_manifest.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        checkpoint_path,
        dict(
            files=manifest,
            notes=[
                "Local task checkpoints are available; none is verified as the native alpha_walking/stand/getup PPO checkpoint.",
                "Wandb files around 130 bytes are metadata pointers, excluded.",
                "This inventory excludes the separate Duck Fu training worktree; new candidates are locked separately.",
            ],
        ),
    )
    result = dict(
        schema="duckfu.source-lock.v1",
        base_commit=git(ROOT, "merge-base", "HEAD", "origin/main"),
        source_parent_commit=git(ROOT, "rev-parse", "HEAD"),
        branch=git(ROOT, "branch", "--show-current"),
        reviewed_getup_commit="5db0586ca451454c09796f79a985ef924b0d6c4b",
        robot=dict(path=ROBOT, sha256=digest(ASSETS / ROBOT)),
        meshes={
            str(p.relative_to(meshes)): digest(p)
            for p in sorted(meshes.rglob("*"))
            if p.is_file()
        },
        policies={
            n: dict(
                path="microduck/policies/" + p,
                sha256=digest(ASSETS / "microduck/policies" / p),
            )
            for n, p in POLICIES.items()
        },
        candidate_motor_models={
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "policies/candidates").glob("*.onnx"))
        },
        configuration_files={
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "configs/combat").glob("*.yaml"))
            if not p.name.startswith("source.")
        },
        training=dict(
            commit=git(training, "rev-parse", "HEAD"),
            dirty_status=git(training, "status", "--porcelain"),
            local_diff_sha256=__import__("hashlib")
            .sha256(git(training, "diff", "HEAD").encode())
            .hexdigest(),
            checkpoint_count=len(manifest),
            checkpoint_bytes=sum(r["bytes"] for r in manifest),
        ),
        runtime_upstream_commit=git(ASSETS / "microduck", "rev-parse", "HEAD"),
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            versions={
                p: metadata.version(p)
                for p in ["mujoco", "onnxruntime", "numpy", "pyyaml", "scipy"]
            },
        ),
        sources=source_hashes(),
    )
    combat_training = ROOT.parent / "duck-fu-training"
    if combat_training.is_dir():
        result["combat_training"] = dict(
            base_commit=git(combat_training, "rev-parse", "HEAD"),
            branch=git(combat_training, "branch", "--show-current"),
            patch_sha256=digest(ROOT / "training/combat/duckfu.patch"),
            r1_patch_sha256=digest(ROOT / "training/combat/duckfu-r1.patch"),
        )
    path.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    print(
        json.dumps(
            dict(
                output=str(path),
                mesh_files=len(result["meshes"]),
                checkpoints=len(manifest),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.out)
