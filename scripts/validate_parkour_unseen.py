"""Freeze source, then evaluate explicit seeds once; never overwrite a cohort."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, hashlib, json, tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from microduck_lab.demos.parkour_rehearsal import run
from audit_parkour_contacts import audit
from replay_parkour_run import replay

ROOT = Path(__file__).resolve().parents[1]


def trial(job):
    seed, options = job
    output = Path(options["output"]) / str(seed)
    report = run(
        options["policy"],
        seed=seed,
        brain=options["brain"],
        capture=True,
        sweeper_phase=options["sweeper_phase"],
        sweeper_mass=options["sweeper_mass"],
        sweeper_torque=options["sweeper_torque"],
        seconds=35,
        output=output,
        hard_contacts=options["hard_contacts"],
        predictive=options["predictive"],
        difficulty=options["difficulty"],
        roll_policy=options["roll_policy"],
    )
    replay(output)
    audit(output)
    contacts = json.loads((output / "all-contacts.json").read_text())
    return dict(
        seed=seed,
        stunt_gate=report["passed"],
        publication_passed=report["passed"] and contacts["publication_gate"]["passed"],
        escape_publication_passed=report["component_course_passed"]
        and contacts["escape_contact_gate"]["passed"],
        component_course_passed=report["component_course_passed"],
        stage=report["stage"],
        finished=report["finished"],
        hp=report["audit"]["hp"],
        skills=report["skills"],
        contacts=contacts["categories"],
        publication_gate=contacts["publication_gate"],
        longest_chase_s=report["props"]["longest_chase_s"],
        decisions=[r for r in report["decisions"] if r["status"] == "accepted"],
    )


def digest(paths):
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--policy", default=str(ROOT / "policies/parkour_long_jump_v2.onnx"))
    p.add_argument("--roll-policy")
    p.add_argument("--brain", choices=["rule", "jev"], default="jev")
    p.add_argument("--sweeper-phase", type=float, default=0.0)
    p.add_argument("--sweeper-mass", type=float, default=0.4)
    p.add_argument("--sweeper-torque", type=float, default=0.2)
    p.add_argument("--hard-contacts", action="store_true")
    p.add_argument("--predictive", action="store_true")
    p.add_argument("--difficulty", choices=["classic", "chase"], default="classic")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    target = Path(a.output).resolve()
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError("Seeds must be unique")
    target.mkdir(parents=True, exist_ok=False)
    files = sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "src/microduck_lab/parkour").glob("*.json")
    )
    files += [
        ROOT / "scripts" / name
        for name in [
            "validate_parkour_unseen.py",
            "audit_parkour_contacts.py",
            "replay_parkour_run.py",
        ]
    ]
    manifest = digest(files)
    policies = {
        str(Path(name).resolve()): hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in [a.policy, a.roll_policy]
        if name
    }
    options = vars(a).copy()
    options["output"] = str(target)
    (target / "frozen-controller.json").write_text(
        json.dumps(dict(source=manifest, policies=policies, options=options), indent=2)
        + "\n"
    )
    with tarfile.open(target / "frozen-source.tar.gz", "w:gz") as bundle:
        for f in files:
            bundle.add(f, arcname=str(f.relative_to(ROOT)))
        for i, name in enumerate(policies):
            bundle.add(name, arcname=f"policies/evaluated_{i}.onnx")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(trial, [(seed, options) for seed in a.seeds]))
    if digest(files) != manifest:
        raise RuntimeError("Source changed during evaluation; cohort is invalid")
    if any(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha
        for name, sha in policies.items()
    ):
        raise RuntimeError("Policy changed during evaluation")
    result = dict(
        seeds=a.seeds,
        runs=rows,
        total=len(rows),
        escape_passed=sum(r["component_course_passed"] for r in rows),
        publication_passed=sum(r["publication_passed"] for r in rows),
        escape_publication_passed=sum(r["escape_publication_passed"] for r in rows),
        scope="Fixed source/policies and explicit new seeds; no retries. Every run includes matching input replay and all-contact audit. Not a hardware safety guarantee.",
    )
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
