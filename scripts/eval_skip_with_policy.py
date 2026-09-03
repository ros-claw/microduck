"""Measure the rope-skip success rate with a trained jump policy.

Usage: python scripts/eval_skip_with_policy.py policies/jump.onnx [--seeds ...]
Prints per-seed and aggregate skip rates — the Darwin metric.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from microduck_lab.skills.skip import RopeSkipSession, SkipConfig

ROOT = pathlib.Path("~/workspace/microduck").expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jump_onnx", type=str)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    ap.add_argument("--episodes-s", type=float, default=16.0)
    args = ap.parse_args()

    bank = {"stand": str(POL / "alpha_stand.onnx"),
            "walk": str(POL / "alpha_walking.onnx"),
            "sitstand": str(POL / "alpha_sitstand.onnx"),
            "jump": args.jump_onnx}

    tot_s = tot_c = tot_t = 0
    for seed in args.seeds:
        s = RopeSkipSession(str(ROBOT), bank, SkipConfig(seed=seed))
        s.settle(); s.start_rope()
        m = s.run_episode(args.episodes_s)
        tot_s += m.successful_skips; tot_c += m.crossings; tot_t += m.trips
        print(f"seed={seed}: skips={m.successful_skips}/{m.crossings} trips={m.trips} "
              f"consec={m.consecutive_best} up(t/j)={m.turner_upright_frac:.0%}/{m.jumper_upright_frac:.0%}",
              flush=True)
    print(f"\nTOTAL: {tot_s}/{tot_c} skips ({tot_s/max(1,tot_c):.0%}), trips={tot_t}")


if __name__ == "__main__":
    main()
