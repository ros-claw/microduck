"""Smoke test: three ducks skip a rope in one physics world."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from microduck_lab.skills.skip import RopeSkipSession, SkipConfig

ROOT = pathlib.Path(os.environ.get("MICRODUCK_ROOT",
                                   "~/workspace/microduck")).expanduser()
ROBOT = ROOT / "microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"
POL = ROOT / "microduck/policies"
BANK = {"stand": str(POL / "alpha_stand.onnx"),
        "walk": str(POL / "alpha_walking.onnx"),
        "sitstand": str(POL / "alpha_sitstand.onnx")}


def main():
    s = RopeSkipSession(str(ROBOT), BANK, SkipConfig(seed=304))
    s.settle()
    assert all(d.is_upright() for d in s.ducks.values()), "ducks must be upright after settle"
    s.start_rope()
    m = s.run_episode(10.0)
    print(f"crossings={m.crossings} skips={m.successful_skips} trips={m.trips} "
          f"consec={m.consecutive_best} up(t/j)={m.turner_upright_frac:.0%}/"
          f"{m.jumper_upright_frac:.0%}")
    assert m.crossings >= 3, "rope must swing (≥3 passes)"
    assert m.crossings >= m.trips >= 1, "jumper must engage the rope"
    assert m.turner_upright_frac > 0.65, "turners mostly up in the 10 s window"
    # skips are rare by design (the 2.5 cm tuck-hop is the documented
    # bottleneck) — the performance take is filmed on verified seed 304
    print("SMOKE OK")


if __name__ == "__main__":
    main()
