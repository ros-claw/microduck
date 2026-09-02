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
        "sitstand": str(POL / "alpha_sitstand.onnx")}


def main():
    s = RopeSkipSession(str(ROBOT), BANK, SkipConfig(seed=0))
    s.settle()
    assert all(d.is_upright() for d in s.ducks.values()), "ducks must be upright after settle"
    s.start_rope()
    m = s.run_episode(10.0)
    print(f"crossings={m.crossings} skips={m.successful_skips} trips={m.trips} "
          f"consec={m.consecutive_best} up(t/j)={m.turner_upright_frac:.0%}/"
          f"{m.jumper_upright_frac:.0%}")
    assert m.crossings >= 4, "rope must rotate (≥4 passes)"
    assert m.successful_skips >= 1, "jumper must clear at least one pass"
    assert m.turner_upright_frac > 0.8, "turners must stay up"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
