"""Record the classic overhead-rope skip video.

Usage:
  MUJOCO_GL=osmesa .venv/bin/python scripts/record_classic_skip.py

Writes classic_skip_full.mp4 (50 s, spin-up + skipping) to the repo root.
The policy is policies/ropehop_classic.onnx (the trained continuous hopper).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import imageio
from PIL import Image, ImageDraw

from microduck_lab.demos.classic_skip import run_classic_skip

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOP = ROOT / "policies" / "ropehop_classic.onnx"
OUT = ROOT.parent / "classic_skip_full.mp4"


def overlay(img, t, m):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 330, 44], fill=(0, 0, 0))
    dr.text((8, 6), f"t={t:4.1f}s  passes={m['passes']}", fill=(255, 255, 255))
    dr.text((8, 24), f"SKIPS={m['skips']}  trips={m['trips']}", fill=(120, 255, 120))
    return np.array(im)


if __name__ == "__main__":
    metrics, frames = run_classic_skip(None, seconds=50.0, hop_onnx=str(HOP),
                                       overlay_fn=overlay)
    print("metrics:", metrics)
    imageio.mimsave(str(OUT), frames, fps=25)
    print("wrote", OUT)
