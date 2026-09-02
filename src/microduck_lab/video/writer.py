"""MP4 video writer (imageio-ffmpeg backend)."""

from __future__ import annotations

import numpy as np


class VideoWriter:
    def __init__(self, path: str, fps: int = 30):
        import imageio.v2 as imageio
        self._w = imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=8,
            macro_block_size=16, ffmpeg_log_level="error")

    def write(self, frame: np.ndarray):
        self._w.append_data(frame)

    def write_n(self, frame: np.ndarray, n: int):
        for _ in range(n):
            self._w.append_data(frame)

    def close(self):
        self._w.close()
