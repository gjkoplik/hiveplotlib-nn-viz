"""Stitch PNG frames into a movie (mp4 or gif)."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio


def stitch(frame_paths: list[Path], out_path: Path, fps: int = 8) -> Path:
    """Stitch ``frame_paths`` (in order) into ``out_path``; ``.gif`` or ``.mp4``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [imageio.imread(p) for p in frame_paths]
    if out_path.suffix == ".gif":
        imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(out_path, frames, fps=fps)
    return out_path
