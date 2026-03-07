from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable


@dataclass(frozen=True)
class TrajectorySample:
    xyz: tuple[float, float, float]
    tracking_state: str


class Trajectory3DVisualizer:
    def __init__(self, max_points: int = 5000, redraw_every: int = 2) -> None:
        import matplotlib

        backend_override = os.environ.get("MPS_TRAJECTORY_BACKEND")
        if backend_override:
            matplotlib.use(backend_override, force=True)
        else:
            current_backend = str(matplotlib.get_backend()).lower()
            if "agg" in current_backend:
                # Try a GUI backend so the 3D figure can be shown.
                matplotlib.use("TkAgg", force=True)

        import matplotlib.pyplot as plt

        self._plt = plt
        self._max_points = max_points
        self._redraw_every = max(1, redraw_every)
        self._sample_count = 0

        self._track_points: list[tuple[float, float, float]] = []
        self._lost_points: list[tuple[float, float, float]] = []

        self._plt.ion()
        self._fig = self._plt.figure("ORB Trajectory 3D")
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_title("Camera Trajectory")

    def update(self, sample: TrajectorySample) -> None:
        state = sample.tracking_state
        point = sample.xyz

        if state == "TRACKING":
            self._track_points.append(point)
            if len(self._track_points) > self._max_points:
                self._track_points = self._track_points[-self._max_points :]
        elif state in {"RECENTLY_LOST", "LOST"}:
            self._lost_points.append(point)
            if len(self._lost_points) > self._max_points:
                self._lost_points = self._lost_points[-self._max_points :]

        self._sample_count += 1
        if self._sample_count % self._redraw_every == 0:
            self._redraw()

    def close(self) -> None:
        self._plt.close(self._fig)

    def _redraw(self) -> None:
        self._ax.clear()
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_title("Camera Trajectory")

        if self._track_points:
            xs, ys, zs = _unzip_xyz(self._track_points)
            self._ax.plot(xs, ys, zs, color="tab:blue", linewidth=1.5, label="TRACKING")
            self._ax.scatter(xs[-1], ys[-1], zs[-1], color="tab:green", s=30, label="Latest")

        if self._lost_points:
            lxs, lys, lzs = _unzip_xyz(self._lost_points)
            self._ax.scatter(lxs, lys, lzs, color="tab:red", marker="x", s=20, label="LOST")

        all_points = self._track_points + self._lost_points
        if all_points:
            _set_equal_axes(self._ax, all_points)

        self._ax.legend(loc="upper right")
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)


def _unzip_xyz(points: Iterable[tuple[float, float, float]]) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for x, y, z in points:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs


def _set_equal_axes(ax, points: list[tuple[float, float, float]]) -> None:
    xs, ys, zs = _unzip_xyz(points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    x_mid = (x_min + x_max) * 0.5
    y_mid = (y_min + y_max) * 0.5
    z_mid = (z_min + z_max) * 0.5

    span = max(x_max - x_min, y_max - y_min, z_max - z_min)
    radius = max(0.1, span * 0.55)

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(z_mid - radius, z_mid + radius)
