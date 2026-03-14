from __future__ import annotations

from dataclasses import dataclass
import threading

import cv2
import numpy as np

from server.runtime.processing_models import ProcessedPacket, TopDownMapSnapshot


VALID_TRACKING_STATES = {"TRACKING", "TRACKING_KLT", "RECENTLY_LOST"}


@dataclass(frozen=True)
class TopDownMappingConfig:
    enabled: bool
    map_id: str
    resolution_m: float
    width: int
    height: int
    free_space_decrement: float
    occupied_increment: float


class TopDownMappingService:
    def __init__(self, *, config: TopDownMappingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._occupancy_scores = np.zeros((config.height, config.width), dtype=np.float32)
        self._origin_x_m: float | None = None
        self._origin_z_m: float | None = None
        self._latest_snapshot: TopDownMapSnapshot | None = None
        self._revision = 0

    def update(self, processed_packet: ProcessedPacket) -> TopDownMapSnapshot | None:
        if not self._config.enabled:
            return None

        with self._lock:
            snapshot = self._update_locked(processed_packet)
            self._latest_snapshot = snapshot
            return snapshot

    def snapshot(self) -> TopDownMapSnapshot | None:
        with self._lock:
            return self._latest_snapshot

    def _update_locked(self, processed_packet: ProcessedPacket) -> TopDownMapSnapshot:
        pose = processed_packet.pose
        pose_valid = pose.pose_valid and pose.tracking_state in VALID_TRACKING_STATES
        latest_translation = pose.camera_position_xyz if pose_valid else None
        if pose_valid and latest_translation is not None and self._origin_x_m is None:
            half_width_m = self._config.width * self._config.resolution_m / 2.0
            half_height_m = self._config.height * self._config.resolution_m / 2.0
            self._origin_x_m = latest_translation[0] - half_width_m
            self._origin_z_m = latest_translation[2] - half_height_m

        if (
            pose_valid
            and latest_translation is not None
            and self._origin_x_m is not None
            and self._origin_z_m is not None
        ):
            camera_cell = self._world_to_cell(latest_translation[0], latest_translation[2])
            if camera_cell is not None:
                for point_world in processed_packet.projected_points_world:
                    target_cell = self._world_to_cell(float(point_world[0]), float(point_world[2]))
                    if target_cell is None:
                        continue
                    self._apply_ray(camera_cell, target_cell)

        self._revision += 1
        occupancy_bgr = self._render_map_bgr(latest_translation)
        ok, encoded_png = cv2.imencode(".png", occupancy_bgr)
        if not ok:
            raise RuntimeError("failed to encode occupancy map PNG")

        origin_x_m = float(self._origin_x_m or 0.0)
        origin_z_m = float(self._origin_z_m or 0.0)
        return TopDownMapSnapshot(
            map_id=self._config.map_id,
            revision=self._revision,
            timestamp_ns=processed_packet.packet.timestamp_ns,
            tracking_state=pose.tracking_state,
            pose_valid=pose_valid,
            resolution_m=self._config.resolution_m,
            width=self._config.width,
            height=self._config.height,
            origin_x_m=origin_x_m,
            origin_z_m=origin_z_m,
            latest_user_translation_xyz=latest_translation,
            occupancy_png=encoded_png.tobytes(),
            occupancy_bgr=occupancy_bgr,
            backend="occupancy-grid-v1",
        )

    def _apply_ray(self, camera_cell: tuple[int, int], target_cell: tuple[int, int]) -> None:
        ray_cells = _bresenham_line(camera_cell[0], camera_cell[1], target_cell[0], target_cell[1])
        if not ray_cells:
            return

        for row, col in ray_cells[:-1]:
            self._occupancy_scores[row, col] -= self._config.free_space_decrement

        end_row, end_col = ray_cells[-1]
        self._occupancy_scores[end_row, end_col] += self._config.occupied_increment
        np.clip(self._occupancy_scores, -4.0, 4.0, out=self._occupancy_scores)

    def _world_to_cell(self, x_m: float, z_m: float) -> tuple[int, int] | None:
        if self._origin_x_m is None or self._origin_z_m is None:
            return None

        col = int(np.floor((x_m - self._origin_x_m) / self._config.resolution_m))
        row = self._config.height - 1 - int(
            np.floor((z_m - self._origin_z_m) / self._config.resolution_m)
        )
        if row < 0 or row >= self._config.height or col < 0 or col >= self._config.width:
            return None
        return row, col

    def _render_map_bgr(self, latest_translation: tuple[float, float, float] | None) -> np.ndarray:
        free = np.clip((-self._occupancy_scores) / 2.0, 0.0, 1.0)
        occupied = np.clip(self._occupancy_scores / 2.0, 0.0, 1.0)
        grayscale = 127.0 + (free * 128.0) - (occupied * 127.0)
        grayscale = np.clip(grayscale, 0.0, 255.0).astype(np.uint8)
        map_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

        if latest_translation is not None:
            latest_cell = self._world_to_cell(latest_translation[0], latest_translation[2])
            if latest_cell is not None:
                cv2.circle(map_bgr, (latest_cell[1], latest_cell[0]), 4, (0, 0, 255), thickness=-1)
        return map_bgr


def _bresenham_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    error = d_col - d_row

    row = row0
    col = col0
    while True:
        cells.append((row, col))
        if row == row1 and col == col1:
            break
        doubled_error = 2 * error
        if doubled_error > -d_row:
            error -= d_row
            col += step_col
        if doubled_error < d_col:
            error += d_col
            row += step_row
    return cells

