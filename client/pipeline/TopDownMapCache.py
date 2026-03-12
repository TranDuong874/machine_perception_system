from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from proto import perception_pb2


@dataclass(frozen=True)
class TopDownMapState:
    map_id: str
    revision: int
    timestamp_ns: int
    tracking_state: str
    pose_valid: bool
    resolution_m: float
    width: int
    height: int
    origin_x_m: float
    origin_z_m: float
    latest_user_translation_xyz: tuple[float, float, float] | None
    occupancy_bgr: np.ndarray | None
    backend: str


_LOCK = threading.Lock()
_LATEST: TopDownMapState | None = None


def update_from_reply(reply) -> None:
    if not reply.HasField("topdown_map"):
        return
    topdown = reply.topdown_map
    occupancy_bgr = _decode_png(topdown.occupancy_png)
    translation = None
    if len(topdown.latest_user_translation_xyz) == 3:
        translation = (
            float(topdown.latest_user_translation_xyz[0]),
            float(topdown.latest_user_translation_xyz[1]),
            float(topdown.latest_user_translation_xyz[2]),
        )

    state = TopDownMapState(
        map_id=topdown.map_id,
        revision=int(topdown.revision),
        timestamp_ns=int(topdown.timestamp_ns),
        tracking_state=topdown.tracking_state,
        pose_valid=bool(topdown.pose_valid),
        resolution_m=float(topdown.resolution_m),
        width=int(topdown.width),
        height=int(topdown.height),
        origin_x_m=float(topdown.origin_x_m),
        origin_z_m=float(topdown.origin_z_m),
        latest_user_translation_xyz=translation,
        occupancy_bgr=occupancy_bgr,
        backend=topdown.backend,
    )
    with _LOCK:
        global _LATEST
        _LATEST = state


def get_latest_topdown_map() -> TopDownMapState | None:
    with _LOCK:
        return _LATEST


def world_xz_to_pixel(state: TopDownMapState, world_x_m: float, world_z_m: float) -> tuple[int, int] | None:
    if state.width <= 0 or state.height <= 0 or state.resolution_m <= 0:
        return None
    col = int(round((world_x_m - state.origin_x_m) / state.resolution_m))
    row = int(round((state.origin_z_m - world_z_m) / state.resolution_m))
    if col < 0 or row < 0 or col >= state.width or row >= state.height:
        return None
    return col, row


def _decode_png(encoded_png: bytes) -> np.ndarray | None:
    if not encoded_png:
        return None
    arr = np.frombuffer(encoded_png, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
