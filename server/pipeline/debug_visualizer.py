from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .runtime import PacketEnvelope, PipelineMetrics


@dataclass(frozen=True)
class DebugVisualizerConfig:
    window_name: str = "MPS Server Debug"
    draw_every_n: int = 1
    axis_scale_px: int = 90


class ServerDebugVisualizer:
    def __init__(self, config: DebugVisualizerConfig) -> None:
        self._cfg = config
        self._frame_index = 0
        cv2.namedWindow(self._cfg.window_name, cv2.WINDOW_NORMAL)

    def update(self, envelope: PacketEnvelope, metrics: PipelineMetrics) -> None:
        self._frame_index += 1
        if self._frame_index % max(1, self._cfg.draw_every_n) != 0:
            self.pump_events()
            return

        frame = self._compose_frame(envelope, metrics)
        cv2.imshow(self._cfg.window_name, frame)
        self.pump_events()

    def pump_events(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.close()
            raise RuntimeError("server debug visualizer closed by user")

    def close(self) -> None:
        cv2.destroyWindow(self._cfg.window_name)

    def _compose_frame(self, envelope: PacketEnvelope, metrics: PipelineMetrics) -> np.ndarray:
        packet = envelope.packet

        panels: list[np.ndarray] = []
        input_image = _decode_jpeg(packet.image_jpeg)
        if input_image is not None:
            panels.append(_label_panel(input_image, "Input"))

        if packet.HasField("yolo_output"):
            yolo_img = _decode_jpeg(packet.yolo_output.annotated_image_jpeg)
            if yolo_img is not None:
                panels.append(_label_panel(yolo_img, "YOLO"))

        if packet.HasField("orb_tracking"):
            orb_img = _decode_jpeg(packet.orb_tracking.tracking_image_jpeg)
            if orb_img is not None:
                panels.append(_label_panel(orb_img, "ORB Tracking"))

        if not panels:
            panels.append(_blank_panel("No image payload"))

        composed = _hconcat_with_resize(panels)

        quat = None
        if packet.HasField("orb_tracking") and len(packet.orb_tracking.camera_quaternion_wxyz) == 4:
            quat = list(packet.orb_tracking.camera_quaternion_wxyz)
        _draw_pose_axes(composed, quat, axis_scale_px=self._cfg.axis_scale_px)

        yolo_count = len(packet.yolo_output.detections) if packet.HasField("yolo_output") else 0
        tracking_state = packet.orb_tracking.tracking_state if packet.HasField("orb_tracking") else "NONE"
        lines = [
            f"ts={packet.timestamp_ns}",
            f"peer={envelope.client_peer}",
            f"tracking_state={tracking_state}",
            f"yolo_detections={yolo_count}",
            (
                "telemetry_processed="
                f"{metrics.telemetry_processed} rejected={metrics.rejected_packets} "
                f"router_drops={metrics.dropped_by_router}"
            ),
        ]
        return _overlay_text(composed, lines)


def _decode_jpeg(encoded: bytes) -> np.ndarray | None:
    if not encoded:
        return None
    arr = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def _label_panel(image_bgr: np.ndarray, label: str) -> np.ndarray:
    panel = image_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (20, 20, 20), thickness=-1)
    cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _blank_panel(message: str) -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(image, message, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    return image


def _hconcat_with_resize(panels: list[np.ndarray]) -> np.ndarray:
    target_height = max(panel.shape[0] for panel in panels)
    resized = []
    for panel in panels:
        if panel.shape[0] == target_height:
            resized.append(panel)
            continue
        scale = target_height / float(panel.shape[0])
        target_width = max(1, int(round(panel.shape[1] * scale)))
        resized.append(cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_LINEAR))
    return cv2.hconcat(resized)


def _overlay_text(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image_bgr.copy()
    for idx, line in enumerate(lines):
        y = 48 + idx * 24
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _draw_pose_axes(image_bgr: np.ndarray, quaternion_wxyz: list[float] | None, axis_scale_px: int = 90) -> None:
    h, w = image_bgr.shape[:2]
    center = np.array([w // 2, h // 2], dtype=np.float32)
    cv2.circle(image_bgr, tuple(center.astype(int)), 3, (255, 255, 255), -1)

    if quaternion_wxyz is None:
        return

    rotation = _quaternion_to_rotation_matrix(quaternion_wxyz)
    if rotation is None:
        return

    unit_axes = [
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), (0, 0, 255)),   # X red
        (np.array([0.0, 1.0, 0.0], dtype=np.float32), (0, 255, 0)),   # Y green
        (np.array([0.0, 0.0, 1.0], dtype=np.float32), (255, 0, 0)),   # Z blue
    ]

    for axis, color in unit_axes:
        rotated = rotation @ axis
        endpoint = _project_to_screen(rotated, center=center, scale_px=axis_scale_px)
        cv2.line(image_bgr, tuple(center.astype(int)), tuple(endpoint.astype(int)), color, 2, cv2.LINE_AA)


def _project_to_screen(vec3: np.ndarray, center: np.ndarray, scale_px: int) -> np.ndarray:
    depth = 2.5
    z = float(vec3[2])
    perspective = depth / max(0.25, depth + z)
    x = float(vec3[0]) * scale_px * perspective
    y = float(vec3[1]) * scale_px * perspective
    return np.array([center[0] + x, center[1] - y], dtype=np.float32)


def _quaternion_to_rotation_matrix(quaternion_wxyz: list[float]) -> np.ndarray | None:
    if len(quaternion_wxyz) != 4:
        return None
    w, x, y, z = [float(v) for v in quaternion_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
