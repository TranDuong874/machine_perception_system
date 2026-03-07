from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import cv2
import numpy as np

from schema.SensorSchema import LocalSensorPacket, OrbResult, ServerPerceptionPacket, YoloResult
from .Trajectory3D import Trajectory3DVisualizer, TrajectorySample

if TYPE_CHECKING:
    from local_services import DetectionService, SlamService


HEADLESS_PREVIEW_PATH = Path("/tmp/machine_perception_system_server_packet_preview.jpg")
_TRAJECTORY_VISUALIZER: Trajectory3DVisualizer | None = None
_TRAJECTORY_VISUALIZER_DISABLED = False
_TRAJECTORY_VISUALIZER_DISABLE_REASON: str | None = None


def close_visualizers() -> None:
    global _TRAJECTORY_VISUALIZER
    if _TRAJECTORY_VISUALIZER is None:
        return
    try:
        _TRAJECTORY_VISUALIZER.close()
    finally:
        _TRAJECTORY_VISUALIZER = None


def collect_server_perception_packet(
    local_packet: LocalSensorPacket,
    detection_service: "DetectionService | None" = None,
    slam_service: "SlamService | None" = None,
    yolo_timeout_s: float = 2.0,
    slam_timeout_s: float = 1.0,
) -> ServerPerceptionPacket:
    yolo_output = detection_service.await_result(local_packet.timestamp_ns, timeout_s=yolo_timeout_s) if detection_service else None
    orb_tracking = slam_service.await_result(local_packet.timestamp_ns, timeout_s=slam_timeout_s) if slam_service else None

    if yolo_output is None:
        yolo_output = YoloResult(error="timeout")
    if orb_tracking is None:
        orb_tracking = OrbResult(tracking_state="TIMEOUT", error="timeout")

    return ServerPerceptionPacket(
        timestamp_ns=local_packet.timestamp_ns,
        image_bgr=local_packet.image_bgr,
        imu_samples=local_packet.imu_samples,
        orb_tracking=orb_tracking,
        yolo_output=yolo_output,
        segmentation_mask=None,
    )


def stream_server_perception_packet(packet: ServerPerceptionPacket) -> None:
    panels = [
        _label_panel(packet.image_bgr, "Input Frame"),
    ]

    if packet.yolo_output and packet.yolo_output.annotated_image_bgr is not None:
        panels.append(_label_panel(packet.yolo_output.annotated_image_bgr, "YOLO"))
    if packet.orb_tracking and packet.orb_tracking.tracking_image_bgr is not None:
        panels.append(_label_panel(packet.orb_tracking.tracking_image_bgr, "ORB Tracking"))

    composed = _compose_panels(panels)
    lines = [
        f"timestamp_ns={packet.timestamp_ns}",
        f"imu_samples={len(packet.imu_samples)}",
        f"tracking_state={packet.orb_tracking.tracking_state if packet.orb_tracking else 'PENDING'}",
        f"yolo_detections={len(packet.yolo_output.detections) if packet.yolo_output else 0}",
    ]

    if packet.orb_tracking and packet.orb_tracking.camera_translation_xyz is not None:
        tx, ty, tz = packet.orb_tracking.camera_translation_xyz
        lines.append(f"camera_xyz=({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
    if packet.yolo_output and packet.yolo_output.error:
        lines.append(f"yolo_error={packet.yolo_output.error}")
    if packet.orb_tracking and packet.orb_tracking.error:
        lines.append(f"orb_error={packet.orb_tracking.error}")

    frame = _overlay_text(composed, lines)
    _update_trajectory_visualizer(packet)
    if os.environ.get("MPS_ENABLE_GUI") == "1":
        cv2.imshow("Server Packet Preview", frame)
        return
    cv2.imwrite(str(HEADLESS_PREVIEW_PATH), frame)


def _update_trajectory_visualizer(packet: ServerPerceptionPacket) -> None:
    if os.environ.get("MPS_ENABLE_TRAJECTORY_3D", "0") != "1":
        return
    if os.environ.get("MPS_ENABLE_GUI") != "1":
        return
    if packet.orb_tracking is None or packet.orb_tracking.camera_translation_xyz is None:
        return

    visualizer = _get_trajectory_visualizer()
    if visualizer is None:
        return

    state = packet.orb_tracking.tracking_state or "UNKNOWN"
    sample = TrajectorySample(xyz=packet.orb_tracking.camera_translation_xyz, tracking_state=state)
    try:
        visualizer.update(sample)
    except Exception:
        _disable_trajectory_visualizer()


def _get_trajectory_visualizer() -> Trajectory3DVisualizer | None:
    global _TRAJECTORY_VISUALIZER
    if _TRAJECTORY_VISUALIZER_DISABLED:
        return None
    if _TRAJECTORY_VISUALIZER is not None:
        return _TRAJECTORY_VISUALIZER
    try:
        _TRAJECTORY_VISUALIZER = Trajectory3DVisualizer()
    except Exception as exc:
        _disable_trajectory_visualizer(f"trajectory visualizer init failed: {exc}")
        return None
    return _TRAJECTORY_VISUALIZER


def _disable_trajectory_visualizer(reason: str | None = None) -> None:
    global _TRAJECTORY_VISUALIZER_DISABLED, _TRAJECTORY_VISUALIZER, _TRAJECTORY_VISUALIZER_DISABLE_REASON
    _TRAJECTORY_VISUALIZER_DISABLED = True
    _TRAJECTORY_VISUALIZER = None
    if reason:
        _TRAJECTORY_VISUALIZER_DISABLE_REASON = reason
    if _TRAJECTORY_VISUALIZER_DISABLE_REASON:
        print(
            f"[ServerPacketBridge] 3D trajectory disabled: {_TRAJECTORY_VISUALIZER_DISABLE_REASON}",
            file=sys.stderr,
        )
        _TRAJECTORY_VISUALIZER_DISABLE_REASON = None


def _compose_panels(panels: list[np.ndarray]) -> np.ndarray:
    target_height = max(panel.shape[0] for panel in panels)
    resized = [_resize_to_height(panel, target_height) for panel in panels]
    return cv2.hconcat(resized)


def _resize_to_height(image_bgr: np.ndarray, target_height: int) -> np.ndarray:
    if image_bgr.shape[0] == target_height:
        return image_bgr
    scale = target_height / image_bgr.shape[0]
    width = max(1, int(round(image_bgr.shape[1] * scale)))
    return cv2.resize(image_bgr, (width, target_height), interpolation=cv2.INTER_LINEAR)


def _label_panel(image_bgr: np.ndarray, label: str) -> np.ndarray:
    panel = image_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (24, 24, 24), thickness=-1)
    cv2.putText(panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _overlay_text(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image_bgr.copy()
    for i, line in enumerate(lines):
        y = 54 + i * 24
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
    return out
