from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

from server.runtime.processing_models import ProcessedPacket
from server.services.depth_estimation_service import DepthEstimationService


class ServerPreviewRenderer:
    def __init__(
        self,
        *,
        enable_gui: bool,
        frame_delay_ms: int,
        headless_preview_path: Path,
        depth_estimator: DepthEstimationService | None = None,
        window_name: str = "Server Preview",
    ) -> None:
        self._enable_gui = enable_gui
        self._frame_delay_ms = max(1, int(frame_delay_ms))
        self._headless_ingress_preview_path = Path(headless_preview_path)
        self._headless_depth_preview_path = _derive_depth_preview_path(headless_preview_path)
        self._depth_estimator = depth_estimator
        self._ingress_window_name = f"{window_name} - Ingress"
        self._depth_window_name = f"{window_name} - Depth"

        self._lock = threading.Lock()
        self._latest_ingress_packet: ProcessedPacket | None = None
        self._latest_ingress_seq = 0
        self._rendered_ingress_seq = 0
        self._latest_depth_packet: ProcessedPacket | None = None
        self._latest_depth_seq = 0
        self._rendered_depth_seq = 0

        self._cached_depth_panel: np.ndarray | None = None
        self._cached_map_panel: np.ndarray | None = None
        self._initialized_windows: set[str] = set()
        self._has_rendered_ingress_frame = False
        self._has_rendered_depth_frame = False
        self._ingress_waiting_displayed = False
        self._depth_waiting_displayed = False
        self._quit_requested = False

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    @property
    def depth_estimator(self) -> DepthEstimationService | None:
        return self._depth_estimator

    @property
    def headless_ingress_preview_path(self) -> Path:
        return self._headless_ingress_preview_path

    @property
    def headless_depth_preview_path(self) -> Path:
        return self._headless_depth_preview_path

    def start(self) -> None:
        self._quit_requested = False
        self._has_rendered_ingress_frame = False
        self._has_rendered_depth_frame = False
        self._ingress_waiting_displayed = False
        self._depth_waiting_displayed = False
        if self._enable_gui:
            self._ensure_window(self._ingress_window_name)
            self._ensure_window(self._depth_window_name)

    def stop(self) -> None:
        self._quit_requested = False
        if self._enable_gui:
            for window_name in list(self._initialized_windows):
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass
        self._initialized_windows.clear()

    def publish_ingress(self, packet: ProcessedPacket) -> None:
        with self._lock:
            self._latest_ingress_packet = packet
            self._latest_ingress_seq += 1

    def publish_depth(self, packet: ProcessedPacket) -> None:
        with self._lock:
            self._latest_depth_packet = packet
            self._latest_depth_seq += 1

    def pump(self) -> None:
        if self._quit_requested:
            return

        ingress_packet: ProcessedPacket | None = None
        depth_packet: ProcessedPacket | None = None
        with self._lock:
            if (
                self._latest_ingress_packet is not None
                and self._latest_ingress_seq != self._rendered_ingress_seq
            ):
                ingress_packet = self._latest_ingress_packet
                self._rendered_ingress_seq = self._latest_ingress_seq
            if self._latest_depth_packet is not None and self._latest_depth_seq != self._rendered_depth_seq:
                depth_packet = self._latest_depth_packet
                self._rendered_depth_seq = self._latest_depth_seq

        if ingress_packet is not None:
            self._render_to_stream(
                window_name=self._ingress_window_name,
                frame_bgr=_build_ingress_preview_frame(ingress_packet),
                headless_path=self._headless_ingress_preview_path,
            )
            self._has_rendered_ingress_frame = True
            self._ingress_waiting_displayed = False
        elif not self._has_rendered_ingress_frame and not self._ingress_waiting_displayed:
            self._render_to_stream(
                window_name=self._ingress_window_name,
                frame_bgr=_build_waiting_frame(
                    title="Server Ingress Preview",
                    message="Waiting for decoded ingress frames...",
                ),
                headless_path=self._headless_ingress_preview_path,
            )
            self._ingress_waiting_displayed = True

        if depth_packet is not None:
            (
                depth_frame,
                self._cached_depth_panel,
                self._cached_map_panel,
            ) = _build_depth_preview_frame(
                depth_packet,
                depth_estimator=self._depth_estimator,
                cached_depth_panel=self._cached_depth_panel,
                cached_map_panel=self._cached_map_panel,
            )
            self._render_to_stream(
                window_name=self._depth_window_name,
                frame_bgr=depth_frame,
                headless_path=self._headless_depth_preview_path,
            )
            self._has_rendered_depth_frame = True
            self._depth_waiting_displayed = False
        elif not self._has_rendered_depth_frame and not self._depth_waiting_displayed:
            self._render_to_stream(
                window_name=self._depth_window_name,
                frame_bgr=_build_waiting_frame(
                    title="Server Depth Preview",
                    message="Waiting for selected depth/map updates...",
                ),
                headless_path=self._headless_depth_preview_path,
            )
            self._depth_waiting_displayed = True

        if self._enable_gui:
            key = cv2.waitKey(self._frame_delay_ms) & 0xFF
            if key == ord("q"):
                self._quit_requested = True

    def _render_to_stream(
        self,
        *,
        window_name: str,
        frame_bgr: np.ndarray,
        headless_path: Path,
    ) -> None:
        if self._enable_gui:
            self._ensure_window(window_name)
            cv2.imshow(window_name, frame_bgr)
            return
        headless_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(headless_path), frame_bgr)

    def _ensure_window(self, window_name: str) -> None:
        if window_name in self._initialized_windows:
            return
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self._initialized_windows.add(window_name)


def _build_ingress_preview_frame(processed_packet: ProcessedPacket) -> np.ndarray:
    packet = processed_packet.packet
    source_panel = _build_source_panel(processed_packet=processed_packet, label="Source")
    rectified_panel = _label_panel(processed_packet.rectified_frame.image_bgr.copy(), "Rectified")

    lines = [
        f"ts={packet.timestamp_ns}",
        f"client={processed_packet.envelope.client_peer}",
        f"detections={processed_packet.yolo_detection_count}",
        f"tracking={processed_packet.pose.tracking_state}",
        f"rectified={processed_packet.rectified_frame.rectified}",
    ]
    if processed_packet.pose.camera_position_xyz is not None:
        px, py, pz = processed_packet.pose.camera_position_xyz
        lines.append(f"cam_xyz=({px:+.2f}, {py:+.2f}, {pz:+.2f})")

    ingress_panel = _placeholder_panel(
        shape=processed_packet.rectified_frame.image_bgr.shape,
        label="Ingress",
        message="Packet accepted and stored. Depth runs on the selected-frame queue.",
    )
    ingress_panel = _overlay_text(ingress_panel, lines)

    target_height = max(source_panel.shape[0], rectified_panel.shape[0], ingress_panel.shape[0])
    panels = [
        _resize_to_height(source_panel, target_height),
        _resize_to_height(rectified_panel, target_height),
        _resize_to_height(ingress_panel, target_height),
    ]
    return cv2.hconcat(panels)


def _build_depth_preview_frame(
    processed_packet: ProcessedPacket,
    *,
    depth_estimator: DepthEstimationService | None,
    cached_depth_panel: np.ndarray | None,
    cached_map_panel: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_panel = _build_source_panel(processed_packet=processed_packet, label="Source")
    rectified_panel = _label_panel(processed_packet.rectified_frame.image_bgr.copy(), "Rectified")
    depth_panel = _build_depth_panel(
        processed_packet=processed_packet,
        depth_estimator=depth_estimator,
        cached_depth_panel=cached_depth_panel,
    )
    map_panel = _build_map_panel(
        processed_packet=processed_packet,
        cached_map_panel=cached_map_panel,
    )

    target_height = max(
        source_panel.shape[0],
        rectified_panel.shape[0],
        depth_panel.shape[0],
        map_panel.shape[0],
    )
    panels = [
        _resize_to_height(source_panel, target_height),
        _resize_to_height(rectified_panel, target_height),
        _resize_to_height(depth_panel, target_height),
        _resize_to_height(map_panel, target_height),
    ]
    return cv2.hconcat(panels), depth_panel, map_panel


def _build_source_panel(*, processed_packet: ProcessedPacket, label: str) -> np.ndarray:
    packet = processed_packet.packet
    frame = processed_packet.source_frame_bgr.copy()
    yolo_output = packet.yolo_output if packet.HasField("yolo_output") else None
    if yolo_output is not None:
        for detection in yolo_output.detections:
            if len(detection.xyxy) != 4:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label_text = f"{detection.class_name}:{detection.confidence:.2f}"
            cv2.putText(
                frame,
                label_text,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    lines = [
        f"ts={packet.timestamp_ns}",
        f"tracking={processed_packet.pose.tracking_state}",
        f"detections={processed_packet.yolo_detection_count}",
        f"points={processed_packet.projected_points_world.shape[0]}",
    ]
    if processed_packet.map_snapshot is not None:
        lines.append(f"map_rev={processed_packet.map_snapshot.revision}")
    if processed_packet.memory_record is not None:
        lines.append(f"memory_idx={processed_packet.memory_record.index_position}")
    return _label_panel(_overlay_text(frame, lines), label)


def _build_depth_panel(
    *,
    processed_packet: ProcessedPacket,
    depth_estimator: DepthEstimationService | None,
    cached_depth_panel: np.ndarray | None,
) -> np.ndarray:
    reference_frame = processed_packet.rectified_frame.image_bgr
    depth_result = processed_packet.depth_result
    if depth_result is not None and depth_estimator is not None and depth_estimator.enabled:
        depth_vis = depth_estimator.colorize(depth_result.depth)
        if depth_vis.shape[:2] != reference_frame.shape[:2]:
            depth_vis = cv2.resize(
                depth_vis,
                (reference_frame.shape[1], reference_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        depth_lines = [
            f"depth_model={depth_result.model_id}",
            f"depth_device={depth_result.device}",
            f"depth_ms={depth_result.inference_ms:.1f}",
        ]
        if processed_packet.reprojection_confidence_threshold is not None:
            depth_lines.append(f"conf_thr={processed_packet.reprojection_confidence_threshold:.2f}")
        return _label_panel(_overlay_text(depth_vis, depth_lines), "Depth")
    if cached_depth_panel is not None:
        return _stamp_stale_panel(cached_depth_panel, "Depth (cached)")
    if depth_estimator is not None and depth_estimator.enabled:
        return _placeholder_panel(
            shape=reference_frame.shape,
            label="Depth",
            message="Waiting for selected depth frames...",
        )
    return _placeholder_panel(shape=reference_frame.shape, label="Depth", message="Depth disabled")


def _build_map_panel(
    *,
    processed_packet: ProcessedPacket,
    cached_map_panel: np.ndarray | None,
) -> np.ndarray:
    reference_frame = processed_packet.rectified_frame.image_bgr
    if processed_packet.map_snapshot is not None:
        map_panel = processed_packet.map_snapshot.occupancy_bgr.copy()
        if map_panel.shape[:2] != reference_frame.shape[:2]:
            map_panel = cv2.resize(
                map_panel,
                (reference_frame.shape[1], reference_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        map_lines = [
            f"map_rev={processed_packet.map_snapshot.revision}",
            f"backend={processed_packet.map_snapshot.backend}",
        ]
        return _label_panel(_overlay_text(map_panel, map_lines), "Topdown")
    if cached_map_panel is not None:
        return _stamp_stale_panel(cached_map_panel, "Topdown (cached)")
    return _placeholder_panel(
        shape=reference_frame.shape,
        label="Topdown",
        message="Waiting for valid pose and depth map updates...",
    )


def _stamp_stale_panel(panel_bgr: np.ndarray, label: str) -> np.ndarray:
    panel = panel_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (24, 24, 24), thickness=-1)
    cv2.putText(
        panel,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(panel, (0, panel.shape[0] - 34), (panel.shape[1], panel.shape[0]), (24, 24, 24), thickness=-1)
    cv2.putText(
        panel,
        "cached result",
        (10, panel.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 210, 80),
        1,
        cv2.LINE_AA,
    )
    return panel


def _placeholder_panel(*, shape: tuple[int, int, int], label: str, message: str) -> np.ndarray:
    height, width = shape[:2]
    panel = np.full((height, width, 3), 26, dtype=np.uint8)
    cv2.putText(
        panel,
        message,
        (18, max(60, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return _label_panel(panel, label)


def _build_waiting_frame(*, title: str, message: str) -> np.ndarray:
    frame = np.full((520, 960, 3), 24, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (16, 16, 16), thickness=-1)
    cv2.putText(
        frame,
        title,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (36, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _label_panel(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    panel = frame_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (24, 24, 24), thickness=-1)
    cv2.putText(
        panel,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _resize_to_height(image_bgr: np.ndarray, target_height: int) -> np.ndarray:
    if image_bgr.shape[0] == target_height:
        return image_bgr
    scale = target_height / image_bgr.shape[0]
    width = max(1, int(round(image_bgr.shape[1] * scale)))
    return cv2.resize(image_bgr, (width, target_height), interpolation=cv2.INTER_LINEAR)


def _overlay_text(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    for index, line in enumerate(lines):
        y = 28 + index * 22
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _derive_depth_preview_path(ingress_preview_path: Path | str) -> Path:
    ingress_preview_path = Path(ingress_preview_path)
    return ingress_preview_path.with_name(
        f"{ingress_preview_path.stem}_depth{ingress_preview_path.suffix or '.jpg'}"
    )
