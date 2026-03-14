from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

from server.runtime import PacketEnvelope
from server.services import DepthEstimationService


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
        self._frame_delay_ms = frame_delay_ms
        self._headless_preview_path = Path(headless_preview_path)
        self._depth_estimator = depth_estimator
        self._window_name = window_name

        self._lock = threading.Lock()
        self._latest_envelope: PacketEnvelope | None = None
        self._latest_seq = 0
        self._rendered_seq = 0

        self._stop_event = threading.Event()
        self._quit_requested = threading.Event()
        self._worker_thread: threading.Thread | None = None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested.is_set()

    @property
    def depth_estimator(self) -> DepthEstimationService | None:
        return self._depth_estimator

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._quit_requested.clear()
        self._worker_thread = threading.Thread(
            target=self._run,
            name="server-preview-renderer",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        if self._enable_gui:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    def publish(self, envelope: PacketEnvelope) -> None:
        with self._lock:
            self._latest_envelope = envelope
            self._latest_seq += 1

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame_to_render: np.ndarray | None = None
            with self._lock:
                if self._latest_envelope is not None and self._latest_seq != self._rendered_seq:
                    envelope = self._latest_envelope
                    self._rendered_seq = self._latest_seq
                else:
                    envelope = None

            if envelope is not None:
                frame_to_render = _build_preview_frame(
                    envelope,
                    depth_estimator=self._depth_estimator,
                )

            if frame_to_render is not None:
                self._render(frame_to_render)

            if self._enable_gui:
                key = cv2.waitKey(self._frame_delay_ms) & 0xFF
                if key == ord("q"):
                    self._quit_requested.set()
                    self._stop_event.set()
                    break
            else:
                self._stop_event.wait(timeout=0.05)

    def _render(self, frame_bgr: np.ndarray) -> None:
        if self._enable_gui:
            cv2.imshow(self._window_name, frame_bgr)
            return

        self._headless_preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self._headless_preview_path), frame_bgr)


def _build_preview_frame(
    envelope: PacketEnvelope,
    *,
    depth_estimator: DepthEstimationService | None = None,
) -> np.ndarray | None:
    packet = envelope.packet
    source_frame = _decode_jpeg(packet.image_jpeg)
    if source_frame is None:
        return None

    frame = source_frame.copy()
    yolo_output = packet.yolo_output if packet.HasField("yolo_output") else None
    if yolo_output is not None:
        for detection in yolo_output.detections:
            if len(detection.xyxy) != 4:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{detection.class_name}:{detection.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    lines = [
        f"ts={packet.timestamp_ns}",
        f"client={envelope.client_peer}",
        f"detections={len(yolo_output.detections) if yolo_output is not None else 0}",
    ]
    if packet.HasField("orb_tracking") and packet.orb_tracking.tracking_state:
        lines.append(f"tracking={packet.orb_tracking.tracking_state}")

    annotated_frame = _overlay_text(frame, lines)
    if depth_estimator is None or not depth_estimator.enabled:
        return annotated_frame

    try:
        depth_result = depth_estimator.estimate(source_frame)
    except Exception as exc:
        depth_lines = [f"depth_error={exc}"]
        return _overlay_text(annotated_frame, depth_lines + lines)

    if depth_result is None:
        depth_lines = []
        if depth_estimator.last_error:
            depth_lines.append(f"depth_error={depth_estimator.last_error}")
        return _overlay_text(annotated_frame, depth_lines + lines)

    depth_vis = depth_estimator.colorize(depth_result.depth)
    if depth_vis.shape[:2] != annotated_frame.shape[:2]:
        depth_vis = cv2.resize(
            depth_vis,
            (annotated_frame.shape[1], annotated_frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    depth_lines = [
        f"depth_model={depth_result.model_id}",
        f"depth_device={depth_result.device}",
        f"depth_ms={depth_result.inference_ms:.1f}",
    ]
    depth_panel = _overlay_text(depth_vis, depth_lines)
    return np.hstack([annotated_frame, depth_panel])


def _decode_jpeg(encoded_jpeg: bytes) -> np.ndarray | None:
    if not encoded_jpeg:
        return None
    encoded = np.frombuffer(encoded_jpeg, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _overlay_text(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    for index, line in enumerate(lines):
        y = 28 + index * 22
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out
