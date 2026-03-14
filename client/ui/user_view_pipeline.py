from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

from config import ClientConfig
from state import SharedClientState
from transport import VlmHttpClient
from user_interface import AssistantChatRenderer, UserViewRenderer


HEADLESS_PREVIEW_PATH = Path("/tmp/machine_perception_system_server_packet_preview.jpg")
LOCAL_PREVIEW_WINDOW_NAME = "Local Processing Preview"


class UserViewPipeline:
    def __init__(self, config: ClientConfig, shared_state: SharedClientState) -> None:
        self._config = config
        self._shared_state = shared_state
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._user_view_renderer = UserViewRenderer() if config.user_gui_enabled else None
        self._chat_renderer = (
            AssistantChatRenderer(
                client=VlmHttpClient(
                    host=config.vlm_host,
                    port=config.vlm_port,
                    timeout_s=config.vlm_timeout_s,
                ),
                top_k=config.chat_top_k,
                default_mode=config.chat_default_mode,
            )
            if config.user_gui_enabled
            else None
        )

    def start(self) -> None:
        if not self._config.any_gui_enabled:
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run,
            name="user-view-pipeline",
            daemon=True,
        )
        self._worker_thread.start()

    def run_foreground(self) -> None:
        if not self._config.any_gui_enabled:
            return
        self._stop_event.clear()
        self._run()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

    def _run(self) -> None:
        last_timestamp_ns: int | None = None
        try:
            while not self._stop_event.is_set() and not self._shared_state.stop_requested:
                snapshot = self._shared_state.get_latest_snapshot()
                if snapshot is not None and snapshot.synchronized_packet.timestamp_ns != last_timestamp_ns:
                    self._render_snapshot(snapshot)
                    last_timestamp_ns = snapshot.synchronized_packet.timestamp_ns
                elif self._user_view_renderer is not None and last_timestamp_ns is None:
                    self._user_view_renderer.render_waiting()

                if self._chat_renderer is not None:
                    self._chat_renderer.render(latest_timestamp_ns=last_timestamp_ns)

                if self._config.any_gui_enabled:
                    key = cv2.waitKeyEx(self._config.frame_delay_ms)
                    if self._chat_renderer is not None and self._chat_renderer.handle_key(key):
                        continue
                    if key == ord("q"):
                        self._shared_state.request_stop()
                        break
                else:
                    self._stop_event.wait(timeout=0.05)
        finally:
            self._close_windows()

    def _render_snapshot(self, snapshot) -> None:
        if self._config.gui_enabled:
            cv2.imshow(LOCAL_PREVIEW_WINDOW_NAME, _build_local_preview(snapshot.enriched_packet))
        else:
            cv2.imwrite(str(HEADLESS_PREVIEW_PATH), _build_local_preview(snapshot.enriched_packet))

        if self._user_view_renderer is not None:
            tracking_state = "NO_TRACK"
            if snapshot.enriched_packet.orb_tracking is not None:
                tracking_state = snapshot.enriched_packet.orb_tracking.tracking_state or "NO_TRACK"
            self._user_view_renderer.render(
                snapshot.synchronized_packet.image_bgr,
                overlay_lines=[
                    f"ts={snapshot.synchronized_packet.timestamp_ns}",
                    f"tracking={tracking_state}",
                ],
            )

    def _close_windows(self) -> None:
        if self._user_view_renderer is not None:
            self._user_view_renderer.close()
        if self._chat_renderer is not None:
            self._chat_renderer.close()
        if self._config.gui_enabled:
            try:
                cv2.destroyWindow(LOCAL_PREVIEW_WINDOW_NAME)
            except cv2.error:
                pass


def _build_local_preview(packet) -> np.ndarray:
    panels = [_label_panel(packet.image_bgr, "Input Frame")]
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

    return _overlay_text(composed, lines)


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
    cv2.putText(
        panel, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return panel


def _overlay_text(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image_bgr.copy()
    for i, line in enumerate(lines):
        y = 54 + i * 24
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
    return out
