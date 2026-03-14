from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


OverlayDrawFn = Callable[[np.ndarray], None]


class UserViewRenderer:
    def __init__(self, window_name: str = "User View") -> None:
        self.window_name = window_name
        self._window_initialized = False

    def render(
        self,
        image_bgr: np.ndarray,
        overlay_lines: list[str] | None = None,
        overlay_draw_fn: OverlayDrawFn | None = None,
    ) -> None:
        self._ensure_window()
        frame = image_bgr.copy()

        # Reserve a top HUD strip so overlay text/widgets can be added later.
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (16, 16, 16), thickness=-1)
        cv2.putText(
            frame,
            "Wearable View",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        if overlay_lines:
            y = 62
            for line in overlay_lines[:4]:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                y += 22

        if overlay_draw_fn is not None:
            overlay_draw_fn(frame)

        cv2.imshow(self.window_name, frame)

    def render_waiting(self) -> None:
        self._ensure_window()
        frame = np.full((540, 960, 3), 24, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (16, 16, 16), thickness=-1)
        cv2.putText(
            frame,
            "Wearable View",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Waiting for frames...",
            (24, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "The assistant chat window is available now.",
            (24, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, frame)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _ensure_window(self) -> None:
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._window_initialized = True
