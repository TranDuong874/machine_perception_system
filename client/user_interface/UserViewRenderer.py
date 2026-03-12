from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


OverlayDrawFn = Callable[[np.ndarray], None]


class UserViewRenderer:
    def __init__(self, window_name: str = "User View") -> None:
        self.window_name = window_name

    def render(
        self,
        image_bgr: np.ndarray,
        overlay_lines: list[str] | None = None,
        overlay_draw_fn: OverlayDrawFn | None = None,
    ) -> None:
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

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
