from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPTH_ANYTHING_SRC = REPO_ROOT / "dependency" / "Depth-Anything-3" / "src"
if DEPTH_ANYTHING_SRC.exists() and str(DEPTH_ANYTHING_SRC) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_SRC))


@dataclass(frozen=True)
class DepthEstimationResult:
    depth: np.ndarray
    confidence: np.ndarray | None
    inference_ms: float
    device: str
    model_id: str


class DepthEstimationService:
    def __init__(
        self,
        *,
        enabled: bool,
        model_id: str,
        device: str,
        process_res: int,
    ) -> None:
        self._enabled = enabled
        self._model_id = model_id
        self._requested_device = device
        self._process_res = process_res

        self._model = None
        self._runtime_device = "disabled"
        self._last_error: str | None = None
        self._logged_init = False
        self._preflight_checked = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def runtime_device(self) -> str:
        return self._runtime_device

    def estimate(self, frame_bgr: np.ndarray) -> DepthEstimationResult | None:
        if not self._enabled:
            return None

        model = self._ensure_model()
        if model is None:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        prediction = model.inference(
            [rgb],
            process_res=self._process_res,
            export_format="mini_npz",
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        confidence = None
        if prediction.conf is not None:
            confidence = np.asarray(prediction.conf[0], dtype=np.float32)
        return DepthEstimationResult(
            depth=depth,
            confidence=confidence,
            inference_ms=inference_ms,
            device=self._runtime_device,
            model_id=self._model_id,
        )

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        finite_mask = np.isfinite(depth)
        if not finite_mask.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        valid = depth[finite_mask]
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        if hi <= lo:
            hi = lo + 1e-6

        clipped = np.clip(depth, lo, hi)
        normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self._enabled:
            return None
        if not self.preflight():
            return None

        from depth_anything_3.api import DepthAnything3
        runtime_device = self._runtime_device
        try:
            model = DepthAnything3.from_pretrained(self._model_id)
            model = model.to(device=runtime_device)
            model.eval()
        except Exception as exc:
            self._last_error = f"model init failed: {exc}"
            self._enabled = False
            print(f"[depth] disabled: {self._last_error}")
            return None

        self._model = model
        self._runtime_device = runtime_device
        if not self._logged_init:
            print(
                f"[depth] model ready model_id={self._model_id} "
                f"device={self._runtime_device} process_res={self._process_res}"
            )
            self._logged_init = True
        return self._model

    def preflight(self) -> bool:
        if not self._enabled:
            return False
        if self._preflight_checked and self._last_error is None:
            return True
        if self._preflight_checked and self._last_error is not None:
            return False

        runtime_device = self._resolve_device()
        if self._requested_device == "cuda" and runtime_device != "cuda":
            self._last_error = "requested cuda but torch.cuda.is_available() is false"
            self._enabled = False
            self._preflight_checked = True
            print(f"[depth] disabled: {self._last_error}")
            return False

        try:
            from depth_anything_3.api import DepthAnything3  # noqa: F401
        except Exception as exc:
            self._last_error = f"import failed: {exc}"
            self._enabled = False
            self._preflight_checked = True
            print(f"[depth] disabled: {self._last_error}")
            return False

        self._runtime_device = runtime_device
        self._last_error = None
        self._preflight_checked = True
        return True

    def _resolve_device(self) -> str:
        if self._requested_device and self._requested_device != "auto":
            return self._requested_device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
