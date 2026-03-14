from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from server.runtime.processing_models import RectifiedFrame


@dataclass(frozen=True)
class _CalibrationSpec:
    calibration_id: str
    intrinsics: np.ndarray
    distortion_coeffs: np.ndarray
    distortion_model: str
    resolution: tuple[int, int]


@dataclass
class _RectificationCache:
    map1: np.ndarray
    map2: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray


class CameraCalibrationService:
    def __init__(
        self,
        *,
        enabled: bool,
        calibration_file: Path | None,
        camera_key: str,
        rectification_balance: float,
    ) -> None:
        self._enabled = enabled
        self._calibration_file = calibration_file
        self._camera_key = camera_key
        self._rectification_balance = float(rectification_balance)
        self._spec = self._load_spec(calibration_file, camera_key)
        self._cache_by_size: dict[tuple[int, int], _RectificationCache] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled and self._spec is not None

    @property
    def calibration_id(self) -> str:
        if self._spec is None:
            return "unconfigured"
        return self._spec.calibration_id

    def rectify(self, image_bgr: np.ndarray) -> RectifiedFrame:
        height, width = image_bgr.shape[:2]
        if not self.enabled:
            intrinsics = _fallback_intrinsics(width=width, height=height)
            valid_mask = np.ones((height, width), dtype=bool)
            return RectifiedFrame(
                image_bgr=image_bgr,
                intrinsics=intrinsics,
                valid_mask=valid_mask,
                rectified=False,
                calibration_id="identity",
            )

        cache = self._cache_by_size.get((width, height))
        if cache is None:
            cache = self._build_cache(width=width, height=height)
            self._cache_by_size[(width, height)] = cache

        rectified = cv2.remap(
            image_bgr,
            cache.map1,
            cache.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return RectifiedFrame(
            image_bgr=rectified,
            intrinsics=cache.intrinsics.copy(),
            valid_mask=cache.valid_mask.copy(),
            rectified=True,
            calibration_id=self._spec.calibration_id,
        )

    def _build_cache(self, *, width: int, height: int) -> _RectificationCache:
        assert self._spec is not None
        k = self._spec.intrinsics
        distortion = self._spec.distortion_coeffs.reshape(-1, 1)
        identity = np.eye(3, dtype=np.float64)

        if self._spec.distortion_model == "equidistant":
            intrinsics = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k,
                distortion,
                (width, height),
                identity,
                balance=self._rectification_balance,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k,
                distortion,
                identity,
                intrinsics,
                (width, height),
                cv2.CV_32FC1,
            )
        else:
            intrinsics = k.copy()
            map1, map2 = cv2.initUndistortRectifyMap(
                k,
                distortion.ravel(),
                identity,
                intrinsics,
                (width, height),
                cv2.CV_32FC1,
            )

        valid_source = np.full((height, width), 255, dtype=np.uint8)
        valid_mask = cv2.remap(
            valid_source,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return _RectificationCache(
            map1=map1,
            map2=map2,
            valid_mask=valid_mask > 0,
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    def _load_spec(self, calibration_file: Path | None, camera_key: str) -> _CalibrationSpec | None:
        if not self._enabled or calibration_file is None or not calibration_file.exists():
            if self._enabled:
                print(f"[camera] calibration unavailable: {calibration_file}")
            return None

        with calibration_file.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid calibration file: {calibration_file}")

        camera_payload = payload.get(camera_key)
        if not isinstance(camera_payload, dict):
            raise KeyError(f"camera key '{camera_key}' not found in {calibration_file}")

        intrinsics_raw = camera_payload.get("intrinsics")
        distortion_raw = camera_payload.get("distortion_coeffs")
        resolution_raw = camera_payload.get("resolution")
        if intrinsics_raw is None or len(intrinsics_raw) != 4:
            raise ValueError(f"camera '{camera_key}' is missing 4-value intrinsics")
        if distortion_raw is None or len(distortion_raw) not in {4, 5, 8}:
            raise ValueError(f"camera '{camera_key}' has unsupported distortion coefficients")
        if resolution_raw is None or len(resolution_raw) != 2:
            raise ValueError(f"camera '{camera_key}' is missing 2-value resolution")

        fx, fy, cx, cy = (float(value) for value in intrinsics_raw)
        intrinsics = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(distortion_raw, dtype=np.float64)
        resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
        distortion_model = str(camera_payload.get("distortion_model", "")).strip().lower()
        calibration_id = f"{calibration_file}:{camera_key}"
        print(
            f"[camera] loaded calibration={calibration_id} "
            f"distortion_model={distortion_model or 'none'} resolution={resolution}"
        )
        return _CalibrationSpec(
            calibration_id=calibration_id,
            intrinsics=intrinsics,
            distortion_coeffs=distortion,
            distortion_model=distortion_model,
            resolution=resolution,
        )


def _fallback_intrinsics(*, width: int, height: int) -> np.ndarray:
    focal = float(max(width, height))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

