from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

@dataclass(frozen=True)
class IMUSample:
    timestamp_ns: int
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]

@dataclass(frozen=True)
class FrameSample:
    timestamp_ns: int
    image_bgr: np.ndarray

@dataclass(frozen=True)
class SynchronizedSensorPacket:
    timestamp_ns: int
    image_bgr: np.ndarray
    imu_samples: tuple[IMUSample, ...]


@dataclass(frozen=True)
class OrbResult:
    tracking_state: str | None = None
    camera_translation_xyz: tuple[float, float, float] | None = None
    camera_quaternion_wxyz: tuple[float, float, float, float] | None = None
    body_translation_xyz: tuple[float, float, float] | None = None
    body_quaternion_wxyz: tuple[float, float, float, float] | None = None
    tracking_image_bgr: np.ndarray | None = None
    error: str | None = None


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloResult:
    detections: tuple[YoloDetection, ...] = ()
    annotated_image_bgr: np.ndarray | None = None
    error: str | None = None


@dataclass(frozen=True)
class EnrichedPerceptionPacket:
    timestamp_ns: int
    image_bgr: np.ndarray
    imu_samples: tuple[IMUSample, ...]
    orb_tracking: OrbResult | None = None
    yolo_output: YoloResult | None = None
    segmentation_mask: np.ndarray | None = None

InputSample: TypeAlias = FrameSample | IMUSample
