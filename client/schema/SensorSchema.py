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
class LocalSensorPacket:
    timestamp_ns: int
    image_bgr: np.ndarray
    imu_samples: tuple[IMUSample, ...]


InputSample: TypeAlias = FrameSample | IMUSample

# TODO: Add ServerSensorPacket when the server handoff contract is defined.
