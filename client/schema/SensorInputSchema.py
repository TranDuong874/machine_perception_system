from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class IMUSample:
    timestamp_ns: int
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]

@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    timestamp_ns: int
    image_bgr: np.ndarray

@dataclass(frozen=True)
class SensorPacket:
    timestamp_ns: int
    frame_index: int
    image_bgr: np.ndarray
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]
    
    