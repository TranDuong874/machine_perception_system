from dataclasses import dataclass

import numpy as np

from schema.PoseSchema import PoseSample


@dataclass(frozen=True)
class PerceptionPacket:
    timestamp_ns: int
    frame_index: int
    image_bgr: np.ndarray
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]
    pose: PoseSample
