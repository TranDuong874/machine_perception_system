from dataclasses import dataclass


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    translation_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    tracking_state: str
    
