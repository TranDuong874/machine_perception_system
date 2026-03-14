from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from server.runtime.packet_models import PacketEnvelope

if TYPE_CHECKING:
    from server.services.depth_estimation_service import DepthEstimationResult


@dataclass(frozen=True)
class CameraPose:
    tracking_state: str
    world_to_camera: np.ndarray | None
    camera_to_world: np.ndarray | None
    camera_position_xyz: tuple[float, float, float] | None
    raw_translation_xyz: tuple[float, float, float] | None
    raw_quaternion_wxyz: tuple[float, float, float, float] | None
    error: str | None = None

    @property
    def pose_valid(self) -> bool:
        return self.world_to_camera is not None and self.camera_to_world is not None


@dataclass(frozen=True)
class RectifiedFrame:
    image_bgr: np.ndarray
    intrinsics: np.ndarray
    valid_mask: np.ndarray
    rectified: bool
    calibration_id: str


@dataclass
class TopDownMapSnapshot:
    map_id: str
    revision: int
    timestamp_ns: int
    tracking_state: str
    pose_valid: bool
    resolution_m: float
    width: int
    height: int
    origin_x_m: float
    origin_z_m: float
    latest_user_translation_xyz: tuple[float, float, float] | None
    occupancy_png: bytes
    occupancy_bgr: np.ndarray
    backend: str


@dataclass(frozen=True)
class MemoryIndexRecord:
    timestamp_ns: int
    index_position: int
    image_path: str
    metadata_text: str
    embedding_model_name: str
    embedding_pretrained: str
    embedding_device: str
    index_backend: str
    yolo_class_names: tuple[str, ...]


@dataclass
class ProcessedPacket:
    envelope: PacketEnvelope
    source_frame_bgr: np.ndarray
    rectified_frame: RectifiedFrame
    segmentation_mask: np.ndarray | None
    pose: CameraPose
    depth_result: DepthEstimationResult | None
    projected_points_world: np.ndarray
    yolo_detection_count: int
    reprojection_confidence_threshold: float | None
    memory_record: MemoryIndexRecord | None = None
    map_snapshot: TopDownMapSnapshot | None = None

    @property
    def packet(self):
        return self.envelope.packet
