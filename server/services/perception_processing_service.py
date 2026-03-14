from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from server.runtime.packet_models import PacketEnvelope
from server.runtime.processing_models import CameraPose, ProcessedPacket
from server.services.camera_calibration_service import CameraCalibrationService
from server.services.depth_estimation_service import DepthEstimationResult, DepthEstimationService


@dataclass(frozen=True)
class ReprojectionConfig:
    point_stride: int
    max_points_per_frame: int
    min_depth_m: float
    max_depth_m: float
    confidence_percentile: float
    min_confidence: float
    min_height_offset_m: float
    max_height_offset_m: float


class PerceptionProcessingService:
    def __init__(
        self,
        *,
        camera_calibration: CameraCalibrationService,
        depth_estimator: DepthEstimationService | None,
        reprojection_config: ReprojectionConfig,
    ) -> None:
        self._camera_calibration = camera_calibration
        self._depth_estimator = depth_estimator
        self._reprojection_config = reprojection_config

    def process(self, envelope: PacketEnvelope) -> ProcessedPacket:
        return self.enrich_with_depth(self.prepare(envelope))

    def prepare(self, envelope: PacketEnvelope) -> ProcessedPacket:
        packet = envelope.packet
        source_frame = _decode_image(packet.image_jpeg, cv2.IMREAD_COLOR)
        if source_frame is None:
            raise ValueError("failed to decode packet.image_jpeg")

        segmentation_mask = None
        if packet.segmentation_mask_png:
            segmentation_mask = _decode_image(packet.segmentation_mask_png, cv2.IMREAD_UNCHANGED)

        rectified_frame = self._camera_calibration.rectify(source_frame)
        pose = _parse_pose(packet)

        yolo_output = packet.yolo_output if packet.HasField("yolo_output") else None
        yolo_detection_count = len(yolo_output.detections) if yolo_output is not None else 0
        return ProcessedPacket(
            envelope=envelope,
            source_frame_bgr=source_frame,
            rectified_frame=rectified_frame,
            segmentation_mask=segmentation_mask,
            pose=pose,
            depth_result=None,
            projected_points_world=np.empty((0, 3), dtype=np.float32),
            yolo_detection_count=yolo_detection_count,
            reprojection_confidence_threshold=None,
        )

    def enrich_with_depth(self, prepared_packet: ProcessedPacket) -> ProcessedPacket:
        depth_result = None
        if self._depth_estimator is not None and self._depth_estimator.enabled:
            depth_result = self._estimate_depth(prepared_packet.rectified_frame.image_bgr)

        projected_points_world, confidence_threshold = self._reproject_depth(
            rectified_frame=prepared_packet.rectified_frame,
            pose=prepared_packet.pose,
            depth_result=depth_result,
        )
        return ProcessedPacket(
            envelope=prepared_packet.envelope,
            source_frame_bgr=prepared_packet.source_frame_bgr,
            rectified_frame=prepared_packet.rectified_frame,
            segmentation_mask=prepared_packet.segmentation_mask,
            pose=prepared_packet.pose,
            depth_result=depth_result,
            projected_points_world=projected_points_world,
            yolo_detection_count=prepared_packet.yolo_detection_count,
            reprojection_confidence_threshold=confidence_threshold,
            memory_record=prepared_packet.memory_record,
            map_snapshot=prepared_packet.map_snapshot,
        )

    def _estimate_depth(self, rectified_frame_bgr: np.ndarray) -> DepthEstimationResult | None:
        try:
            return self._depth_estimator.estimate(rectified_frame_bgr)
        except Exception as exc:
            print(f"[processing] depth estimation failed: {exc}")
            return None

    def _reproject_depth(
        self,
        *,
        rectified_frame,
        pose: CameraPose,
        depth_result: DepthEstimationResult | None,
    ) -> tuple[np.ndarray, float | None]:
        if depth_result is None or not pose.pose_valid:
            return np.empty((0, 3), dtype=np.float32), None

        depth = depth_result.depth
        confidence = depth_result.confidence
        image_height, image_width = rectified_frame.image_bgr.shape[:2]
        if depth.shape[:2] != (image_height, image_width):
            depth = cv2.resize(depth, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            if confidence is not None:
                confidence = cv2.resize(
                    confidence,
                    (image_width, image_height),
                    interpolation=cv2.INTER_LINEAR,
                )

        rows = np.arange(0, image_height, self._reprojection_config.point_stride, dtype=np.int32)
        cols = np.arange(0, image_width, self._reprojection_config.point_stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(cols, rows)

        sampled_depth = depth[grid_y, grid_x]
        valid = (
            rectified_frame.valid_mask[grid_y, grid_x]
            & np.isfinite(sampled_depth)
            & (sampled_depth >= self._reprojection_config.min_depth_m)
            & (sampled_depth <= self._reprojection_config.max_depth_m)
        )

        confidence_threshold = None
        if confidence is not None:
            sampled_confidence = confidence[grid_y, grid_x]
            confidence_threshold = _compute_confidence_threshold(
                sampled_confidence,
                valid,
                min_confidence=self._reprojection_config.min_confidence,
                percentile=self._reprojection_config.confidence_percentile,
            )
            valid &= sampled_confidence >= confidence_threshold

        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), confidence_threshold

        u = grid_x[valid].astype(np.float32)
        v = grid_y[valid].astype(np.float32)
        z = sampled_depth[valid].astype(np.float32)
        intrinsics = rectified_frame.intrinsics.astype(np.float32)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        camera_points = np.stack([x, y, z], axis=1)

        camera_to_world = pose.camera_to_world
        assert camera_to_world is not None
        rotation = camera_to_world[:3, :3].astype(np.float32)
        translation = camera_to_world[:3, 3].astype(np.float32)
        world_points = (camera_points @ rotation.T) + translation

        camera_position = np.asarray(pose.camera_position_xyz, dtype=np.float32)
        height_offsets = world_points[:, 1] - camera_position[1]
        height_valid = (
            height_offsets >= self._reprojection_config.min_height_offset_m
        ) & (
            height_offsets <= self._reprojection_config.max_height_offset_m
        )
        world_points = world_points[height_valid]

        if world_points.shape[0] > self._reprojection_config.max_points_per_frame:
            stride = int(np.ceil(world_points.shape[0] / self._reprojection_config.max_points_per_frame))
            world_points = world_points[::stride]

        return world_points.astype(np.float32, copy=False), confidence_threshold


def _decode_image(encoded: bytes, flags: int) -> np.ndarray | None:
    if not encoded:
        return None
    raw = np.frombuffer(encoded, dtype=np.uint8)
    return cv2.imdecode(raw, flags)


def _parse_pose(packet) -> CameraPose:
    if not packet.HasField("orb_tracking"):
        return CameraPose(
            tracking_state="MISSING_ORB",
            world_to_camera=None,
            camera_to_world=None,
            camera_position_xyz=None,
            raw_translation_xyz=None,
            raw_quaternion_wxyz=None,
            error="missing orb_tracking field",
        )

    orb_tracking = packet.orb_tracking
    tracking_state = orb_tracking.tracking_state or "UNKNOWN"
    translation = tuple(float(value) for value in orb_tracking.camera_translation_xyz)
    quaternion = tuple(float(value) for value in orb_tracking.camera_quaternion_wxyz)
    if len(translation) != 3 or len(quaternion) != 4:
        return CameraPose(
            tracking_state=tracking_state,
            world_to_camera=None,
            camera_to_world=None,
            camera_position_xyz=None,
            raw_translation_xyz=translation if len(translation) == 3 else None,
            raw_quaternion_wxyz=quaternion if len(quaternion) == 4 else None,
            error="invalid pose payload lengths",
        )

    try:
        rotation = _rotation_matrix_from_quaternion_wxyz(quaternion)
    except ValueError as exc:
        return CameraPose(
            tracking_state=tracking_state,
            world_to_camera=None,
            camera_to_world=None,
            camera_position_xyz=None,
            raw_translation_xyz=translation,
            raw_quaternion_wxyz=quaternion,
            error=str(exc),
        )

    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = np.asarray(translation, dtype=np.float32)
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = -rotation.T @ np.asarray(translation, dtype=np.float32)
    camera_position_xyz = tuple(float(value) for value in camera_to_world[:3, 3])
    return CameraPose(
        tracking_state=tracking_state,
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        camera_position_xyz=camera_position_xyz,
        raw_translation_xyz=translation,
        raw_quaternion_wxyz=quaternion,
        error=None,
    )


def _rotation_matrix_from_quaternion_wxyz(
    quaternion_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("invalid quaternion norm")

    w /= norm
    x /= norm
    y /= norm
    z /= norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _compute_confidence_threshold(
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_confidence: float,
    percentile: float,
) -> float:
    valid_values = confidence[valid_mask]
    if valid_values.size == 0:
        return float(min_confidence)
    percentile_value = float(np.percentile(valid_values, percentile))
    return max(float(min_confidence), percentile_value)
