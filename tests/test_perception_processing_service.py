from __future__ import annotations

import unittest

import cv2
import numpy as np

from proto import perception_pb2
from server.runtime.packet_models import PacketEnvelope
from server.services.camera_calibration_service import CameraCalibrationService
from server.services.depth_estimation_service import DepthEstimationResult
from server.services.perception_processing_service import (
    PerceptionProcessingService,
    ReprojectionConfig,
)


class _FakeDepthEstimator:
    def __init__(self, depth: np.ndarray, confidence: np.ndarray) -> None:
        self.enabled = True
        self._result = DepthEstimationResult(
            depth=depth.astype(np.float32),
            confidence=confidence.astype(np.float32),
            inference_ms=12.5,
            device="cpu",
            model_id="fake-depth",
        )

    def estimate(self, _frame_bgr: np.ndarray) -> DepthEstimationResult:
        return self._result


class PerceptionProcessingServiceTest(unittest.TestCase):
    def test_reprojection_filters_depth_by_confidence(self) -> None:
        image = np.full((3, 3, 3), 120, dtype=np.uint8)
        depth = np.ones((3, 3), dtype=np.float32)
        confidence = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        )
        packet = perception_pb2.ServerPerceptionPacket(
            timestamp_ns=123,
            image_jpeg=_encode_jpeg(image),
        )
        packet.orb_tracking.tracking_state = "TRACKING"
        packet.orb_tracking.camera_translation_xyz[:] = [0.0, 0.0, 0.0]
        packet.orb_tracking.camera_quaternion_wxyz[:] = [1.0, 0.0, 0.0, 0.0]

        processing_service = PerceptionProcessingService(
            camera_calibration=CameraCalibrationService(
                enabled=False,
                calibration_file=None,
                camera_key="cam0",
                rectification_balance=0.0,
            ),
            depth_estimator=_FakeDepthEstimator(depth=depth, confidence=confidence),
            reprojection_config=ReprojectionConfig(
                point_stride=1,
                max_points_per_frame=100,
                min_depth_m=0.1,
                max_depth_m=5.0,
                confidence_percentile=70.0,
                min_confidence=0.0,
                min_height_offset_m=-10.0,
                max_height_offset_m=10.0,
            ),
        )

        processed = processing_service.process(
            PacketEnvelope(packet=packet, received_unix_s=0.0, client_peer="local")
        )

        self.assertIsNotNone(processed.depth_result)
        self.assertAlmostEqual(processed.reprojection_confidence_threshold or 0.0, 0.66, places=2)
        self.assertEqual(processed.projected_points_world.shape[0], 3)
        self.assertTrue(np.allclose(processed.projected_points_world[:, 2], 1.0))


def _encode_jpeg(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("failed to encode jpeg for test")
    return encoded.tobytes()


if __name__ == "__main__":
    unittest.main()
