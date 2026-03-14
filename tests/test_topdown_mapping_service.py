from __future__ import annotations

import unittest

import numpy as np

from proto import perception_pb2
from server.runtime.packet_models import PacketEnvelope
from server.runtime.processing_models import CameraPose, ProcessedPacket, RectifiedFrame
from server.services.topdown_mapping_service import TopDownMappingConfig, TopDownMappingService


class TopDownMappingServiceTest(unittest.TestCase):
    def test_mapping_service_updates_snapshot(self) -> None:
        service = TopDownMappingService(
            config=TopDownMappingConfig(
                enabled=True,
                map_id="test-map",
                resolution_m=0.1,
                width=80,
                height=80,
                free_space_decrement=0.1,
                occupied_increment=0.7,
            )
        )
        processed = _make_processed_packet(
            timestamp_ns=1,
            camera_position=(0.0, 0.0, 0.0),
            projected_points=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.3, 0.0, 1.5],
                ],
                dtype=np.float32,
            ),
        )

        snapshot = service.update(processed)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.map_id, "test-map")
        self.assertEqual(snapshot.revision, 1)
        self.assertTrue(snapshot.pose_valid)
        self.assertEqual(snapshot.latest_user_translation_xyz, (0.0, 0.0, 0.0))
        self.assertGreater(len(snapshot.occupancy_png), 0)
        self.assertLess(int(snapshot.occupancy_bgr.min()), 100)

        processed_next = _make_processed_packet(
            timestamp_ns=2,
            camera_position=(0.1, 0.0, 0.2),
            projected_points=np.array([[0.2, 0.0, 1.2]], dtype=np.float32),
        )
        next_snapshot = service.update(processed_next)
        self.assertIsNotNone(next_snapshot)
        assert next_snapshot is not None
        self.assertEqual(next_snapshot.revision, 2)
        self.assertEqual(next_snapshot.latest_user_translation_xyz, (0.1, 0.0, 0.2))


def _make_processed_packet(
    *,
    timestamp_ns: int,
    camera_position: tuple[float, float, float],
    projected_points: np.ndarray,
) -> ProcessedPacket:
    packet = perception_pb2.ServerPerceptionPacket(timestamp_ns=timestamp_ns, image_jpeg=b"")
    world_to_camera = np.eye(4, dtype=np.float32)
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, 3] = np.asarray(camera_position, dtype=np.float32)
    return ProcessedPacket(
        envelope=PacketEnvelope(packet=packet, received_unix_s=0.0, client_peer="local"),
        source_frame_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
        rectified_frame=RectifiedFrame(
            image_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
            intrinsics=np.eye(3, dtype=np.float32),
            valid_mask=np.ones((16, 16), dtype=bool),
            rectified=False,
            calibration_id="identity",
        ),
        segmentation_mask=None,
        pose=CameraPose(
            tracking_state="TRACKING",
            world_to_camera=world_to_camera,
            camera_to_world=camera_to_world,
            camera_position_xyz=camera_position,
            raw_translation_xyz=(0.0, 0.0, 0.0),
            raw_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            error=None,
        ),
        depth_result=None,
        projected_points_world=projected_points,
        yolo_detection_count=0,
        reprojection_confidence_threshold=None,
        map_snapshot=None,
    )


if __name__ == "__main__":
    unittest.main()
