from __future__ import annotations

import unittest

import numpy as np

from proto import perception_pb2
from server.runtime.packet_models import PacketEnvelope
from server.runtime.processing_models import CameraPose, ProcessedPacket, RectifiedFrame
from server.services.frame_selection_service import FrameSelectionService, TaskSelectionConfig


class FrameSelectionServiceTest(unittest.TestCase):
    def test_depth_selection_uses_motion_threshold_and_memory_detects_label_change(self) -> None:
        service = FrameSelectionService(
            config=TaskSelectionConfig(
                depth_every_n_frames=10,
                depth_translation_m=0.15,
                depth_rotation_deg=8.0,
                memory_every_n_frames=20,
                memory_translation_m=1.0,
                memory_rotation_deg=20.0,
                memory_schedule_on_detection_change=True,
            )
        )

        first = _make_processed_packet(timestamp_ns=1, x=0.0, class_name="chair")
        second = _make_processed_packet(timestamp_ns=2, x=0.05, class_name="chair")
        third = _make_processed_packet(timestamp_ns=3, x=0.20, class_name="table")

        first_decision = service.decide(first)
        second_decision = service.decide(second)
        third_decision = service.decide(third)

        self.assertTrue(first_decision.schedule_depth)
        self.assertTrue(first_decision.schedule_memory)
        self.assertFalse(second_decision.schedule_depth)
        self.assertFalse(second_decision.schedule_memory)
        self.assertTrue(third_decision.schedule_depth)
        self.assertTrue(third_decision.schedule_memory)


def _make_processed_packet(*, timestamp_ns: int, x: float, class_name: str) -> ProcessedPacket:
    packet = perception_pb2.ServerPerceptionPacket(timestamp_ns=timestamp_ns, image_jpeg=b"")
    packet.yolo_output.detections.add(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        xyxy=[0.0, 0.0, 10.0, 10.0],
    )

    image = np.full((16, 16, 3), 100, dtype=np.uint8)
    return ProcessedPacket(
        envelope=PacketEnvelope(packet=packet, received_unix_s=0.0, client_peer="local"),
        source_frame_bgr=image.copy(),
        rectified_frame=RectifiedFrame(
            image_bgr=image,
            intrinsics=np.eye(3, dtype=np.float32),
            valid_mask=np.ones((16, 16), dtype=bool),
            rectified=False,
            calibration_id="identity",
        ),
        segmentation_mask=None,
        pose=CameraPose(
            tracking_state="TRACKING",
            world_to_camera=np.eye(4, dtype=np.float32),
            camera_to_world=np.eye(4, dtype=np.float32),
            camera_position_xyz=(x, 0.0, 0.0),
            raw_translation_xyz=(x, 0.0, 0.0),
            raw_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            error=None,
        ),
        depth_result=None,
        projected_points_world=np.empty((0, 3), dtype=np.float32),
        yolo_detection_count=1,
        reprojection_confidence_threshold=None,
    )


if __name__ == "__main__":
    unittest.main()
