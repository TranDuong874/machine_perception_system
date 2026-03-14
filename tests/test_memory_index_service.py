from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from proto import perception_pb2
from server.runtime.packet_models import PacketEnvelope
from server.runtime.processing_models import CameraPose, ProcessedPacket, RectifiedFrame
from server.services.memory_index_service import MemoryIndexService


class _FakeEncoder:
    def __init__(self) -> None:
        self.device = "cpu"
        self.dimension = 3

    def encode_image_arrays(self, images_bgr):
        images_bgr = list(images_bgr)
        vectors = []
        for image in images_bgr:
            vectors.append(
                np.array(
                    [
                        float(image.mean()),
                        float(image[:, :, 0].mean()),
                        1.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.asarray(vectors, dtype=np.float32)

    def encode_texts(self, texts):
        texts = list(texts)
        vectors = []
        for text in texts:
            normalized = text.lower()
            vectors.append(
                np.array(
                    [
                        10.0 if "chair" in normalized else 1.0,
                        8.0 if "table" in normalized else 1.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.asarray(vectors, dtype=np.float32)


class MemoryIndexServiceTest(unittest.TestCase):
    def test_add_query_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_dir = Path(tmp_dir) / "memory_index"
            service = MemoryIndexService(
                enabled=True,
                index_dir=index_dir,
                model_name="fake-model",
                pretrained="fake-pretrained",
                device="cpu",
                batch_size=1,
                save_every_n_updates=1,
                visual_weight=0.7,
                metadata_weight=0.3,
                encoder=_FakeEncoder(),
            )

            chair_packet = _make_processed_packet(timestamp_ns=1, class_name="chair")
            table_packet = _make_processed_packet(timestamp_ns=2, class_name="table")
            chair_record = service.add_packet(chair_packet)
            table_record = service.add_packet(table_packet)

            self.assertIsNotNone(chair_record)
            self.assertIsNotNone(table_record)
            assert chair_record is not None
            assert table_record is not None
            self.assertEqual(service.entry_count, 2)
            self.assertTrue(Path(chair_record.image_path).exists())

            hits = service.query(query_text="chair", top_k=2)
            self.assertEqual(len(hits), 2)
            self.assertEqual(hits[0].record.timestamp_ns, 1)

            service.close()

            reloaded = MemoryIndexService(
                enabled=True,
                index_dir=index_dir,
                model_name="fake-model",
                pretrained="fake-pretrained",
                device="cpu",
                batch_size=1,
                save_every_n_updates=1,
                visual_weight=0.7,
                metadata_weight=0.3,
                encoder=_FakeEncoder(),
            )
            self.assertTrue(reloaded.preflight())
            self.assertEqual(reloaded.entry_count, 2)
            reloaded_hits = reloaded.query(query_text="table", top_k=1)
            self.assertEqual(len(reloaded_hits), 1)
            self.assertEqual(reloaded_hits[0].record.timestamp_ns, 2)


def _make_processed_packet(*, timestamp_ns: int, class_name: str) -> ProcessedPacket:
    packet = perception_pb2.ServerPerceptionPacket(timestamp_ns=timestamp_ns, image_jpeg=b"")
    packet.yolo_output.detections.add(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        xyxy=[0.0, 0.0, 10.0, 10.0],
    )

    image = np.full((32, 32, 3), 32 if class_name == "chair" else 128, dtype=np.uint8)
    return ProcessedPacket(
        envelope=PacketEnvelope(packet=packet, received_unix_s=0.0, client_peer="local"),
        source_frame_bgr=image.copy(),
        rectified_frame=RectifiedFrame(
            image_bgr=image,
            intrinsics=np.eye(3, dtype=np.float32),
            valid_mask=np.ones((32, 32), dtype=bool),
            rectified=False,
            calibration_id="identity",
        ),
        segmentation_mask=None,
        pose=CameraPose(
            tracking_state="TRACKING",
            world_to_camera=np.eye(4, dtype=np.float32),
            camera_to_world=np.eye(4, dtype=np.float32),
            camera_position_xyz=(0.0, 0.0, 0.0),
            raw_translation_xyz=(0.0, 0.0, 0.0),
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
