from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from proto import perception_pb2
from server.runtime.frame_store_service import FrameStoreService
from server.runtime.packet_models import PacketEnvelope
from server.runtime.processing_models import CameraPose, MemoryIndexRecord, ProcessedPacket, RectifiedFrame
from server.services.memory_index_service import SearchHit
from server.services.perception_storage_service import StoredFrameRecord
from server.services.video_rag_service import VideoRagService


class _FakeGroqService:
    enabled = True

    def generate_search_query(self, *, question: str, latest_metadata: str) -> str:
        assert question
        assert latest_metadata
        return "chair corridor"

    def answer_question(self, *, question: str, retrieval_query: str, context_blocks: list[str], images_bgr):
        return f"answer::{question}::{retrieval_query}::ctx={len(context_blocks)}::img={len(images_bgr)}"


class _FakeMemoryIndexService:
    def __init__(self, hit: SearchHit) -> None:
        self.enabled = True
        self._hit = hit

    def query(self, *, query_text: str, top_k: int):
        assert query_text
        assert top_k >= 1
        return [self._hit]


class _FakeStorageService:
    def __init__(self, latest_record: StoredFrameRecord) -> None:
        self._latest_record = latest_record

    def get_latest_frame_record(self) -> StoredFrameRecord:
        return self._latest_record


class VideoRagServiceTest(unittest.TestCase):
    def test_query_current_scene_uses_latest_frame_and_memory_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            latest_path = tmp_path / "latest.jpg"
            memory_path = tmp_path / "memory.jpg"
            cv2.imwrite(str(latest_path), np.full((32, 32, 3), 64, dtype=np.uint8))
            cv2.imwrite(str(memory_path), np.full((32, 32, 3), 128, dtype=np.uint8))

            latest_record = StoredFrameRecord(
                timestamp_ns=10,
                received_unix_s=0.0,
                client_peer="local",
                tracking_state="TRACKING",
                pose_valid=True,
                camera_position_xyz=(0.0, 0.0, 0.0),
                yolo_detection_count=1,
                yolo_class_names=("chair",),
                map_revision=2,
                source_frame_path=str(latest_path),
                rectified_frame_path=str(latest_path),
                depth_preview_path=None,
                created_unix_s=0.0,
            )
            hit = SearchHit(
                rank=1,
                score=0.9,
                visual_score=0.8,
                metadata_score=1.0,
                record=MemoryIndexRecord(
                    timestamp_ns=9,
                    index_position=0,
                    image_path=str(memory_path),
                    metadata_text="objects: chair",
                    embedding_model_name="fake",
                    embedding_pretrained="fake",
                    embedding_device="cpu",
                    index_backend="faiss",
                    yolo_class_names=("chair",),
                ),
            )
            frame_store = FrameStoreService(max_packets=4)
            frame_store.put(_make_processed_packet(timestamp_ns=10))

            service = VideoRagService(
                storage_service=_FakeStorageService(latest_record),
                memory_index_service=_FakeMemoryIndexService(hit),
                groq_service=_FakeGroqService(),
                frame_store=frame_store,
                default_top_k=3,
                max_images_for_answer=3,
            )

            result = service.query_current_scene(question="What am I looking at?")

            self.assertEqual(result.retrieval_query, "chair corridor")
            self.assertEqual(result.latest_timestamp_ns, 10)
            self.assertEqual(len(result.hits), 1)
            self.assertIn("answer::What am I looking at?::chair corridor", result.answer)


def _make_processed_packet(*, timestamp_ns: int) -> ProcessedPacket:
    packet = perception_pb2.ServerPerceptionPacket(timestamp_ns=timestamp_ns, image_jpeg=b"")
    packet.yolo_output.detections.add(
        class_id=0,
        class_name="chair",
        confidence=0.9,
        xyxy=[0.0, 0.0, 10.0, 10.0],
    )
    image = np.full((32, 32, 3), 32, dtype=np.uint8)
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
