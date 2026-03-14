from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.runtime.frame_store_service import FrameStoreService
from server.runtime.processing_models import ProcessedPacket
from server.services.groq_vlm_service import GroqVLMService, load_image_bgr
from server.services.memory_index_service import MemoryIndexService, SearchHit
from server.services.perception_storage_service import PerceptionStorageService, StoredFrameRecord


@dataclass(frozen=True)
class VideoRagResult:
    question: str
    retrieval_query: str
    answer: str
    latest_timestamp_ns: int | None
    latest_frame_path: str | None
    hits: list[SearchHit]


class VideoRagService:
    def __init__(
        self,
        *,
        storage_service: PerceptionStorageService,
        memory_index_service: MemoryIndexService,
        groq_service: GroqVLMService,
        frame_store: FrameStoreService | None,
        default_top_k: int,
        max_images_for_answer: int,
    ) -> None:
        self._storage_service = storage_service
        self._memory_index_service = memory_index_service
        self._groq_service = groq_service
        self._frame_store = frame_store
        self._default_top_k = max(1, int(default_top_k))
        self._max_images_for_answer = max(1, int(max_images_for_answer))

    def query_current_scene(self, *, question: str, top_k: int | None = None) -> VideoRagResult:
        latest_packet = self._frame_store.latest() if self._frame_store is not None else None
        latest_record = self._storage_service.get_latest_frame_record()
        latest_metadata = _latest_metadata_text(latest_packet, latest_record)
        retrieval_query = self._groq_service.generate_search_query(
            question=question,
            latest_metadata=latest_metadata,
        )
        hits = self._query_memory(retrieval_query=retrieval_query, top_k=top_k)
        answer = self._answer_question(
            question=question,
            retrieval_query=retrieval_query,
            latest_packet=latest_packet,
            latest_record=latest_record,
            hits=hits,
        )
        latest_timestamp_ns = (
            int(latest_packet.packet.timestamp_ns)
            if latest_packet is not None
            else (latest_record.timestamp_ns if latest_record is not None else None)
        )
        latest_frame_path = (
            latest_record.rectified_frame_path
            if latest_record is not None and latest_record.rectified_frame_path
            else (latest_record.source_frame_path if latest_record is not None else None)
        )
        return VideoRagResult(
            question=question,
            retrieval_query=retrieval_query,
            answer=answer,
            latest_timestamp_ns=latest_timestamp_ns,
            latest_frame_path=latest_frame_path,
            hits=hits,
        )

    def query_video_rag(self, *, question: str, top_k: int | None = None) -> VideoRagResult:
        latest_packet = self._frame_store.latest() if self._frame_store is not None else None
        latest_record = self._storage_service.get_latest_frame_record()
        latest_metadata = _latest_metadata_text(latest_packet, latest_record)
        retrieval_query = self._groq_service.generate_search_query(
            question=question,
            latest_metadata=latest_metadata,
        )
        hits = self._query_memory(retrieval_query=retrieval_query, top_k=top_k)
        answer = self._answer_question(
            question=question,
            retrieval_query=retrieval_query,
            latest_packet=latest_packet,
            latest_record=latest_record,
            hits=hits,
        )
        latest_timestamp_ns = latest_record.timestamp_ns if latest_record is not None else None
        latest_frame_path = latest_record.rectified_frame_path if latest_record is not None else None
        return VideoRagResult(
            question=question,
            retrieval_query=retrieval_query,
            answer=answer,
            latest_timestamp_ns=latest_timestamp_ns,
            latest_frame_path=latest_frame_path,
            hits=hits,
        )

    def _query_memory(self, *, retrieval_query: str, top_k: int | None) -> list[SearchHit]:
        if not self._memory_index_service.enabled:
            return []
        return self._memory_index_service.query(
            query_text=retrieval_query,
            top_k=top_k or self._default_top_k,
        )

    def _answer_question(
        self,
        *,
        question: str,
        retrieval_query: str,
        latest_packet: ProcessedPacket | None,
        latest_record: StoredFrameRecord | None,
        hits: list[SearchHit],
    ) -> str:
        context_blocks = []
        if latest_record is not None:
            context_blocks.append(_frame_record_text(latest_record))
        for hit in hits:
            context_blocks.append(
                f"[memory hit #{hit.rank}] score={hit.score:.4f}\n"
                f"timestamp_ns: {hit.record.timestamp_ns}\n"
                f"{hit.record.metadata_text}"
            )
        images = _collect_images(
            latest_packet=latest_packet,
            latest_record=latest_record,
            hits=hits,
            max_images=self._max_images_for_answer,
        )
        return self._groq_service.answer_question(
            question=question,
            retrieval_query=retrieval_query,
            context_blocks=context_blocks,
            images_bgr=images,
        )


def _collect_images(
    *,
    latest_packet: ProcessedPacket | None,
    latest_record: StoredFrameRecord | None,
    hits: list[SearchHit],
    max_images: int,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    if latest_packet is not None:
        images.append(latest_packet.rectified_frame.image_bgr)
    elif latest_record is not None:
        latest_image_path = latest_record.rectified_frame_path or latest_record.source_frame_path
        if latest_image_path is not None:
            image = load_image_bgr(latest_image_path)
            if image is not None:
                images.append(image)

    for hit in hits:
        if len(images) >= max_images:
            break
        image = load_image_bgr(hit.record.image_path)
        if image is not None:
            images.append(image)
    return images[:max_images]


def _frame_record_text(frame_record: StoredFrameRecord) -> str:
    lines = [
        f"[latest frame] timestamp_ns: {frame_record.timestamp_ns}",
        f"tracking_state: {frame_record.tracking_state}",
        f"pose_valid: {frame_record.pose_valid}",
        f"detections: {frame_record.yolo_detection_count}",
    ]
    if frame_record.yolo_class_names:
        lines.append("objects: " + ", ".join(frame_record.yolo_class_names))
    if frame_record.camera_position_xyz is not None:
        x, y, z = frame_record.camera_position_xyz
        lines.append(f"camera_xyz_m: {x:.3f}, {y:.3f}, {z:.3f}")
    if frame_record.map_revision is not None:
        lines.append(f"map_revision: {frame_record.map_revision}")
    return "\n".join(lines)


def _latest_metadata_text(
    latest_packet: ProcessedPacket | None,
    latest_record: StoredFrameRecord | None,
) -> str:
    if latest_packet is not None:
        lines = [
            f"timestamp_ns: {latest_packet.packet.timestamp_ns}",
            f"tracking_state: {latest_packet.pose.tracking_state}",
            f"pose_valid: {latest_packet.pose.pose_valid}",
            f"detections: {latest_packet.yolo_detection_count}",
        ]
        if latest_packet.pose.camera_position_xyz is not None:
            x, y, z = latest_packet.pose.camera_position_xyz
            lines.append(f"camera_xyz_m: {x:.3f}, {y:.3f}, {z:.3f}")
        if latest_packet.packet.HasField("yolo_output"):
            class_names = sorted(
                {
                    str(detection.class_name).strip()
                    for detection in latest_packet.packet.yolo_output.detections
                    if detection.class_name
                }
            )
            if class_names:
                lines.append("objects: " + ", ".join(class_names))
        return "\n".join(lines)
    if latest_record is not None:
        return _frame_record_text(latest_record)
    return "No live or stored frame is available."
