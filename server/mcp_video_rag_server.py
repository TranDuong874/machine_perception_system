from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from server.services.groq_vlm_service import GroqVLMService
from server.services.memory_index_service import MemoryIndexService
from server.services.perception_storage_service import PerceptionStorageService
from server.services.video_rag_service import VideoRagService
from server.settings.server_settings import load_server_config


def main() -> None:
    config = load_server_config()
    storage_service = PerceptionStorageService(
        db_path=config.storage_db_path,
        artifact_dir=config.storage_artifact_dir,
        save_debug_artifacts=config.storage_save_debug_artifacts,
        save_map_every_n_updates=config.storage_save_map_every_n_updates,
    )
    memory_index_service = MemoryIndexService(
        enabled=config.enable_memory_index,
        index_dir=config.memory_index_dir,
        model_name=config.memory_model_name,
        pretrained=config.memory_pretrained,
        device=config.memory_device,
        batch_size=config.memory_batch_size,
        save_every_n_updates=config.memory_save_every_n_updates,
        visual_weight=config.memory_visual_weight,
        metadata_weight=config.memory_metadata_weight,
    )
    memory_index_service.preflight()
    groq_service = GroqVLMService(
        api_key=config.groq_api_key,
        base_url=config.groq_base_url,
        text_model=config.groq_text_model,
        vision_model=config.groq_vision_model,
        timeout_s=config.groq_timeout_s,
    )
    service = VideoRagService(
        storage_service=storage_service,
        memory_index_service=memory_index_service,
        groq_service=groq_service,
        frame_store=None,
        default_top_k=config.video_rag_top_k,
        max_images_for_answer=config.video_rag_max_images,
    )
    mcp = FastMCP("machine-perception-video-rag")

    @mcp.tool()
    def describe_current_scene(question: str, top_k: int = 3) -> dict:
        result = service.query_current_scene(question=question, top_k=top_k)
        return {
            "question": result.question,
            "retrieval_query": result.retrieval_query,
            "answer": result.answer,
            "latest_timestamp_ns": result.latest_timestamp_ns,
            "latest_frame_path": result.latest_frame_path,
            "hits": [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "timestamp_ns": hit.record.timestamp_ns,
                    "image_path": hit.record.image_path,
                    "metadata_text": hit.record.metadata_text,
                }
                for hit in result.hits
            ],
        }

    @mcp.tool()
    def video_rag(question: str, top_k: int = 5) -> dict:
        result = service.query_video_rag(question=question, top_k=top_k)
        return {
            "question": result.question,
            "retrieval_query": result.retrieval_query,
            "answer": result.answer,
            "latest_timestamp_ns": result.latest_timestamp_ns,
            "latest_frame_path": result.latest_frame_path,
            "hits": [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "timestamp_ns": hit.record.timestamp_ns,
                    "image_path": hit.record.image_path,
                    "metadata_text": hit.record.metadata_text,
                }
                for hit in result.hits
            ],
        }

    @mcp.tool()
    def search_memory(query: str, top_k: int = 5) -> dict:
        hits = memory_index_service.query(query_text=query, top_k=top_k)
        return {
            "query": query,
            "hits": [
                {
                    "rank": hit.rank,
                    "score": hit.score,
                    "visual_score": hit.visual_score,
                    "metadata_score": hit.metadata_score,
                    "timestamp_ns": hit.record.timestamp_ns,
                    "image_path": hit.record.image_path,
                    "metadata_text": hit.record.metadata_text,
                }
                for hit in hits
            ],
        }

    @mcp.tool()
    def latest_frame() -> dict:
        latest = storage_service.get_latest_frame_record()
        if latest is None:
            return {"latest_timestamp_ns": None, "latest_frame_path": None, "tracking_state": None}
        latest_frame_path = latest.rectified_frame_path or latest.source_frame_path
        return {
            "latest_timestamp_ns": latest.timestamp_ns,
            "latest_frame_path": latest_frame_path,
            "tracking_state": latest.tracking_state,
            "pose_valid": latest.pose_valid,
            "yolo_detection_count": latest.yolo_detection_count,
            "yolo_class_names": list(latest.yolo_class_names),
            "map_revision": latest.map_revision,
            "camera_position_xyz": list(latest.camera_position_xyz) if latest.camera_position_xyz else None,
        }

    try:
        mcp.run(transport="stdio")
    finally:
        memory_index_service.close()
        storage_service.close()


if __name__ == "__main__":
    main()
