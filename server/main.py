from __future__ import annotations

from concurrent import futures
from pathlib import Path
import signal
import sys
import threading

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grpc

from proto import perception_pb2_grpc
from server.api.grpc_ingest_servicer import PacketIngressServicer
from server.api.vlm_http_service import VlmHttpService
from server.runtime.frame_store_service import FrameStoreService
from server.rendering.preview_renderer import ServerPreviewRenderer
from server.runtime.packet_ingestion_service import PacketIngestionService
from server.services.camera_calibration_service import CameraCalibrationService
from server.services.depth_estimation_service import DepthEstimationService
from server.services.frame_selection_service import FrameSelectionService, TaskSelectionConfig
from server.services.groq_vlm_service import GroqVLMService
from server.services.memory_index_service import MemoryIndexService
from server.services.perception_processing_service import (
    PerceptionProcessingService,
    ReprojectionConfig,
)
from server.services.perception_storage_service import PerceptionStorageService
from server.services.topdown_mapping_service import TopDownMappingConfig, TopDownMappingService
from server.services.video_rag_service import VideoRagService
from server.settings.server_settings import load_server_config


class ServerApplication:
    def __init__(
        self,
        *,
        config,
        ingestion_service: PacketIngestionService,
        preview_renderer: ServerPreviewRenderer,
        vlm_http_service: VlmHttpService | None,
    ) -> None:
        self._config = config
        self._ingestion_service = ingestion_service
        self._preview_renderer = preview_renderer
        self._vlm_http_service = vlm_http_service
        self._stop_event = threading.Event()

        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.grpc_workers))
        perception_pb2_grpc.add_PerceptionIngestServiceServicer_to_server(
            PacketIngressServicer(ingestion_service),
            self._grpc_server,
        )

    @classmethod
    def from_env(cls) -> "ServerApplication":
        config = load_server_config()
        depth_estimator = DepthEstimationService(
            enabled=config.enable_depth_estimation,
            model_id=config.depth_model_id,
            device=config.depth_device,
            process_res=config.depth_process_res,
        )
        depth_estimator.preflight()
        camera_calibration = CameraCalibrationService(
            enabled=config.enable_rectification,
            calibration_file=config.camera_calibration_file,
            camera_key=config.camera_calibration_camera_key,
            rectification_balance=config.camera_rectification_balance,
        )
        processing_service = PerceptionProcessingService(
            camera_calibration=camera_calibration,
            depth_estimator=depth_estimator,
            reprojection_config=ReprojectionConfig(
                point_stride=config.map_point_stride,
                max_points_per_frame=config.map_max_points_per_frame,
                min_depth_m=config.map_min_depth_m,
                max_depth_m=config.map_max_depth_m,
                confidence_percentile=config.map_confidence_percentile,
                min_confidence=config.map_min_confidence,
                min_height_offset_m=config.map_min_height_offset_m,
                max_height_offset_m=config.map_max_height_offset_m,
            ),
        )
        mapping_service = TopDownMappingService(
            config=TopDownMappingConfig(
                enabled=config.enable_mapping,
                map_id=config.map_id,
                resolution_m=config.map_resolution_m,
                width=config.map_width,
                height=config.map_height,
                free_space_decrement=config.map_free_space_decrement,
                occupied_increment=config.map_occupied_increment,
            )
        )
        storage_service = PerceptionStorageService(
            db_path=config.storage_db_path,
            artifact_dir=config.storage_artifact_dir,
            save_debug_artifacts=config.storage_save_debug_artifacts,
            save_map_every_n_updates=config.storage_save_map_every_n_updates,
        )
        frame_store = FrameStoreService(max_packets=config.frame_store_max_packets)
        selection_service = FrameSelectionService(
            config=TaskSelectionConfig(
                depth_every_n_frames=config.depth_select_every_n_frames,
                depth_translation_m=config.depth_select_translation_m,
                depth_rotation_deg=config.depth_select_rotation_deg,
                memory_every_n_frames=config.memory_select_every_n_frames,
                memory_translation_m=config.memory_select_translation_m,
                memory_rotation_deg=config.memory_select_rotation_deg,
                memory_schedule_on_detection_change=config.memory_schedule_on_detection_change,
            )
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
        preview_renderer = ServerPreviewRenderer(
            enable_gui=config.enable_preview_gui,
            frame_delay_ms=config.preview_frame_delay_ms,
            headless_preview_path=config.headless_preview_path,
            depth_estimator=depth_estimator,
        )
        ingestion_service = PacketIngestionService(
            log_every_n_packets=config.log_every_n_packets,
            preview_renderer=preview_renderer,
            processing_service=processing_service,
            mapping_service=mapping_service,
            memory_index_service=memory_index_service,
            storage_service=storage_service,
            frame_store=frame_store,
            selection_service=selection_service,
            depth_task_queue_size=config.depth_task_queue_size,
            memory_task_queue_size=config.memory_task_queue_size,
        )
        groq_service = GroqVLMService(
            api_key=config.groq_api_key,
            base_url=config.groq_base_url,
            text_model=config.groq_text_model,
            vision_model=config.groq_vision_model,
            timeout_s=config.groq_timeout_s,
        )
        video_rag_service = VideoRagService(
            storage_service=storage_service,
            memory_index_service=memory_index_service,
            groq_service=groq_service,
            frame_store=frame_store,
            default_top_k=config.video_rag_top_k,
            max_images_for_answer=config.video_rag_max_images,
        )
        vlm_http_service = None
        if config.enable_vlm_api:
            vlm_http_service = VlmHttpService(
                host=config.vlm_bind_host,
                port=config.vlm_bind_port,
                video_rag_service=video_rag_service,
            )
        return cls(
            config=config,
            ingestion_service=ingestion_service,
            preview_renderer=preview_renderer,
            vlm_http_service=vlm_http_service,
        )

    def run(self) -> None:
        self._install_signal_handlers()
        listen_address = f"{self._config.bind_host}:{self._config.bind_port}"
        self._grpc_server.add_insecure_port(listen_address)

        self._ingestion_service.start()
        self._grpc_server.start()
        if self._vlm_http_service is not None:
            self._vlm_http_service.start()

        print(f"gRPC server listening on {listen_address}")
        if self._config.enable_preview_gui:
            print("Preview: GUI windows enabled (ingress + depth)")
        else:
            print(
                "Preview: headless -> "
                f"ingress={self._preview_renderer.headless_ingress_preview_path} "
                f"depth={self._preview_renderer.headless_depth_preview_path}"
            )
        if self._config.enable_depth_estimation:
            depth_estimator = self._preview_renderer.depth_estimator
            if depth_estimator is not None and depth_estimator.enabled:
                print(
                    "Depth: enabled "
                    f"model={self._config.depth_model_id} "
                    f"device={self._config.depth_device} "
                    f"process_res={self._config.depth_process_res}"
                )
            else:
                error = None
                if depth_estimator is not None:
                    error = depth_estimator.last_error
                print(f"Depth: unavailable ({error or 'startup check failed'})")
        else:
            print("Depth: disabled")
        calibration_file = self._config.camera_calibration_file
        if self._config.enable_rectification and calibration_file is not None:
            print(
                "Rectification: enabled "
                f"camera={self._config.camera_calibration_camera_key} "
                f"calibration={calibration_file}"
            )
        else:
            print("Rectification: disabled")
        if self._config.enable_mapping:
            print(
                "Topdown map: enabled "
                f"map_id={self._config.map_id} "
                f"resolution_m={self._config.map_resolution_m} "
                f"size={self._config.map_width}x{self._config.map_height}"
            )
            print(
                "Storage: "
                f"db={self._config.storage_db_path} "
                f"artifacts={self._config.storage_artifact_dir}"
            )
        else:
            print("Topdown map: disabled")
        if self._config.enable_memory_index:
            print(
                "Memory index: enabled "
                f"dir={self._config.memory_index_dir} "
                f"model={self._config.memory_model_name} "
                f"pretrained={self._config.memory_pretrained} "
                f"device={self._config.memory_device}"
            )
        else:
            print("Memory index: disabled")
        if self._vlm_http_service is not None:
            print(
                "VLM API: enabled "
                f"http://{self._vlm_http_service.host}:{self._vlm_http_service.port}"
            )
            if not self._config.groq_api_key:
                print("VLM API: Groq key missing, query calls will fail until MPS_GROQ_API_KEY is set")
        else:
            print("VLM API: disabled")
        print(
            "Server path: SubmitPacket -> prepare/store -> ingress preview + "
            "depth queue -> depth preview/map/storage"
        )

        try:
            while not self._stop_event.is_set():
                self._preview_renderer.pump()
                if self._preview_renderer.quit_requested:
                    self._stop_event.set()
                    break
                if self._stop_event.wait(timeout=0.02):
                    break
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self._grpc_server.stop(grace=2.0).wait()
        if self._vlm_http_service is not None:
            self._vlm_http_service.stop()
        self._ingestion_service.stop()

        metrics = self._ingestion_service.snapshot_metrics()
        print(
            "Server stopped. "
            f"accepted={metrics.accepted_packets} "
            f"rejected={metrics.rejected_packets} "
            f"processed={metrics.processed_packets} "
            f"processing_errors={metrics.processing_errors}"
        )

    def _install_signal_handlers(self) -> None:
        def _handle_signal(_sig: int, _frame) -> None:
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    ServerApplication.from_env().run()


if __name__ == "__main__":
    main()
