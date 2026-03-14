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
from server.api import PacketIngressServicer
from server.rendering import ServerPreviewRenderer
from server.runtime import PacketIngestionService
from server.settings import load_server_config
from server.services import DepthEstimationService


class ServerApplication:
    def __init__(
        self,
        *,
        config,
        ingestion_service: PacketIngestionService,
        preview_renderer: ServerPreviewRenderer,
    ) -> None:
        self._config = config
        self._ingestion_service = ingestion_service
        self._preview_renderer = preview_renderer
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
        preview_renderer = ServerPreviewRenderer(
            enable_gui=config.enable_preview_gui,
            frame_delay_ms=config.preview_frame_delay_ms,
            headless_preview_path=config.headless_preview_path,
            depth_estimator=depth_estimator,
        )
        ingestion_service = PacketIngestionService(
            queue_size=config.ingress_queue_size,
            enqueue_timeout_s=config.enqueue_timeout_s,
            log_every_n_packets=config.log_every_n_packets,
            preview_renderer=preview_renderer,
        )
        return cls(
            config=config,
            ingestion_service=ingestion_service,
            preview_renderer=preview_renderer,
        )

    def run(self) -> None:
        self._install_signal_handlers()
        listen_address = f"{self._config.bind_host}:{self._config.bind_port}"
        self._grpc_server.add_insecure_port(listen_address)

        self._ingestion_service.start()
        self._grpc_server.start()

        print(f"gRPC server listening on {listen_address}")
        if self._config.enable_preview_gui:
            print("Preview: GUI window enabled")
        else:
            print(f"Preview: headless -> {self._config.headless_preview_path}")
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
        print("Server path: SubmitPacket -> ingestion queue -> preview renderer")

        try:
            while not self._stop_event.wait(timeout=0.2):
                if self._preview_renderer.quit_requested:
                    self._stop_event.set()
                pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        self._grpc_server.stop(grace=2.0).wait()
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
