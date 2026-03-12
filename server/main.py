from __future__ import annotations

from concurrent import futures
import os
from pathlib import Path
import signal
import sys
import threading

import grpc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proto import perception_pb2, perception_pb2_grpc
from pipeline import ServerPipelineRuntime


class PerceptionIngestServicer(perception_pb2_grpc.PerceptionIngestServiceServicer):
    def __init__(self, runtime: ServerPipelineRuntime) -> None:
        self._runtime = runtime

    def SubmitPacket(
        self, request: perception_pb2.SubmitPacketRequest, context: grpc.ServicerContext
    ) -> perception_pb2.SubmitPacketReply:
        packet = request.packet
        accepted = self._runtime.submit_packet(packet=packet, client_peer=context.peer())
        metrics = self._runtime.snapshot_metrics()

        if not accepted:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("server ingress queue is full")
            return perception_pb2.SubmitPacketReply(
                status="backpressure",
                message="server ingress queue is full",
                timestamp_ns=packet.timestamp_ns,
                received_packets=metrics.enqueued_packets,
                rejected_packets=metrics.rejected_packets,
            )

        reply = perception_pb2.SubmitPacketReply(
            status="ok",
            message="accepted",
            timestamp_ns=packet.timestamp_ns,
            received_packets=metrics.enqueued_packets,
            rejected_packets=metrics.rejected_packets,
        )
        topdown = self._runtime.snapshot_topdown_map()
        if topdown is not None:
            reply.topdown_map.CopyFrom(topdown)
        return reply


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean env {name}={raw}")


def main() -> None:
    bind_host = os.environ.get("MPS_SERVER_BIND_HOST", "0.0.0.0")
    bind_port = int(os.environ.get("MPS_SERVER_PORT", "19100"))
    grpc_workers = int(os.environ.get("MPS_SERVER_GRPC_WORKERS", "8"))
    ingress_queue_size = int(os.environ.get("MPS_SERVER_INGRESS_QUEUE_SIZE", "256"))
    worker_queue_size = int(os.environ.get("MPS_SERVER_WORKER_QUEUE_SIZE", "256"))
    enqueue_timeout_s = float(os.environ.get("MPS_SERVER_ENQUEUE_TIMEOUT_S", "0.02"))
    debug_visualizer_enabled = _env_bool("MPS_SERVER_DEBUG_VIS", default=False)
    debug_draw_every_n = int(os.environ.get("MPS_SERVER_DEBUG_EVERY_N", "1"))
    persistence_jsonl_path = Path(
        os.environ.get("MPS_SERVER_PERSIST_JSONL", "/tmp/mps_server_packets.jsonl")
    )

    stop_event = threading.Event()

    def _handle_signal(_sig: int, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    runtime = ServerPipelineRuntime(
        ingress_queue_size=ingress_queue_size,
        worker_queue_size=worker_queue_size,
        enqueue_timeout_s=enqueue_timeout_s,
        persistence_jsonl_path=persistence_jsonl_path,
        debug_visualizer_enabled=debug_visualizer_enabled,
        debug_draw_every_n=debug_draw_every_n,
    )
    runtime.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_workers))
    perception_pb2_grpc.add_PerceptionIngestServiceServicer_to_server(
        PerceptionIngestServicer(runtime), server
    )

    listen_addr = f"{bind_host}:{bind_port}"
    server.add_insecure_port(listen_addr)
    server.start()

    print(f"gRPC server listening on {listen_addr}")
    print("Pipeline: ingress queue -> router -> telemetry + persistence + topdown-map")
    print(f"Debug visualizer: {'on' if debug_visualizer_enabled else 'off'}")

    try:
        while not stop_event.is_set():
            runtime.pump_debug_visualizer(max_updates=1)
            wait_timeout = 0.01 if runtime.debug_visualizer_active() else 1.0
            stop_event.wait(timeout=wait_timeout)
    finally:
        server.stop(grace=2.0).wait()
        runtime.stop()
        metrics = runtime.snapshot_metrics()
        print(
            "Server stopped. "
            f"received={metrics.enqueued_packets} "
            f"rejected={metrics.rejected_packets} "
            f"router_drops={metrics.dropped_by_router} "
            f"telemetry_processed={metrics.telemetry_processed} "
            f"persistence_processed={metrics.persistence_processed} "
            f"mapping_processed={metrics.mapping_processed} "
            f"debug_visualized={metrics.debug_visualized}"
        )


if __name__ == "__main__":
    main()
