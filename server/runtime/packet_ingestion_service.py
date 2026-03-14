from __future__ import annotations

import queue
import threading
import time

from proto import perception_pb2
from server.rendering import ServerPreviewRenderer
from server.runtime.packet_models import IngestionMetrics, PacketEnvelope


class PacketIngestionService:
    def __init__(
        self,
        *,
        queue_size: int = 256,
        enqueue_timeout_s: float = 0.02,
        log_every_n_packets: int = 20,
        preview_renderer: ServerPreviewRenderer,
    ) -> None:
        self._queue: queue.Queue[PacketEnvelope] = queue.Queue(maxsize=queue_size)
        self._enqueue_timeout_s = enqueue_timeout_s
        self._log_every_n_packets = max(1, log_every_n_packets)
        self._preview_renderer = preview_renderer

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._metrics = IngestionMetrics()
        self._metrics_lock = threading.Lock()

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._preview_renderer.start()
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            name="packet-ingestion-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self._preview_renderer.stop()

    def submit_packet(self, packet: perception_pb2.ServerPerceptionPacket, client_peer: str) -> bool:
        envelope = PacketEnvelope(packet=packet, received_unix_s=time.time(), client_peer=client_peer)
        try:
            self._queue.put(envelope, timeout=self._enqueue_timeout_s)
        except queue.Full:
            with self._metrics_lock:
                self._metrics.rejected_packets += 1
            return False

        with self._metrics_lock:
            self._metrics.accepted_packets += 1
            self._metrics.last_timestamp_ns = packet.timestamp_ns
        return True

    def snapshot_metrics(self) -> IngestionMetrics:
        with self._metrics_lock:
            return IngestionMetrics(**self._metrics.__dict__)

    def process_packet(self, envelope: PacketEnvelope) -> None:
        self._preview_renderer.publish(envelope)

    def _run_worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                envelope = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.process_packet(envelope)
            except Exception as exc:
                with self._metrics_lock:
                    self._metrics.processing_errors += 1
                print(f"[ingestion] processing error: {exc}")

            with self._metrics_lock:
                self._metrics.processed_packets += 1
                processed = self._metrics.processed_packets
                rejected = self._metrics.rejected_packets
                timestamp_ns = envelope.packet.timestamp_ns

            if processed == 1 or processed % self._log_every_n_packets == 0:
                print(
                    f"[ingestion] processed={processed} "
                    f"rejected={rejected} "
                    f"timestamp_ns={timestamp_ns}"
                )

            self._queue.task_done()
