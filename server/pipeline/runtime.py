from __future__ import annotations

from dataclasses import dataclass
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from proto import perception_pb2
from .debug_visualizer import DebugVisualizerConfig, ServerDebugVisualizer
from .topdown_mapper import TopDownMapper


@dataclass(frozen=True)
class PacketEnvelope:
    packet: perception_pb2.ServerPerceptionPacket
    received_unix_s: float
    client_peer: str


@dataclass
class PipelineMetrics:
    enqueued_packets: int = 0
    rejected_packets: int = 0
    dropped_by_router: int = 0
    telemetry_processed: int = 0
    persistence_processed: int = 0
    mapping_processed: int = 0
    debug_visualized: int = 0
    last_timestamp_ns: int | None = None


class ServerPipelineRuntime:
    def __init__(
        self,
        ingress_queue_size: int = 256,
        worker_queue_size: int = 256,
        enqueue_timeout_s: float = 0.02,
        persistence_jsonl_path: Path | None = None,
        debug_visualizer_enabled: bool = False,
        debug_draw_every_n: int = 1,
    ) -> None:
        self.ingress_queue: queue.Queue[PacketEnvelope] = queue.Queue(maxsize=ingress_queue_size)
        self._telemetry_queue: queue.Queue[PacketEnvelope] = queue.Queue(maxsize=worker_queue_size)
        self._persistence_queue: queue.Queue[PacketEnvelope] = queue.Queue(maxsize=worker_queue_size)
        self._mapping_queue: queue.Queue[PacketEnvelope] = queue.Queue(maxsize=worker_queue_size)
        self._debug_queue: queue.Queue[PacketEnvelope] = queue.Queue(maxsize=max(8, worker_queue_size // 2))

        self._enqueue_timeout_s = enqueue_timeout_s
        self._persistence_jsonl_path = persistence_jsonl_path
        self._debug_visualizer_enabled = debug_visualizer_enabled
        self._debug_draw_every_n = max(1, debug_draw_every_n)
        self._debug_visualizer: ServerDebugVisualizer | None = None
        self._topdown_mapper = TopDownMapper()
        self._latest_topdown_map = perception_pb2.TopDownMap()
        self._has_topdown_map = False

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        self._metrics = PipelineMetrics()
        self._metrics_lock = threading.Lock()
        self._jsonl_lock = threading.Lock()
        self._topdown_lock = threading.Lock()

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        if self._debug_visualizer_enabled:
            self._try_init_debug_visualizer()
        self._threads = [
            threading.Thread(target=self._run_router, name="server-router", daemon=True),
            threading.Thread(target=self._run_telemetry_pipeline, name="server-telemetry", daemon=True),
            threading.Thread(target=self._run_persistence_pipeline, name="server-persistence", daemon=True),
            threading.Thread(target=self._run_mapping_pipeline, name="server-topdown-map", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        if self._debug_visualizer is not None:
            self._debug_visualizer.close()
            self._debug_visualizer = None

    def submit_packet(self, packet: perception_pb2.ServerPerceptionPacket, client_peer: str) -> bool:
        envelope = PacketEnvelope(packet=packet, received_unix_s=time.time(), client_peer=client_peer)
        try:
            self.ingress_queue.put(envelope, timeout=self._enqueue_timeout_s)
        except queue.Full:
            with self._metrics_lock:
                self._metrics.rejected_packets += 1
            return False

        with self._metrics_lock:
            self._metrics.enqueued_packets += 1
            self._metrics.last_timestamp_ns = packet.timestamp_ns
        return True

    def snapshot_metrics(self) -> PipelineMetrics:
        with self._metrics_lock:
            return PipelineMetrics(**self._metrics.__dict__)

    def snapshot_topdown_map(self) -> perception_pb2.TopDownMap | None:
        with self._topdown_lock:
            if not self._has_topdown_map:
                return None
            out = perception_pb2.TopDownMap()
            out.CopyFrom(self._latest_topdown_map)
            return out

    def debug_visualizer_active(self) -> bool:
        return self._debug_visualizer is not None

    def pump_debug_visualizer(self, max_updates: int = 1) -> None:
        if self._debug_visualizer is None:
            return

        updates = 0
        while updates < max(1, int(max_updates)):
            try:
                envelope = self._debug_queue.get_nowait()
            except queue.Empty:
                try:
                    self._debug_visualizer.pump_events()
                except Exception as exc:
                    print(f"[debug] visualizer disabled: {exc}")
                    self._debug_visualizer = None
                return

            try:
                self._debug_visualizer.update(envelope, self.snapshot_metrics())
                with self._metrics_lock:
                    self._metrics.debug_visualized += 1
            except Exception as exc:
                print(f"[debug] visualizer disabled: {exc}")
                self._debug_visualizer = None
                return
            updates += 1

    def _run_router(self) -> None:
        while not self._stop_event.is_set():
            try:
                envelope = self.ingress_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._fan_out(self._telemetry_queue, envelope)
            self._fan_out(self._persistence_queue, envelope)
            self._fan_out(self._mapping_queue, envelope)
            if self._debug_visualizer is not None:
                self._fan_out(self._debug_queue, envelope)

    def _fan_out(self, output_queue: queue.Queue[PacketEnvelope], envelope: PacketEnvelope) -> None:
        if not output_queue.full():
            output_queue.put_nowait(envelope)
            return

        try:
            output_queue.get_nowait()
        except queue.Empty:
            pass
        output_queue.put_nowait(envelope)
        with self._metrics_lock:
            self._metrics.dropped_by_router += 1

    def _run_telemetry_pipeline(self) -> None:
        while not self._stop_event.is_set():
            try:
                envelope = self._telemetry_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._metrics_lock:
                self._metrics.telemetry_processed += 1
                count = self._metrics.telemetry_processed
                rejected = self._metrics.rejected_packets
            if count == 1 or count % 20 == 0:
                print(
                    f"[telemetry] processed={count} "
                    f"rejected={rejected} "
                    f"timestamp_ns={envelope.packet.timestamp_ns}"
                )

    def _run_persistence_pipeline(self) -> None:
        while not self._stop_event.is_set():
            try:
                envelope = self._persistence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._metrics_lock:
                self._metrics.persistence_processed += 1

            if self._persistence_jsonl_path is None:
                continue
            self._append_jsonl(envelope)

    def _run_mapping_pipeline(self) -> None:
        while not self._stop_event.is_set():
            try:
                envelope = self._mapping_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._topdown_mapper.enabled:
                continue

            try:
                topdown = self._topdown_mapper.process_packet(envelope.packet)
            except Exception as exc:
                print(f"[topdown] mapping worker error: {exc}")
                continue
            if topdown is None:
                continue
            with self._topdown_lock:
                self._latest_topdown_map.CopyFrom(topdown)
                self._has_topdown_map = True
            with self._metrics_lock:
                self._metrics.mapping_processed += 1

    def _append_jsonl(self, envelope: PacketEnvelope) -> None:
        packet = envelope.packet
        record = {
            "received_unix_s": envelope.received_unix_s,
            "client_peer": envelope.client_peer,
            "timestamp_ns": packet.timestamp_ns,
            "imu_samples": len(packet.imu_samples),
            "has_orb_tracking": packet.HasField("orb_tracking"),
            "has_yolo_output": packet.HasField("yolo_output"),
            "image_jpeg_bytes": len(packet.image_jpeg),
            "tracking_state": packet.orb_tracking.tracking_state if packet.HasField("orb_tracking") else None,
            "yolo_detections": len(packet.yolo_output.detections) if packet.HasField("yolo_output") else 0,
        }
        self._persistence_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_lock:
            with self._persistence_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")

    def _try_init_debug_visualizer(self) -> None:
        if not _has_gui_display():
            self._debug_visualizer = None
            print("[telemetry] debug visualizer unavailable: no valid GUI display")
            return
        try:
            self._debug_visualizer = ServerDebugVisualizer(
                DebugVisualizerConfig(draw_every_n=self._debug_draw_every_n)
            )
            print(
                "[telemetry] debug visualizer enabled "
                f"(draw_every_n={self._debug_draw_every_n})"
            )
        except Exception as exc:
            self._debug_visualizer = None
            print(f"[telemetry] debug visualizer unavailable: {exc}")


def _has_gui_display() -> bool:
    display = os.environ.get("DISPLAY")
    if not display:
        return False
    try:
        probe = subprocess.run(
            ["xdpyinfo"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return probe.returncode == 0
