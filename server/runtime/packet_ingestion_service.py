from __future__ import annotations

import threading
import time

from proto import perception_pb2
from server.runtime.frame_store_service import FrameStoreService
from server.runtime.packet_models import IngestionMetrics, PacketEnvelope
from server.runtime.processing_models import ProcessedPacket, TopDownMapSnapshot
from server.rendering.preview_renderer import ServerPreviewRenderer
from server.services.frame_selection_service import FrameSelectionService
from server.services.frame_worker_service import FrameWorkerService
from server.services.memory_index_service import MemoryIndexService
from server.services.perception_processing_service import PerceptionProcessingService
from server.services.perception_storage_service import PerceptionStorageService
from server.services.topdown_mapping_service import TopDownMappingService


class PacketIngestionService:
    def __init__(
        self,
        *,
        log_every_n_packets: int = 20,
        preview_renderer: ServerPreviewRenderer,
        processing_service: PerceptionProcessingService,
        mapping_service: TopDownMappingService,
        memory_index_service: MemoryIndexService,
        storage_service: PerceptionStorageService,
        frame_store: FrameStoreService,
        selection_service: FrameSelectionService,
        depth_task_queue_size: int,
        memory_task_queue_size: int,
    ) -> None:
        self._log_every_n_packets = max(1, log_every_n_packets)
        self._preview_renderer = preview_renderer
        self._processing_service = processing_service
        self._mapping_service = mapping_service
        self._memory_index_service = memory_index_service
        self._storage_service = storage_service
        self._frame_store = frame_store
        self._selection_service = selection_service

        self._metrics = IngestionMetrics()
        self._metrics_lock = threading.Lock()
        self._depth_worker = FrameWorkerService(
            name="depth-map-worker",
            queue_size=depth_task_queue_size,
            handler=self._run_depth_task,
            drop_oldest_when_full=False,
        )
        self._memory_worker = FrameWorkerService(
            name="memory-worker",
            queue_size=memory_task_queue_size,
            handler=self._run_memory_task,
            drop_oldest_when_full=True,
        )

    def start(self) -> None:
        self._preview_renderer.start()
        self._depth_worker.start()
        self._memory_worker.start()

    def stop(self) -> None:
        self._depth_worker.stop()
        self._memory_worker.stop()
        self._preview_renderer.stop()
        self._memory_index_service.close()
        self._storage_service.close()

    def submit_packet(self, packet: perception_pb2.ServerPerceptionPacket, client_peer: str) -> bool:
        envelope = PacketEnvelope(packet=packet, received_unix_s=time.time(), client_peer=client_peer)
        try:
            prepared_packet = self._processing_service.prepare(envelope)
            self._storage_service.record_ingested_frame(prepared_packet)
            self._frame_store.put(prepared_packet)
            self._preview_renderer.publish_ingress(prepared_packet)
            decision = self._selection_service.decide(prepared_packet)
        except Exception as exc:
            with self._metrics_lock:
                self._metrics.rejected_packets += 1
                self._metrics.processing_errors += 1
            print(f"[ingestion] ingress error: {exc}")
            return False

        depth_enqueued = False
        memory_enqueued = False
        if decision.schedule_depth:
            depth_enqueued = self._depth_worker.submit(packet.timestamp_ns)
        if decision.schedule_memory:
            memory_enqueued = self._memory_worker.submit(packet.timestamp_ns)

        with self._metrics_lock:
            self._metrics.accepted_packets += 1
            self._metrics.processed_packets += 1
            self._metrics.last_timestamp_ns = packet.timestamp_ns
            if decision.schedule_depth:
                if depth_enqueued:
                    self._metrics.depth_jobs_enqueued += 1
                else:
                    self._metrics.depth_jobs_dropped += 1
            if decision.schedule_memory:
                if memory_enqueued:
                    self._metrics.memory_jobs_enqueued += 1
                else:
                    self._metrics.memory_jobs_dropped += 1
            processed = self._metrics.processed_packets
            rejected = self._metrics.rejected_packets
            depth_dropped = self._metrics.depth_jobs_dropped
            memory_dropped = self._metrics.memory_jobs_dropped

        if processed == 1 or processed % self._log_every_n_packets == 0:
            print(
                f"[ingestion] accepted={processed} "
                f"rejected={rejected} "
                f"timestamp_ns={packet.timestamp_ns} "
                f"depth_dropped={depth_dropped} "
                f"memory_dropped={memory_dropped}"
            )
        return True

    def snapshot_metrics(self) -> IngestionMetrics:
        with self._metrics_lock:
            return IngestionMetrics(**self._metrics.__dict__)

    def snapshot_latest_map(self) -> TopDownMapSnapshot | None:
        return self._mapping_service.snapshot()

    def _run_depth_task(self, timestamp_ns: int) -> None:
        prepared_packet = self._frame_store.get(timestamp_ns)
        if prepared_packet is None:
            return
        processed_packet = self._processing_service.enrich_with_depth(_clone_packet(prepared_packet))
        processed_packet.map_snapshot = self._mapping_service.update(processed_packet)
        self._storage_service.record_map_update(processed_packet)
        self._frame_store.put(processed_packet)
        self._preview_renderer.publish_depth(processed_packet)

    def _run_memory_task(self, timestamp_ns: int) -> None:
        prepared_packet = self._frame_store.get(timestamp_ns)
        if prepared_packet is None:
            return
        memory_packet = _clone_packet(prepared_packet)
        memory_packet.map_snapshot = self._mapping_service.snapshot()
        memory_record = self._memory_index_service.add_packet(memory_packet)
        if memory_record is None:
            return
        memory_packet.memory_record = memory_record
        self._storage_service.record_memory_entry(
            memory_packet,
            memory_record=memory_record,
        )


def _clone_packet(processed_packet: ProcessedPacket) -> ProcessedPacket:
    return ProcessedPacket(
        envelope=processed_packet.envelope,
        source_frame_bgr=processed_packet.source_frame_bgr,
        rectified_frame=processed_packet.rectified_frame,
        segmentation_mask=processed_packet.segmentation_mask,
        pose=processed_packet.pose,
        depth_result=processed_packet.depth_result,
        projected_points_world=processed_packet.projected_points_world,
        yolo_detection_count=processed_packet.yolo_detection_count,
        reprojection_confidence_threshold=processed_packet.reprojection_confidence_threshold,
        memory_record=processed_packet.memory_record,
        map_snapshot=processed_packet.map_snapshot,
    )
