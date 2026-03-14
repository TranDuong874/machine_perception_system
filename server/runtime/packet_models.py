from __future__ import annotations

from dataclasses import dataclass

from proto import perception_pb2


@dataclass(frozen=True)
class PacketEnvelope:
    packet: perception_pb2.ServerPerceptionPacket
    received_unix_s: float
    client_peer: str


@dataclass
class IngestionMetrics:
    accepted_packets: int = 0
    rejected_packets: int = 0
    processed_packets: int = 0
    processing_errors: int = 0
    last_timestamp_ns: int | None = None
    depth_jobs_enqueued: int = 0
    depth_jobs_dropped: int = 0
    memory_jobs_enqueued: int = 0
    memory_jobs_dropped: int = 0
