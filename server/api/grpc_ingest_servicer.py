from __future__ import annotations

import grpc

from proto import perception_pb2, perception_pb2_grpc
from server.runtime import PacketIngestionService


class PacketIngressServicer(perception_pb2_grpc.PerceptionIngestServiceServicer):
    def __init__(self, ingestion_service: PacketIngestionService) -> None:
        self._ingestion_service = ingestion_service

    def SubmitPacket(
        self,
        request: perception_pb2.SubmitPacketRequest,
        context: grpc.ServicerContext,
    ) -> perception_pb2.SubmitPacketReply:
        packet = request.packet
        accepted = self._ingestion_service.submit_packet(packet=packet, client_peer=context.peer())
        metrics = self._ingestion_service.snapshot_metrics()

        if not accepted:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("server ingress queue is full")
            return perception_pb2.SubmitPacketReply(
                status="backpressure",
                message="server ingress queue is full",
                timestamp_ns=packet.timestamp_ns,
                received_packets=metrics.accepted_packets,
                rejected_packets=metrics.rejected_packets,
            )

        return perception_pb2.SubmitPacketReply(
            status="ok",
            message="accepted",
            timestamp_ns=packet.timestamp_ns,
            received_packets=metrics.accepted_packets,
            rejected_packets=metrics.rejected_packets,
        )
