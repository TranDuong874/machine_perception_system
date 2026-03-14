from __future__ import annotations

import grpc

from proto import perception_pb2, perception_pb2_grpc
from server.runtime.packet_ingestion_service import PacketIngestionService


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
        latest_map = self._ingestion_service.snapshot_latest_map()

        if not accepted:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("server failed to accept packet")
            reply = perception_pb2.SubmitPacketReply(
                status="rejected",
                message="server failed to accept packet",
                timestamp_ns=packet.timestamp_ns,
                received_packets=metrics.accepted_packets,
                rejected_packets=metrics.rejected_packets,
            )
            if latest_map is not None:
                reply.topdown_map.CopyFrom(_to_proto_topdown_map(latest_map))
            return reply

        reply = perception_pb2.SubmitPacketReply(
            status="ok",
            message="accepted",
            timestamp_ns=packet.timestamp_ns,
            received_packets=metrics.accepted_packets,
            rejected_packets=metrics.rejected_packets,
        )
        if latest_map is not None:
            reply.topdown_map.CopyFrom(_to_proto_topdown_map(latest_map))
        return reply


def _to_proto_topdown_map(map_snapshot) -> perception_pb2.TopDownMap:
    latest_position = (
        list(map_snapshot.latest_user_translation_xyz)
        if map_snapshot.latest_user_translation_xyz is not None
        else []
    )
    return perception_pb2.TopDownMap(
        map_id=map_snapshot.map_id,
        revision=int(map_snapshot.revision),
        timestamp_ns=int(map_snapshot.timestamp_ns),
        tracking_state=map_snapshot.tracking_state,
        pose_valid=bool(map_snapshot.pose_valid),
        resolution_m=float(map_snapshot.resolution_m),
        width=int(map_snapshot.width),
        height=int(map_snapshot.height),
        origin_x_m=float(map_snapshot.origin_x_m),
        origin_z_m=float(map_snapshot.origin_z_m),
        latest_user_translation_xyz=latest_position,
        occupancy_png=map_snapshot.occupancy_png,
        backend=map_snapshot.backend,
    )
