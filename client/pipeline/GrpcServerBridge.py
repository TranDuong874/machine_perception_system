from __future__ import annotations

import os
from pathlib import Path
import sys

import cv2
import grpc

from schema.SensorSchema import ServerPerceptionPacket


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proto import perception_pb2, perception_pb2_grpc  # noqa: E402


class GrpcServerBridgeClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self._target = f"{host}:{int(port)}"
        self._timeout_s = float(timeout_s)
        self._channel = grpc.insecure_channel(self._target)
        self._stub = perception_pb2_grpc.PerceptionIngestServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def send_packet(self, packet: ServerPerceptionPacket) -> perception_pb2.SubmitPacketReply:
        request = perception_pb2.SubmitPacketRequest(packet=_to_proto_packet(packet))
        return self._stub.SubmitPacket(request, timeout=self._timeout_s)


def create_from_env() -> GrpcServerBridgeClient:
    host = os.environ.get("MPS_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("MPS_SERVER_PORT", "19100"))
    timeout_s = float(os.environ.get("MPS_SERVER_TIMEOUT_S", "0.4"))
    return GrpcServerBridgeClient(host=host, port=port, timeout_s=timeout_s)


def _to_proto_packet(packet: ServerPerceptionPacket) -> perception_pb2.ServerPerceptionPacket:
    proto_packet = perception_pb2.ServerPerceptionPacket(
        timestamp_ns=int(packet.timestamp_ns),
        image_jpeg=_encode_jpeg(packet.image_bgr),
    )

    for imu in packet.imu_samples:
        proto_packet.imu_samples.add(
            timestamp_ns=int(imu.timestamp_ns),
            angular_velocity_rad_s=list(imu.angular_velocity_rad_s),
            linear_acceleration_m_s2=list(imu.linear_acceleration_m_s2),
        )

    if packet.orb_tracking is not None:
        orb = proto_packet.orb_tracking
        if packet.orb_tracking.tracking_state is not None:
            orb.tracking_state = packet.orb_tracking.tracking_state
        if packet.orb_tracking.camera_translation_xyz is not None:
            orb.camera_translation_xyz[:] = list(packet.orb_tracking.camera_translation_xyz)
        if packet.orb_tracking.camera_quaternion_wxyz is not None:
            orb.camera_quaternion_wxyz[:] = list(packet.orb_tracking.camera_quaternion_wxyz)
        if packet.orb_tracking.body_translation_xyz is not None:
            orb.body_translation_xyz[:] = list(packet.orb_tracking.body_translation_xyz)
        if packet.orb_tracking.body_quaternion_wxyz is not None:
            orb.body_quaternion_wxyz[:] = list(packet.orb_tracking.body_quaternion_wxyz)
        if packet.orb_tracking.tracking_image_bgr is not None:
            orb.tracking_image_jpeg = _encode_jpeg(packet.orb_tracking.tracking_image_bgr)
        if packet.orb_tracking.error:
            orb.error = packet.orb_tracking.error

    if packet.yolo_output is not None:
        yolo = proto_packet.yolo_output
        for det in packet.yolo_output.detections:
            yolo.detections.add(
                class_id=int(det.class_id),
                class_name=det.class_name,
                confidence=float(det.confidence),
                xyxy=list(det.xyxy),
            )
        if packet.yolo_output.annotated_image_bgr is not None:
            yolo.annotated_image_jpeg = _encode_jpeg(packet.yolo_output.annotated_image_bgr)
        if packet.yolo_output.error:
            yolo.error = packet.yolo_output.error

    if packet.segmentation_mask is not None:
        ok, encoded_png = cv2.imencode(".png", packet.segmentation_mask)
        if ok:
            proto_packet.segmentation_mask_png = encoded_png.tobytes()

    return proto_packet


def _encode_jpeg(image_bgr, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode image to JPEG")
    return encoded.tobytes()
