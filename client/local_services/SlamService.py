from __future__ import annotations

import base64
import json
import queue
import socket
import threading

import cv2
import numpy as np

from schema.SensorSchema import LocalSensorPacket, OrbResult


MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class SlamService:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19090,
        queue_size: int = 4,
        result_timeout_s: float = 1.5,
        connect_timeout_s: float = 0.5,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.result_timeout_s = result_timeout_s
        self.connect_timeout_s = connect_timeout_s

        self.input_queue: queue.Queue[LocalSensorPacket] = queue.Queue(maxsize=queue_size)
        self._results: dict[int, OrbResult] = {}
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._service_state = "NOT_STARTED"

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run, name="slam-service", daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def submit(self, packet: LocalSensorPacket) -> None:
        if not self.input_queue.full():
            self.input_queue.put_nowait(packet)
            return
        try:
            self.input_queue.get_nowait()
        except queue.Empty:
            pass
        self.input_queue.put_nowait(packet)

    def await_result(self, timestamp_ns: int, timeout_s: float = 1.0) -> OrbResult | None:
        with self._condition:
            if timestamp_ns in self._results:
                return self._results.pop(timestamp_ns)
            self._condition.wait_for(lambda: timestamp_ns in self._results, timeout=timeout_s)
            return self._results.pop(timestamp_ns, None)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result = self._process_packet(packet)

            with self._condition:
                self._results[packet.timestamp_ns] = result
                self._condition.notify_all()

    def _process_packet(self, packet: LocalSensorPacket) -> OrbResult:
        try:
            payload = self._request_orb(packet)
            self._service_state = "RUNNING"
            return _orb_result_from_payload(payload)
        except TimeoutError as exc:
            self._service_state = "ADAPTER_TIMEOUT"
            return OrbResult(tracking_state="NO_OUTPUT", error=str(exc))
        except ConnectionError as exc:
            self._service_state = "ADAPTER_UNAVAILABLE"
            return OrbResult(tracking_state=self._service_state, error=str(exc))
        except Exception as exc:
            self._service_state = "ADAPTER_PROTOCOL_ERROR"
            return OrbResult(tracking_state=self._service_state, error=f"unexpected adapter error: {exc}")

    def _request_orb(self, packet: LocalSensorPacket) -> dict:
        payload = _build_request_payload(packet)
        try:
            with socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s) as connection:
                connection.settimeout(self.result_timeout_s)
                _send_json_message(connection, payload)
                response = _recv_json_message(connection)
        except socket.timeout as exc:
            raise TimeoutError(
                f"timed out waiting for slam adapter {self.host}:{self.port}"
            ) from exc
        except OSError as exc:
            raise ConnectionError(
                f"cannot reach slam adapter {self.host}:{self.port}: {exc}"
            ) from exc
        return response


def _build_request_payload(packet: LocalSensorPacket) -> dict:
    ok, encoded = cv2.imencode(".jpg", packet.image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode request image")
    image_jpeg_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "type": "orb_request",
        "timestamp_ns": packet.timestamp_ns,
        "image_jpeg_b64": image_jpeg_b64,
        "imu_samples": [
            {
                "timestamp_ns": imu.timestamp_ns,
                "angular_velocity_rad_s": list(imu.angular_velocity_rad_s),
                "linear_acceleration_m_s2": list(imu.linear_acceleration_m_s2),
            }
            for imu in packet.imu_samples
        ],
    }


def _orb_result_from_payload(payload: dict) -> OrbResult:
    tracking_image_bgr = None
    decode_error: str | None = None

    tracking_image_jpeg_b64 = payload.get("tracking_image_jpeg_b64")
    if tracking_image_jpeg_b64:
        try:
            tracking_image_bgr = _decode_jpeg_b64(tracking_image_jpeg_b64)
        except Exception as exc:
            decode_error = f"tracking_image_decode_failed:{exc}"

    error_parts = []
    if payload.get("error"):
        error_parts.append(str(payload["error"]))
    if decode_error:
        error_parts.append(decode_error)
    error = " | ".join(error_parts) if error_parts else None

    return OrbResult(
        tracking_state=payload.get("tracking_state"),
        camera_translation_xyz=_tuple3(payload.get("camera_translation_xyz")),
        camera_quaternion_wxyz=_tuple4(payload.get("camera_quaternion_wxyz")),
        body_translation_xyz=_tuple3(payload.get("body_translation_xyz")),
        body_quaternion_wxyz=_tuple4(payload.get("body_quaternion_wxyz")),
        tracking_image_bgr=tracking_image_bgr,
        error=error,
    )


def _decode_jpeg_b64(jpeg_b64: str) -> np.ndarray:
    raw = base64.b64decode(jpeg_b64)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None")
    return image


def _send_json_message(connection: socket.socket, payload: dict) -> None:
    message = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise ValueError(f"request too large: {len(message)} bytes")
    header = len(message).to_bytes(4, byteorder="big", signed=False)
    connection.sendall(header + message)


def _recv_json_message(connection: socket.socket) -> dict:
    header = _recv_exact(connection, 4)
    message_size = int.from_bytes(header, byteorder="big", signed=False)
    if message_size <= 0 or message_size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid response size: {message_size}")
    body = _recv_exact(connection, message_size)
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    return payload


def _recv_exact(connection: socket.socket, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < expected_size:
        chunk = connection.recv(expected_size - received)
        if not chunk:
            raise ConnectionError("connection closed while receiving message")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _tuple3(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float] | None:
    if values is None or len(values) != 3:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple4(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float] | None:
    if values is None or len(values) != 4:
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
