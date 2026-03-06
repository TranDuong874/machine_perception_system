from __future__ import annotations

import queue
import threading

from schema.InputSensorSchema import SensorPacket
from schema.PerceptionSchema import PerceptionPacket
from schema.PoseSchema import PoseSample


class SensorFusionPipeline:
    """
    Fuses sensor packets with nearest pose samples and emits perception packets.
    """

    def __init__(
        self,
        input_queue: queue.Queue[SensorPacket],
        output_queue_size: int = 8,
        poll_timeout_s: float = 0.1,
        pose_time_tolerance_ns: int = 50_000_000,
        max_pose_buffer_size: int = 256,
        drop_oldest_when_full: bool = True,
        emit_without_pose: bool = False,
        auto_estimate_pose_time_offset: bool = True,
        use_latest_pose_if_unsynced: bool = True,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue: queue.Queue[PerceptionPacket] = queue.Queue(maxsize=output_queue_size)

        self.poll_timeout_s = poll_timeout_s
        self.pose_time_tolerance_ns = pose_time_tolerance_ns
        self.max_pose_buffer_size = max_pose_buffer_size
        self.drop_oldest_when_full = drop_oldest_when_full
        self.emit_without_pose = emit_without_pose
        self.auto_estimate_pose_time_offset = auto_estimate_pose_time_offset
        self.use_latest_pose_if_unsynced = use_latest_pose_if_unsynced

        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.upstream_done_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self._pose_lock = threading.Lock()
        self._pose_buffer: list[PoseSample] = []
        self._pose_clock_offset_ns: int | None = None

    def start(self, upstream_done_event: threading.Event | None = None) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            raise RuntimeError("SensorFusionPipeline is already running.")

        if upstream_done_event is not None:
            self.upstream_done_event = upstream_done_event

        self.stop_event.clear()
        self.done_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run,
            name="sensor-fusion-pipeline",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def update_pose(self, pose_sample: PoseSample) -> None:
        with self._pose_lock:
            self._pose_buffer.append(pose_sample)
            if len(self._pose_buffer) > self.max_pose_buffer_size:
                overflow = len(self._pose_buffer) - self.max_pose_buffer_size
                if overflow > 0:
                    del self._pose_buffer[:overflow]

    def read(self, timeout_s: float | None = None) -> PerceptionPacket | None:
        timeout = self.poll_timeout_s if timeout_s is None else timeout_s
        while True:
            if self.done_event.is_set() and self.output_queue.empty():
                return None
            try:
                return self.output_queue.get(timeout=timeout)
            except queue.Empty:
                if self.done_event.is_set():
                    return None

    def is_done(self) -> bool:
        return self.done_event.is_set() and self.output_queue.empty()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                if self.upstream_done_event.is_set() and self.input_queue.empty():
                    break

                try:
                    sensor_packet = self.input_queue.get(timeout=self.poll_timeout_s)
                except queue.Empty:
                    continue

                pose_sample = self._nearest_pose(sensor_packet.timestamp_ns)
                if pose_sample is None and self.auto_estimate_pose_time_offset:
                    self._estimate_pose_clock_offset(sensor_packet.timestamp_ns)
                    pose_sample = self._nearest_pose(sensor_packet.timestamp_ns)

                if pose_sample is None:
                    if self.use_latest_pose_if_unsynced:
                        pose_sample = self._latest_pose_with_state("TRACKING_UNSYNC")
                    if not self.emit_without_pose:
                        if pose_sample is None:
                            continue
                    if pose_sample is None:
                        pose_sample = PoseSample(
                            timestamp_ns=sensor_packet.timestamp_ns,
                            translation_xyz=(0.0, 0.0, 0.0),
                            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                            tracking_state="NO_POSE",
                        )

                perception_packet = PerceptionPacket(
                    timestamp_ns=sensor_packet.timestamp_ns,
                    frame_index=sensor_packet.frame_index,
                    image_bgr=sensor_packet.image_bgr,
                    angular_velocity_rad_s=sensor_packet.angular_velocity_rad_s,
                    linear_acceleration_m_s2=sensor_packet.linear_acceleration_m_s2,
                    pose=pose_sample,
                )
                self._push_output(perception_packet)
        finally:
            self.done_event.set()
            self.stop_event.set()

    def _nearest_pose(self, timestamp_ns: int) -> PoseSample | None:
        with self._pose_lock:
            if not self._pose_buffer:
                return None
            offset_ns = self._pose_clock_offset_ns or 0
            nearest = min(
                self._pose_buffer,
                key=lambda pose: abs((pose.timestamp_ns + offset_ns) - timestamp_ns),
            )

        nearest_delta = abs((nearest.timestamp_ns + (self._pose_clock_offset_ns or 0)) - timestamp_ns)
        if nearest_delta > self.pose_time_tolerance_ns:
            return None
        return nearest

    def _estimate_pose_clock_offset(self, sensor_timestamp_ns: int) -> None:
        with self._pose_lock:
            if self._pose_clock_offset_ns is not None or not self._pose_buffer:
                return
            latest_pose = self._pose_buffer[-1]
            self._pose_clock_offset_ns = sensor_timestamp_ns - latest_pose.timestamp_ns

    def _latest_pose_with_state(self, tracking_state: str) -> PoseSample | None:
        with self._pose_lock:
            if not self._pose_buffer:
                return None
            latest = self._pose_buffer[-1]
        return PoseSample(
            timestamp_ns=latest.timestamp_ns,
            translation_xyz=latest.translation_xyz,
            quaternion_wxyz=latest.quaternion_wxyz,
            tracking_state=tracking_state,
        )

    def _push_output(self, packet: PerceptionPacket) -> None:
        if not self.output_queue.full():
            self.output_queue.put_nowait(packet)
            return

        if not self.drop_oldest_when_full:
            return

        try:
            self.output_queue.get_nowait()
        except queue.Empty:
            pass
        self.output_queue.put_nowait(packet)
