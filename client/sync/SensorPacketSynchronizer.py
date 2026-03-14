from __future__ import annotations

from collections import deque

from schema.SensorSchema import IMUSample, InputSample, SynchronizedSensorPacket


class SensorPacketSynchronizer:
    """Group IMU samples with the next frame to emit a synchronized packet."""

    def __init__(self, max_pending_imu: int = 4000) -> None:
        self._pending_imu: deque[IMUSample] = deque(maxlen=max_pending_imu)

    def push(self, sample: InputSample) -> SynchronizedSensorPacket | None:
        if isinstance(sample, IMUSample):
            self._pending_imu.append(sample)
            return None

        frame_timestamp_ns = sample.timestamp_ns
        imu_for_frame: list[IMUSample] = []
        while self._pending_imu and self._pending_imu[0].timestamp_ns <= frame_timestamp_ns:
            imu_for_frame.append(self._pending_imu.popleft())

        return SynchronizedSensorPacket(
            timestamp_ns=frame_timestamp_ns,
            image_bgr=sample.image_bgr,
            imu_samples=tuple(imu_for_frame),
        )

    def reset(self) -> None:
        self._pending_imu.clear()
