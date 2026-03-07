from __future__ import annotations

from collections import deque

from schema.SensorSchema import IMUSample, InputSample, LocalSensorPacket

# TODO: Add interpolation for live input and define exact frame interval semantics.

class LocalProcessingIngress:
    # Entry boundary from raw sensor samples into downstream local processing.
    def __init__(self, max_pending_imu: int = 4000) -> None:
        # Bound IMU buffering so live ingestion remains memory-stable under load.
        self._pending_imu: deque[IMUSample] = deque(maxlen=max_pending_imu)

    def push(self, sample: InputSample) -> LocalSensorPacket | None:
        if isinstance(sample, IMUSample):
            self._pending_imu.append(sample)
            return None

        # Current behavior groups IMU samples up to the frame timestamp.
        # This is good enough for initial emission tests, but it does not yet
        # interpolate synthetic boundary samples at frame timestamps.
        frame_timestamp_ns = sample.timestamp_ns
        imu_for_frame: list[IMUSample] = []
        while self._pending_imu and self._pending_imu[0].timestamp_ns <= frame_timestamp_ns:
            imu_for_frame.append(self._pending_imu.popleft())

        return LocalSensorPacket(
            timestamp_ns=frame_timestamp_ns,
            image_bgr=sample.image_bgr,
            imu_samples=tuple(imu_for_frame),
        )

    def reset(self) -> None:
        self._pending_imu.clear()
