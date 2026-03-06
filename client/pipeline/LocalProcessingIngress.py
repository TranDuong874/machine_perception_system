from __future__ import annotations

from schema.SensorSchema import IMUSample, InputSample, LocalSensorPacket

# TODO: Add interpolation for live input and define exact frame interval semantics.

class LocalProcessingIngress:
    # Entry boundary from raw sensor samples into downstream local processing.
    def __init__(self) -> None:
        self._pending_imu: list[IMUSample] = []

    def push(self, sample: InputSample) -> LocalSensorPacket | None:
        if isinstance(sample, IMUSample):
            self._pending_imu.append(sample)
            return None

        # Current behavior groups IMU samples up to the frame timestamp.
        # This is good enough for initial emission tests, but it does not yet
        # interpolate synthetic boundary samples at frame timestamps.
        imu_samples = tuple(
            imu_sample
            for imu_sample in self._pending_imu
            if imu_sample.timestamp_ns <= sample.timestamp_ns
        )
        self._pending_imu = [
            imu_sample
            for imu_sample in self._pending_imu
            if imu_sample.timestamp_ns > sample.timestamp_ns
        ]
        return LocalSensorPacket(
            timestamp_ns=sample.timestamp_ns,
            image_bgr=sample.image_bgr,
            imu_samples=imu_samples,
        )

    def reset(self) -> None:
        self._pending_imu.clear()
