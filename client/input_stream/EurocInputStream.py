from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .InputStreamBase import InputStreamBase
from schema.SensorInputSchema import FrameSample, IMUSample, SensorPacket


class EurocInputStream(InputStreamBase):
    def __init__(self, dataset_root, monocular=True, inertial=True):
        self.dataset_root = Path(dataset_root)
        self.monocular = monocular
        self.inertial = inertial

        self.cam0_dir = self.dataset_root / "mav0" / "cam0"
        self.imu0_dir = self.dataset_root / "mav0" / "imu0"

        self._frame_iter: Iterator[FrameSample] | None = None
        self._current_frame: FrameSample | None = None
        self._next_frame: FrameSample | None = None

        self._imu_samples: list[IMUSample] = []
        self._imu_index = 0

        self._opened = False
        self._exhausted = False

    def _load_frames(self) -> Iterator[FrameSample]:
        data_csv = self.cam0_dir / "data.csv"
        image_dir = self.cam0_dir / "data"

        with data_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # skip header
            for index, row in enumerate(reader):
                if len(row) < 2:
                    continue

                timestamp_ns = int(row[0])
                image_path = image_dir / row[1]
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")

                yield FrameSample(
                    frame_index=index,
                    timestamp_ns=timestamp_ns,
                    image_bgr=image_bgr,
                )

    def _load_imu_samples(self) -> list[IMUSample]:
        imu_samples: list[IMUSample] = []
        data_csv = self.imu0_dir / "data.csv"

        with data_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 7:
                    continue
                imu_samples.append(
                    IMUSample(
                        timestamp_ns=int(row[0]),
                        angular_velocity_rad_s=(
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                        ),
                        linear_acceleration_m_s2=(
                            float(row[4]),
                            float(row[5]),
                            float(row[6]),
                        ),
                    )
                )

        return imu_samples

    @staticmethod
    def _mean_imu(imu_window: list[IMUSample]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not imu_window:
            zeros = (0.0, 0.0, 0.0)
            return zeros, zeros

        gyro = np.array([sample.angular_velocity_rad_s for sample in imu_window], dtype=np.float64)
        accel = np.array([sample.linear_acceleration_m_s2 for sample in imu_window], dtype=np.float64)

        mean_gyro = tuple(float(x) for x in gyro.mean(axis=0))
        mean_accel = tuple(float(x) for x in accel.mean(axis=0))
        return mean_gyro, mean_accel

    def _create_sensor_packet(self, frame: FrameSample, imu_window: list[IMUSample]) -> SensorPacket:
        mean_gyro, mean_accel = self._mean_imu(imu_window)
        return SensorPacket(
            timestamp_ns=frame.timestamp_ns,
            frame_index=frame.frame_index,
            image_bgr=frame.image_bgr,
            angular_velocity_rad_s=mean_gyro,
            linear_acceleration_m_s2=mean_accel,
        )

    def open(self):
        if self._opened:
            return

        if not self.monocular:
            raise NotImplementedError("This simple stream currently supports monocular cam0 only.")

        if not (self.cam0_dir / "data.csv").exists():
            raise FileNotFoundError(f"Missing cam0 data.csv under {self.cam0_dir}")
        if self.inertial and not (self.imu0_dir / "data.csv").exists():
            raise FileNotFoundError(f"Missing imu0 data.csv under {self.imu0_dir}")

        self._frame_iter = self._load_frames()
        self._imu_samples = self._load_imu_samples() if self.inertial else []
        self._imu_index = 0

        self._current_frame = next(self._frame_iter, None)
        self._next_frame = next(self._frame_iter, None)
        self._exhausted = self._current_frame is None
        self._opened = True

    def read_next(self):
        if not self._opened:
            raise RuntimeError("EurocInputStream is not opened. Call open() first.")
        if self._exhausted or self._current_frame is None:
            return None

        frame = self._current_frame
        interval_end = self._next_frame.timestamp_ns if self._next_frame is not None else float("inf")

        imu_window: list[IMUSample] = []
        while self._imu_index < len(self._imu_samples):
            sample = self._imu_samples[self._imu_index]
            if sample.timestamp_ns >= interval_end:
                break
            if sample.timestamp_ns >= frame.timestamp_ns:
                imu_window.append(sample)
            self._imu_index += 1

        packet = self._create_sensor_packet(frame, imu_window)

        self._current_frame = self._next_frame
        self._next_frame = next(self._frame_iter, None) if self._current_frame is not None else None
        if self._current_frame is None:
            self._exhausted = True

        return packet

    def close(self):
        self._frame_iter = None
        self._current_frame = None
        self._next_frame = None
        self._imu_samples = []
        self._imu_index = 0
        self._opened = False
        self._exhausted = True

    def is_exhausted(self):
        return self._exhausted


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    stream = EurocInputStream(
        dataset_root=repo_root / "dataset" / "dataset-corridor1_512_16",
        monocular=True,
        inertial=True,
    )
    stream.open()

    for _ in range(5):
        packet = stream.read_next()
        if packet is None:
            break
        print(
            f"frame={packet.frame_index} ts={packet.timestamp_ns} "
            f"gyro={packet.angular_velocity_rad_s} accel={packet.linear_acceleration_m_s2}"
        )

    stream.close()
