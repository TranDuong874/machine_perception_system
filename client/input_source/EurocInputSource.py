from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Iterator

import cv2

from .InputSource import InputSource
from schema.SensorSchema import FrameSample, IMUSample, InputSample


class EurocInputSource(InputSource[InputSample]):
    def __init__(self, dataset_root, monocular=True, inertial=True, replay_speed: float = 1.0):
        self.dataset_root = Path(dataset_root)
        self.monocular = monocular
        self.inertial = inertial
        self.replay_speed = replay_speed

        self.cam0_dir = self.dataset_root / "mav0" / "cam0"
        self.imu0_dir = self.dataset_root / "mav0" / "imu0"

        self._frame_iter: Iterator[FrameSample] | None = None
        self._next_frame: FrameSample | None = None
        self._next_imu: IMUSample | None = None

        self._imu_samples: list[IMUSample] = []
        self._imu_index = 0

        self._opened = False
        self._exhausted = False
        self._wall_start_time_s: float | None = None
        self._stream_start_timestamp_ns: int | None = None

    def _load_frames(self) -> Iterator[FrameSample]:
        data_csv = self.cam0_dir / "data.csv"
        image_dir = self.cam0_dir / "data"

        with data_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue

                timestamp_ns = int(row[0])
                image_path = image_dir / row[1]
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")

                yield FrameSample(
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

    def open(self) -> None:
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

        self._next_frame = next(self._frame_iter, None)
        self._next_imu = self._imu_samples[self._imu_index] if self._imu_samples else None
        self._exhausted = self._next_frame is None and self._next_imu is None
        self._wall_start_time_s = None
        self._stream_start_timestamp_ns = None
        self._opened = True

    def read_next(self) -> InputSample | None:
        if not self._opened:
            raise RuntimeError("EurocInputSource is not opened. Call open() first.")
        if self._exhausted:
            return None

        sample = self._pop_next_sample()
        if sample is None:
            self._exhausted = True
            return None

        self._pace_replay(sample.timestamp_ns)
        self._exhausted = self._next_frame is None and self._next_imu is None
        return sample

    def _pop_next_sample(self) -> InputSample | None:
        if self._next_frame is None and self._next_imu is None:
            return None

        if self._next_frame is None:
            return self._pop_imu()

        if self._next_imu is None:
            return self._pop_frame()

        if self._next_imu.timestamp_ns <= self._next_frame.timestamp_ns:
            return self._pop_imu()
        return self._pop_frame()

    def _pop_frame(self) -> FrameSample | None:
        frame = self._next_frame
        if frame is None:
            return None
        self._next_frame = next(self._frame_iter, None) if self._frame_iter is not None else None
        return frame

    def _pop_imu(self) -> IMUSample | None:
        if self._next_imu is None:
            return None

        sample = self._next_imu
        self._imu_index += 1
        self._next_imu = self._imu_samples[self._imu_index] if self._imu_index < len(self._imu_samples) else None
        return sample

    def _pace_replay(self, sample_timestamp_ns: int) -> None:
        if self.replay_speed <= 0:
            return

        if self._wall_start_time_s is None or self._stream_start_timestamp_ns is None:
            self._wall_start_time_s = time.perf_counter()
            self._stream_start_timestamp_ns = sample_timestamp_ns
            return

        elapsed_dataset_s = (sample_timestamp_ns - self._stream_start_timestamp_ns) / 1_000_000_000.0
        target_time_s = self._wall_start_time_s + (elapsed_dataset_s / self.replay_speed)
        now_s = time.perf_counter()
        if now_s < target_time_s:
            time.sleep(target_time_s - now_s)

    def close(self) -> None:
        self._frame_iter = None
        self._next_frame = None
        self._next_imu = None
        self._imu_samples = []
        self._imu_index = 0
        self._wall_start_time_s = None
        self._stream_start_timestamp_ns = None
        self._opened = False
        self._exhausted = True

    def is_exhausted(self) -> bool:
        return self._exhausted


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    source = EurocInputSource(
        dataset_root=repo_root / "dataset" / "dataset-corridor4_512_16",
        monocular=True,
        inertial=True,
        replay_speed=1.0,
    )
    source.open()

    for _ in range(5):
        sample = source.read_next()
        if sample is None:
            break
        print(f"{type(sample).__name__}: ts={sample.timestamp_ns}")

    source.close()
