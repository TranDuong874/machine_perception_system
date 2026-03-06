from __future__ import annotations

from pathlib import Path
import tarfile

import cv2
import numpy as np

from input_source.EurocInputSource import EurocInputSource
from input_source.StreamRunner import StreamRunner
from pipeline.LocalProcessingIngress import LocalProcessingIngress
from schema.SensorSchema import LocalSensorPacket


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset" / "dataset-corridor4_512_16"
DATASET_TAR = REPO_ROOT / "dataset" / "dataset-corridor4_512_16.tar"
WINDOW_NAME = "Input Stream Preview"
POLL_TIMEOUT_S = 0.5
FRAME_DELAY_MS = 1
EXTRACT_ROOT = Path("/tmp/machine_perception_system_datasets")


def _overlay_text(image_bgr, lines: list[str]):
    out = image_bgr.copy()
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def _run_local_processing(packet: LocalSensorPacket) -> LocalSensorPacket:
    # Placeholder for ORB-SLAM / detection / tracking outputs within the local pipeline.
    return packet


def _mean_imu(packet: LocalSensorPacket) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not packet.imu_samples:
        zeros = (0.0, 0.0, 0.0)
        return zeros, zeros

    gyro = np.array([sample.angular_velocity_rad_s for sample in packet.imu_samples], dtype=np.float64)
    accel = np.array([sample.linear_acceleration_m_s2 for sample in packet.imu_samples], dtype=np.float64)
    mean_gyro = tuple(float(x) for x in gyro.mean(axis=0))
    mean_accel = tuple(float(x) for x in accel.mean(axis=0))
    return mean_gyro, mean_accel


def _resolve_dataset_root() -> Path:
    if DATASET_ROOT.exists():
        return DATASET_ROOT

    if not DATASET_TAR.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected either {DATASET_ROOT} or {DATASET_TAR}."
        )

    extract_root = EXTRACT_ROOT / DATASET_ROOT.name
    if not extract_root.exists():
        EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DATASET_TAR) as archive:
            archive.extractall(EXTRACT_ROOT)

    return extract_root


def main() -> None:
    dataset_root = _resolve_dataset_root()
    source = EurocInputSource(dataset_root=dataset_root, monocular=True, inertial=True)
    runner = StreamRunner(source=source, queue_size=8, poll_timeout_s=0.1, drop_oldest_when_full=True)
    ingress = LocalProcessingIngress()
    runner.start()

    print("Producer: StreamRunner thread")
    print(f"Dataset: {dataset_root}")
    print("Stages: LocalProcessingIngress -> local processing -> main render loop")
    print("Press 'q' to quit.")

    try:
        while True:
            sample = runner.read(timeout_s=POLL_TIMEOUT_S)
            if sample is None:
                break

            sensor_packet = ingress.push(sample)
            if sensor_packet is None:
                continue

            local_packet = _run_local_processing(sensor_packet)
            mean_gyro, mean_accel = _mean_imu(local_packet)

            lines = [
                f"timestamp_ns={local_packet.timestamp_ns}",
                f"imu_samples={len(local_packet.imu_samples)}",
                "gyro(rad/s)="
                f"({mean_gyro[0]:+.3f}, "
                f"{mean_gyro[1]:+.3f}, "
                f"{mean_gyro[2]:+.3f})",
                "accel(m/s^2)="
                f"({mean_accel[0]:+.3f}, "
                f"{mean_accel[1]:+.3f}, "
                f"{mean_accel[2]:+.3f})",
            ]
            frame = _overlay_text(local_packet.image_bgr, lines)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
            if key == ord("q"):
                break
    finally:
        runner.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
