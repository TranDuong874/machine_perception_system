from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tarfile

import cv2

from input_source.EurocInputSource import EurocInputSource
from input_source.StreamRunner import StreamRunner
from local_services import DetectionService, SlamService
from pipeline.LocalProcessingIngress import LocalProcessingIngress
from pipeline.ServerPacketBridge import (
    collect_server_perception_packet,
    stream_server_perception_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset" / "dataset-corridor4_512_16"
DATASET_TAR = REPO_ROOT / "dataset" / "dataset-corridor4_512_16.tar"
POLL_TIMEOUT_S = 0.5
FRAME_DELAY_MS = 1
EXTRACT_ROOT = Path("/tmp/machine_perception_system_datasets")
YOLO_MODEL_PATH = REPO_ROOT / "models" / "yolo26n.pt"


def _has_gui_display() -> bool:
    if not os.environ.get("DISPLAY"):
        return False
    try:
        probe = subprocess.run(
            ["xdpyinfo"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return probe.returncode == 0


HAS_DISPLAY = _has_gui_display()
os.environ["MPS_ENABLE_GUI"] = "1" if HAS_DISPLAY else "0"


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
    detection_service = DetectionService(model_path=YOLO_MODEL_PATH, device="cpu")
    slam_service = SlamService()
    detection_service.start()
    slam_service.start()
    runner.start()

    print("Producer: StreamRunner thread")
    print(f"Dataset: {dataset_root}")
    print("Stages: LocalProcessingIngress -> DetectionService + SlamService -> ServerPacketBridge -> render")
    print("Press 'q' to quit.")

    try:
        while True:
            sample = runner.read(timeout_s=POLL_TIMEOUT_S)
            if sample is None:
                break

            local_packet = ingress.push(sample)
            if local_packet is None:
                continue

            detection_service.submit(local_packet)
            slam_service.submit(local_packet)

            server_packet = collect_server_perception_packet(
                local_packet,
                detection_service=detection_service,
                slam_service=slam_service,
                yolo_timeout_s=2.0,
                slam_timeout_s=1.0,
            )
            stream_server_perception_packet(server_packet)

            if HAS_DISPLAY:
                key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
                if key == ord("q"):
                    break
    finally:
        runner.stop()
        detection_service.stop()
        slam_service.stop()
        if HAS_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
