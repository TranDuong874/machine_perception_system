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
    close_visualizers,
    collect_server_perception_packet,
    stream_server_perception_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POLL_TIMEOUT_S = 0.5
FRAME_DELAY_MS = 1
EXTRACT_ROOT = Path("/tmp/machine_perception_system_datasets")
DEFAULT_DATASET_NAMES = (
    "dataset-corridor4_512_16",
    "dataset-corridor1_512_16",
)


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


def _parse_env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw}")


def _resolve_gui_enabled() -> bool:
    env_override = _parse_env_bool("MPS_ENABLE_GUI")
    has_display = _has_gui_display()
    if env_override is None:
        return has_display
    if env_override and not has_display:
        return False
    return env_override


GUI_ENABLED = _resolve_gui_enabled()
os.environ["MPS_ENABLE_GUI"] = "1" if GUI_ENABLED else "0"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _dataset_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    dataset_root_env = os.environ.get("MPS_DATASET_ROOT")
    if dataset_root_env:
        candidates.append(Path(dataset_root_env).expanduser())

    for base in (REPO_ROOT / "dataset", REPO_ROOT.parent / "dataset"):
        for dataset_name in DEFAULT_DATASET_NAMES:
            candidates.append(base / dataset_name)
    return candidates


def _dataset_tar_candidates() -> list[Path]:
    candidates: list[Path] = []
    dataset_tar_env = os.environ.get("MPS_DATASET_TAR")
    if dataset_tar_env:
        candidates.append(Path(dataset_tar_env).expanduser())

    for base in (REPO_ROOT / "dataset", REPO_ROOT.parent / "dataset"):
        for dataset_name in DEFAULT_DATASET_NAMES:
            candidates.append(base / f"{dataset_name}.tar")
    return candidates


def _resolve_dataset_root() -> Path:
    for candidate in _dataset_root_candidates():
        if candidate.exists():
            return candidate

    for candidate in _dataset_tar_candidates():
        if not candidate.exists():
            continue
        extract_root = EXTRACT_ROOT / candidate.stem
        if not extract_root.exists():
            EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(candidate) as archive:
                archive.extractall(EXTRACT_ROOT)
        return extract_root

    searched_roots = ", ".join(str(path) for path in _dataset_root_candidates())
    searched_tars = ", ".join(str(path) for path in _dataset_tar_candidates())
    raise FileNotFoundError(
        "Dataset not found. "
        f"Checked dataset roots: [{searched_roots}]. "
        f"Checked dataset tar files: [{searched_tars}]."
    )


def _resolve_yolo_model_path() -> Path:
    model_path_env = os.environ.get("MPS_YOLO_MODEL_PATH")
    if model_path_env:
        candidate = Path(model_path_env).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"YOLO model from MPS_YOLO_MODEL_PATH not found: {candidate}"
        )

    candidates = (
        REPO_ROOT / "models" / "yolo26n.pt",
        REPO_ROOT.parent / "models" / "yolo26n.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "YOLO model not found. "
        f"Checked: {[str(path) for path in candidates]}. "
        "Set MPS_YOLO_MODEL_PATH to override."
    )


def main() -> None:
    dataset_root = _resolve_dataset_root()
    yolo_model_path = _resolve_yolo_model_path()
    replay_speed = float(os.environ.get("MPS_REPLAY_SPEED", "1.0"))
    max_frames = _env_int("MPS_MAX_FRAMES")
    max_pending_imu = int(os.environ.get("MPS_MAX_PENDING_IMU", "4000"))
    slam_host = os.environ.get("MPS_SLAM_HOST", "127.0.0.1")
    slam_port = int(os.environ.get("MPS_SLAM_PORT", "19090"))
    slam_timeout_s = float(os.environ.get("MPS_SLAM_TIMEOUT_S", "1.5"))

    source = EurocInputSource(dataset_root=dataset_root, monocular=True, inertial=True, replay_speed=replay_speed)
    runner = StreamRunner(source=source, queue_size=256, poll_timeout_s=0.1, drop_oldest_when_full=False)
    ingress = LocalProcessingIngress(max_pending_imu=max_pending_imu)
    detection_service = DetectionService(model_path=yolo_model_path, device="cpu")
    slam_service = SlamService(host=slam_host, port=slam_port, result_timeout_s=slam_timeout_s)
    detection_service.start()
    slam_service.start()
    runner.start()

    print("Producer: StreamRunner thread")
    print(f"Dataset: {dataset_root}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Replay speed: {replay_speed}")
    print(f"SLAM adapter: {slam_host}:{slam_port} timeout_s={slam_timeout_s}")
    print(f"Ingress max pending IMU: {max_pending_imu}")
    print(f"GUI enabled: {GUI_ENABLED}")
    if max_frames is not None:
        print(f"Max frames: {max_frames}")
    print("Stages: LocalProcessingIngress -> DetectionService + SlamService -> ServerPacketBridge -> render")
    print("Press 'q' to quit.")

    frames_processed = 0
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
                slam_timeout_s=slam_timeout_s,
            )
            stream_server_perception_packet(server_packet)
            frames_processed += 1

            if max_frames is not None and frames_processed >= max_frames:
                break

            if GUI_ENABLED:
                key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
                if key == ord("q"):
                    break
    finally:
        runner.stop()
        detection_service.stop()
        slam_service.stop()
        close_visualizers()
        if GUI_ENABLED:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
