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
    close_bridge_clients,
    collect_server_perception_packet,
    get_latest_topdown_map_state,
    stream_server_perception_packet,
)
from pipeline.TopDownMapCache import TopDownMapState, world_xz_to_pixel
from user_interface import UserViewRenderer


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


def _resolve_user_gui_enabled() -> bool:
    env_override = _parse_env_bool("MPS_ENABLE_USER_GUI")
    has_display = _has_gui_display()
    if env_override is None:
        return GUI_ENABLED
    if env_override and not has_display:
        return False
    return env_override


USER_GUI_ENABLED = _resolve_user_gui_enabled()
ANY_GUI_ENABLED = GUI_ENABLED or USER_GUI_ENABLED


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


def _build_user_overlay_draw_fn(
    map_state: TopDownMapState | None,
    camera_translation_xyz: tuple[float, float, float] | None,
):
    if map_state is None or map_state.occupancy_bgr is None:
        return None

    def _draw(frame_bgr):
        panel_max_w = min(240, max(120, frame_bgr.shape[1] // 3))
        scale = panel_max_w / map_state.occupancy_bgr.shape[1]
        panel_w = panel_max_w
        panel_h = max(80, int(round(map_state.occupancy_bgr.shape[0] * scale)))
        panel = cv2.resize(map_state.occupancy_bgr, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

        x0 = frame_bgr.shape[1] - panel_w - 12
        y0 = 48
        y1 = min(frame_bgr.shape[0], y0 + panel_h)
        x1 = min(frame_bgr.shape[1], x0 + panel_w)
        if x0 < 0 or y0 < 0 or y1 <= y0 or x1 <= x0:
            return

        frame_bgr[y0:y1, x0:x1] = panel[: y1 - y0, : x1 - x0]
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (60, 220, 255), 1)
        cv2.putText(
            frame_bgr,
            "Server Map",
            (x0 + 6, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if camera_translation_xyz is None:
            return
        point = world_xz_to_pixel(
            map_state,
            world_x_m=float(camera_translation_xyz[0]),
            world_z_m=float(camera_translation_xyz[2]),
        )
        if point is None:
            return
        col, row = point
        px = x0 + int(round((col / max(1, map_state.width - 1)) * (panel_w - 1)))
        py = y0 + int(round((row / max(1, map_state.height - 1)) * (panel_h - 1)))
        cv2.circle(frame_bgr, (px, py), 4, (0, 0, 255), -1)
        cv2.circle(frame_bgr, (px, py), 7, (255, 255, 255), 1)

    return _draw


def main() -> None:
    dataset_root = _resolve_dataset_root()
    yolo_model_path = _resolve_yolo_model_path()
    replay_speed = float(os.environ.get("MPS_REPLAY_SPEED", "1.0"))
    max_frames = _env_int("MPS_MAX_FRAMES")
    max_pending_imu = int(os.environ.get("MPS_MAX_PENDING_IMU", "4000"))
    slam_host = os.environ.get("MPS_SLAM_HOST", "127.0.0.1")
    slam_port = int(os.environ.get("MPS_SLAM_PORT", "19090"))
    slam_timeout_s = float(os.environ.get("MPS_SLAM_TIMEOUT_S", "1.5"))
    server_transport = os.environ.get("MPS_SERVER_TRANSPORT", "grpc").lower()
    server_host = os.environ.get("MPS_SERVER_HOST", "127.0.0.1")
    server_port = int(os.environ.get("MPS_SERVER_PORT", "19100"))

    source = EurocInputSource(dataset_root=dataset_root, monocular=True, inertial=True, replay_speed=replay_speed)
    runner = StreamRunner(source=source, queue_size=256, poll_timeout_s=0.1, drop_oldest_when_full=False)
    ingress = LocalProcessingIngress(max_pending_imu=max_pending_imu)
    detection_service = DetectionService(model_path=yolo_model_path, device="cpu")
    slam_service = SlamService(host=slam_host, port=slam_port, result_timeout_s=slam_timeout_s)
    user_view_renderer = UserViewRenderer() if USER_GUI_ENABLED else None
    detection_service.start()
    slam_service.start()
    runner.start()

    print("Producer: StreamRunner thread")
    print(f"Dataset: {dataset_root}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Replay speed: {replay_speed}")
    print(f"SLAM adapter: {slam_host}:{slam_port} timeout_s={slam_timeout_s}")
    print(f"Server transport: {server_transport} target={server_host}:{server_port}")
    print(f"Ingress max pending IMU: {max_pending_imu}")
    print(f"GUI enabled (local processing): {GUI_ENABLED}")
    print(f"GUI enabled (user view): {USER_GUI_ENABLED}")
    if max_frames is not None:
        print(f"Max frames: {max_frames}")
    print("Stages: StreamRunner -> UserView + LocalProcessingIngress -> DetectionService + SlamService -> ServerPacketBridge -> server(topdown-map) + render")
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

            if user_view_renderer is not None:
                map_state = get_latest_topdown_map_state()
                orb = server_packet.orb_tracking
                tracking_state = orb.tracking_state if orb is not None and orb.tracking_state else "NO_TRACK"
                camera_translation = orb.camera_translation_xyz if orb is not None else None
                overlay_lines = [
                    f"ts={local_packet.timestamp_ns}",
                    f"tracking={tracking_state}",
                ]
                if map_state is not None:
                    overlay_lines.append(
                        f"map={map_state.map_id} rev={map_state.revision} backend={map_state.backend}"
                    )
                    overlay_lines.append(
                        f"server_pose_valid={map_state.pose_valid} map_ts={map_state.timestamp_ns}"
                    )
                user_view_renderer.render(
                    local_packet.image_bgr,
                    overlay_lines=overlay_lines,
                    overlay_draw_fn=_build_user_overlay_draw_fn(map_state, camera_translation),
                )

            frames_processed += 1

            if max_frames is not None and frames_processed >= max_frames:
                break

            if ANY_GUI_ENABLED:
                key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
                if key == ord("q"):
                    break
    finally:
        runner.stop()
        detection_service.stop()
        slam_service.stop()
        close_bridge_clients()
        if user_view_renderer is not None:
            user_view_renderer.close()
        if ANY_GUI_ENABLED:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
