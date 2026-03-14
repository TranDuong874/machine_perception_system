from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_ENV_FILE = REPO_ROOT / "config" / "server.env"
DEFAULT_SERVER_ENV_EXAMPLE_FILE = REPO_ROOT / "config" / "server.env.example"
DEFAULT_CAMERA_CALIBRATION_FILE = REPO_ROOT / "dataset" / "dataset-corridor1_512_16" / "dso" / "camchain.yaml"


@dataclass(frozen=True)
class ServerConfig:
    bind_host: str
    bind_port: int
    grpc_workers: int
    ingress_queue_size: int
    enqueue_timeout_s: float
    log_every_n_packets: int
    frame_store_max_packets: int
    depth_task_queue_size: int
    memory_task_queue_size: int
    depth_select_every_n_frames: int
    depth_select_translation_m: float
    depth_select_rotation_deg: float
    memory_select_every_n_frames: int
    memory_select_translation_m: float
    memory_select_rotation_deg: float
    memory_schedule_on_detection_change: bool
    enable_preview_gui: bool
    preview_frame_delay_ms: int
    headless_preview_path: Path
    enable_depth_estimation: bool
    depth_model_id: str
    depth_device: str
    depth_process_res: int
    enable_rectification: bool
    camera_calibration_file: Path | None
    camera_calibration_camera_key: str
    camera_rectification_balance: float
    enable_mapping: bool
    map_id: str
    map_resolution_m: float
    map_width: int
    map_height: int
    map_point_stride: int
    map_max_points_per_frame: int
    map_min_depth_m: float
    map_max_depth_m: float
    map_confidence_percentile: float
    map_min_confidence: float
    map_min_height_offset_m: float
    map_max_height_offset_m: float
    map_free_space_decrement: float
    map_occupied_increment: float
    storage_db_path: Path
    storage_artifact_dir: Path
    storage_save_debug_artifacts: bool
    storage_save_map_every_n_updates: int
    enable_memory_index: bool
    memory_index_dir: Path
    memory_model_name: str
    memory_pretrained: str
    memory_device: str
    memory_batch_size: int
    memory_save_every_n_updates: int
    memory_visual_weight: float
    memory_metadata_weight: float
    enable_vlm_api: bool
    vlm_bind_host: str
    vlm_bind_port: int
    groq_api_key: str | None
    groq_base_url: str
    groq_text_model: str
    groq_vision_model: str
    groq_timeout_s: float
    video_rag_top_k: int
    video_rag_max_images: int


def load_server_config() -> ServerConfig:
    env_file = _resolve_env_file(
        Path(os.environ.get("MPS_SERVER_ENV_FILE", str(DEFAULT_SERVER_ENV_FILE))),
        fallback=DEFAULT_SERVER_ENV_EXAMPLE_FILE,
    )
    _load_env_file(env_file)
    calibration_file = _resolve_calibration_file()
    return ServerConfig(
        bind_host=os.environ.get("MPS_SERVER_BIND_HOST", "0.0.0.0"),
        bind_port=int(os.environ.get("MPS_SERVER_PORT", "19100")),
        grpc_workers=int(os.environ.get("MPS_SERVER_GRPC_WORKERS", "8")),
        ingress_queue_size=int(os.environ.get("MPS_SERVER_INGRESS_QUEUE_SIZE", "256")),
        enqueue_timeout_s=float(os.environ.get("MPS_SERVER_ENQUEUE_TIMEOUT_S", "0.02")),
        log_every_n_packets=max(1, int(os.environ.get("MPS_SERVER_LOG_EVERY_N", "20"))),
        frame_store_max_packets=max(
            32, int(os.environ.get("MPS_SERVER_FRAME_STORE_MAX_PACKETS", "512"))
        ),
        depth_task_queue_size=max(
            1, int(os.environ.get("MPS_SERVER_DEPTH_TASK_QUEUE_SIZE", "16"))
        ),
        memory_task_queue_size=max(
            1, int(os.environ.get("MPS_SERVER_MEMORY_TASK_QUEUE_SIZE", "32"))
        ),
        depth_select_every_n_frames=max(
            1, int(os.environ.get("MPS_SERVER_DEPTH_SELECT_EVERY_N_FRAMES", "4"))
        ),
        depth_select_translation_m=max(
            0.0, float(os.environ.get("MPS_SERVER_DEPTH_SELECT_TRANSLATION_M", "0.15"))
        ),
        depth_select_rotation_deg=max(
            0.0, float(os.environ.get("MPS_SERVER_DEPTH_SELECT_ROTATION_DEG", "8.0"))
        ),
        memory_select_every_n_frames=max(
            1, int(os.environ.get("MPS_SERVER_MEMORY_SELECT_EVERY_N_FRAMES", "8"))
        ),
        memory_select_translation_m=max(
            0.0, float(os.environ.get("MPS_SERVER_MEMORY_SELECT_TRANSLATION_M", "0.25"))
        ),
        memory_select_rotation_deg=max(
            0.0, float(os.environ.get("MPS_SERVER_MEMORY_SELECT_ROTATION_DEG", "12.0"))
        ),
        memory_schedule_on_detection_change=_parse_env_bool(
            "MPS_SERVER_MEMORY_SCHEDULE_ON_DETECTION_CHANGE"
        ) is not False,
        enable_preview_gui=_resolve_preview_gui_enabled(),
        preview_frame_delay_ms=int(os.environ.get("MPS_SERVER_PREVIEW_DELAY_MS", "1")),
        headless_preview_path=Path(
            os.environ.get("MPS_SERVER_HEADLESS_PREVIEW_PATH", "/tmp/mps_server_preview.jpg")
        ),
        enable_depth_estimation=_parse_env_bool("MPS_SERVER_ENABLE_DEPTH") or False,
        depth_model_id=os.environ.get("MPS_SERVER_DEPTH_MODEL_ID", "depth-anything/DA3-BASE"),
        depth_device=os.environ.get("MPS_SERVER_DEPTH_DEVICE", "auto").strip().lower(),
        depth_process_res=max(224, int(os.environ.get("MPS_SERVER_DEPTH_PROCESS_RES", "504"))),
        enable_rectification=_parse_env_bool("MPS_SERVER_ENABLE_RECTIFICATION") is not False,
        camera_calibration_file=calibration_file,
        camera_calibration_camera_key=os.environ.get("MPS_SERVER_CAMERA_KEY", "cam0").strip(),
        camera_rectification_balance=float(os.environ.get("MPS_SERVER_RECTIFICATION_BALANCE", "0.0")),
        enable_mapping=_parse_env_bool("MPS_SERVER_ENABLE_MAPPING") is not False,
        map_id=os.environ.get("MPS_SERVER_MAP_ID", "main"),
        map_resolution_m=max(0.02, float(os.environ.get("MPS_SERVER_MAP_RESOLUTION_M", "0.10"))),
        map_width=max(32, int(os.environ.get("MPS_SERVER_MAP_WIDTH", "400"))),
        map_height=max(32, int(os.environ.get("MPS_SERVER_MAP_HEIGHT", "400"))),
        map_point_stride=max(1, int(os.environ.get("MPS_SERVER_MAP_POINT_STRIDE", "8"))),
        map_max_points_per_frame=max(
            128, int(os.environ.get("MPS_SERVER_MAP_MAX_POINTS_PER_FRAME", "2500"))
        ),
        map_min_depth_m=max(0.01, float(os.environ.get("MPS_SERVER_MAP_MIN_DEPTH_M", "0.3"))),
        map_max_depth_m=max(0.1, float(os.environ.get("MPS_SERVER_MAP_MAX_DEPTH_M", "8.0"))),
        map_confidence_percentile=min(
            99.0, max(1.0, float(os.environ.get("MPS_SERVER_MAP_CONFIDENCE_PERCENTILE", "55.0")))
        ),
        map_min_confidence=float(os.environ.get("MPS_SERVER_MAP_MIN_CONFIDENCE", "1.0")),
        map_min_height_offset_m=float(
            os.environ.get("MPS_SERVER_MAP_MIN_HEIGHT_OFFSET_M", "-1.5")
        ),
        map_max_height_offset_m=float(
            os.environ.get("MPS_SERVER_MAP_MAX_HEIGHT_OFFSET_M", "0.6")
        ),
        map_free_space_decrement=max(
            0.0, float(os.environ.get("MPS_SERVER_MAP_FREE_SPACE_DECREMENT", "0.12"))
        ),
        map_occupied_increment=max(
            0.0, float(os.environ.get("MPS_SERVER_MAP_OCCUPIED_INCREMENT", "0.65"))
        ),
        storage_db_path=_resolve_repo_path(
            os.environ.get("MPS_SERVER_STORAGE_DB_PATH", "/tmp/mps_server_runtime.sqlite3")
        ),
        storage_artifact_dir=_resolve_repo_path(
            os.environ.get("MPS_SERVER_STORAGE_ARTIFACT_DIR", "/tmp/mps_server_artifacts")
        ),
        storage_save_debug_artifacts=_parse_env_bool("MPS_SERVER_SAVE_DEBUG_ARTIFACTS") or False,
        storage_save_map_every_n_updates=max(
            1, int(os.environ.get("MPS_SERVER_SAVE_MAP_EVERY_N_UPDATES", "5"))
        ),
        enable_memory_index=_parse_env_bool("MPS_SERVER_ENABLE_MEMORY_INDEX") is not False,
        memory_index_dir=_resolve_repo_path(
            os.environ.get("MPS_SERVER_MEMORY_INDEX_DIR", "/tmp/mps_server_artifacts/memory_index")
        ),
        memory_model_name=os.environ.get("MPS_SERVER_MEMORY_MODEL_NAME", "ViT-B-32"),
        memory_pretrained=os.environ.get(
            "MPS_SERVER_MEMORY_PRETRAINED",
            "laion2b_s34b_b79k",
        ),
        memory_device=os.environ.get("MPS_SERVER_MEMORY_DEVICE", "auto").strip().lower(),
        memory_batch_size=max(1, int(os.environ.get("MPS_SERVER_MEMORY_BATCH_SIZE", "1"))),
        memory_save_every_n_updates=max(
            1, int(os.environ.get("MPS_SERVER_MEMORY_SAVE_EVERY_N_UPDATES", "10"))
        ),
        memory_visual_weight=max(
            0.0, float(os.environ.get("MPS_SERVER_MEMORY_VISUAL_WEIGHT", "0.7"))
        ),
        memory_metadata_weight=max(
            0.0, float(os.environ.get("MPS_SERVER_MEMORY_METADATA_WEIGHT", "0.3"))
        ),
        enable_vlm_api=_parse_env_bool("MPS_SERVER_ENABLE_VLM_API") is not False,
        vlm_bind_host=os.environ.get("MPS_SERVER_VLM_BIND_HOST", "127.0.0.1"),
        vlm_bind_port=int(os.environ.get("MPS_SERVER_VLM_PORT", "19110")),
        groq_api_key=_resolve_groq_api_key(),
        groq_base_url=os.environ.get("MPS_GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        groq_text_model=os.environ.get(
            "MPS_GROQ_TEXT_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        groq_vision_model=os.environ.get(
            "MPS_GROQ_VISION_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        groq_timeout_s=max(1.0, float(os.environ.get("MPS_GROQ_TIMEOUT_S", "60.0"))),
        video_rag_top_k=max(1, int(os.environ.get("MPS_SERVER_VIDEO_RAG_TOP_K", "3"))),
        video_rag_max_images=max(
            1, int(os.environ.get("MPS_SERVER_VIDEO_RAG_MAX_IMAGES", "3"))
        ),
    )


def _resolve_preview_gui_enabled() -> bool:
    env_override = _parse_env_bool("MPS_SERVER_ENABLE_GUI")
    has_display = _has_gui_display()
    if env_override is None:
        return has_display
    if env_override and not has_display:
        return False
    return env_override


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


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    existing_keys = set(os.environ)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_matching_quotes(value.strip())
        if key in existing_keys:
            continue
        os.environ[key] = value


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _resolve_repo_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _resolve_env_file(path: Path, *, fallback: Path) -> Path:
    if path.exists():
        return path
    return fallback


def _resolve_calibration_file() -> Path | None:
    raw_value = os.environ.get("MPS_SERVER_CAMERA_CALIBRATION_FILE")
    if raw_value is not None:
        value = raw_value.strip()
        if value == "":
            return None
        return _resolve_repo_path(value)

    dataset_root = os.environ.get("MPS_DATASET_ROOT")
    if dataset_root:
        candidate = _resolve_repo_path(dataset_root) / "dso" / "camchain.yaml"
        if candidate.exists():
            return candidate

    return DEFAULT_CAMERA_CALIBRATION_FILE if DEFAULT_CAMERA_CALIBRATION_FILE.exists() else None


def _resolve_groq_api_key() -> str | None:
    for name in ("MPS_GROQ_API_KEY", "GROQ_API_KEY"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None
