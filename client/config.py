from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_ROOT = Path("/tmp/machine_perception_system_datasets")
DEFAULT_DATASET_NAMES = (
    "dataset-corridor4_512_16",
    "dataset-corridor1_512_16",
)
DEFAULT_CLIENT_ENV_FILE = REPO_ROOT / "config" / "client.env"
DEFAULT_CLIENT_ENV_EXAMPLE_FILE = REPO_ROOT / "config" / "client.env.example"


@dataclass(frozen=True)
class ClientConfig:
    dataset_root: Path
    yolo_model_path: Path
    replay_speed: float
    max_frames: int | None
    max_pending_imu: int
    slam_host: str
    slam_port: int
    slam_timeout_s: float
    server_transport: str
    server_host: str
    server_port: int
    vlm_host: str
    vlm_port: int
    gui_enabled: bool
    user_gui_enabled: bool
    poll_timeout_s: float = 0.5
    frame_delay_ms: int = 1
    vlm_timeout_s: float = 90.0
    chat_top_k: int = 3
    chat_default_mode: str = "current"
    runner_queue_size: int = 256
    runner_poll_timeout_s: float = 0.1
    runner_drop_oldest_when_full: bool = False
    yolo_device: str = "cpu"
    yolo_timeout_s: float = 2.0

    @property
    def any_gui_enabled(self) -> bool:
        return self.gui_enabled or self.user_gui_enabled

    @property
    def slam_target(self) -> str:
        return f"{self.slam_host}:{self.slam_port}"

    @property
    def server_target(self) -> str:
        return f"{self.server_host}:{self.server_port}"

    @property
    def vlm_target(self) -> str:
        return f"{self.vlm_host}:{self.vlm_port}"


def load_client_config() -> ClientConfig:
    env_file = _resolve_env_file(
        Path(os.environ.get("MPS_CLIENT_ENV_FILE", str(DEFAULT_CLIENT_ENV_FILE))),
        fallback=DEFAULT_CLIENT_ENV_EXAMPLE_FILE,
    )
    _load_env_file(env_file)
    gui_enabled = _resolve_gui_enabled()
    user_gui_enabled = _resolve_user_gui_enabled(gui_enabled)

    os.environ["MPS_ENABLE_GUI"] = "1" if gui_enabled else "0"
    os.environ["MPS_ENABLE_USER_GUI"] = "1" if user_gui_enabled else "0"

    return ClientConfig(
        dataset_root=_resolve_dataset_root(),
        yolo_model_path=_resolve_yolo_model_path(),
        replay_speed=float(os.environ.get("MPS_REPLAY_SPEED", "1.0")),
        max_frames=_env_int("MPS_MAX_FRAMES"),
        max_pending_imu=int(os.environ.get("MPS_MAX_PENDING_IMU", "4000")),
        slam_host=os.environ.get("MPS_SLAM_HOST", "127.0.0.1"),
        slam_port=int(os.environ.get("MPS_SLAM_PORT", "19090")),
        slam_timeout_s=float(os.environ.get("MPS_SLAM_TIMEOUT_S", "1.5")),
        server_transport=os.environ.get("MPS_SERVER_TRANSPORT", "grpc").strip().lower(),
        server_host=os.environ.get("MPS_SERVER_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("MPS_SERVER_PORT", "19100")),
        vlm_host=os.environ.get(
            "MPS_VLM_HOST",
            os.environ.get(
                "MPS_SERVER_VLM_BIND_HOST",
                os.environ.get("MPS_SERVER_HOST", "127.0.0.1"),
            ),
        ),
        vlm_port=int(
            os.environ.get(
                "MPS_VLM_PORT",
                os.environ.get("MPS_SERVER_VLM_PORT", "19110"),
            )
        ),
        gui_enabled=gui_enabled,
        user_gui_enabled=user_gui_enabled,
        vlm_timeout_s=max(1.0, float(os.environ.get("MPS_VLM_TIMEOUT_S", "90.0"))),
        chat_top_k=max(1, int(os.environ.get("MPS_CHAT_TOP_K", "3"))),
        chat_default_mode=_resolve_chat_default_mode(),
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


def _resolve_user_gui_enabled(gui_enabled: bool) -> bool:
    env_override = _parse_env_bool("MPS_ENABLE_USER_GUI")
    has_display = _has_gui_display()
    if env_override is None:
        return gui_enabled
    if env_override and not has_display:
        return False
    return env_override


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _resolve_chat_default_mode() -> str:
    mode = os.environ.get("MPS_CHAT_DEFAULT_MODE", "current").strip().lower()
    if mode not in {"current", "rag"}:
        raise ValueError(f"Invalid MPS_CHAT_DEFAULT_MODE: {mode}")
    return mode


def _dataset_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    dataset_root_env = os.environ.get("MPS_DATASET_ROOT")
    if dataset_root_env:
        candidates.append(_resolve_repo_path(dataset_root_env))

    for base in (REPO_ROOT / "dataset", REPO_ROOT.parent / "dataset"):
        for dataset_name in DEFAULT_DATASET_NAMES:
            candidates.append(base / dataset_name)
    return candidates


def _dataset_tar_candidates() -> list[Path]:
    candidates: list[Path] = []
    dataset_tar_env = os.environ.get("MPS_DATASET_TAR")
    if dataset_tar_env:
        candidates.append(_resolve_repo_path(dataset_tar_env))

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
        candidate = _resolve_repo_path(model_path_env)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"YOLO model from MPS_YOLO_MODEL_PATH not found: {candidate}")

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
