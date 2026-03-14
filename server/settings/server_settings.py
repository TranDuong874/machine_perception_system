from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_ENV_FILE = REPO_ROOT / "config" / "server.env"


@dataclass(frozen=True)
class ServerConfig:
    bind_host: str
    bind_port: int
    grpc_workers: int
    ingress_queue_size: int
    enqueue_timeout_s: float
    log_every_n_packets: int
    enable_preview_gui: bool
    preview_frame_delay_ms: int
    headless_preview_path: Path
    enable_depth_estimation: bool
    depth_model_id: str
    depth_device: str
    depth_process_res: int


def load_server_config() -> ServerConfig:
    _load_env_file(Path(os.environ.get("MPS_SERVER_ENV_FILE", str(DEFAULT_SERVER_ENV_FILE))))
    return ServerConfig(
        bind_host=os.environ.get("MPS_SERVER_BIND_HOST", "0.0.0.0"),
        bind_port=int(os.environ.get("MPS_SERVER_PORT", "19100")),
        grpc_workers=int(os.environ.get("MPS_SERVER_GRPC_WORKERS", "8")),
        ingress_queue_size=int(os.environ.get("MPS_SERVER_INGRESS_QUEUE_SIZE", "256")),
        enqueue_timeout_s=float(os.environ.get("MPS_SERVER_ENQUEUE_TIMEOUT_S", "0.02")),
        log_every_n_packets=max(1, int(os.environ.get("MPS_SERVER_LOG_EVERY_N", "20"))),
        enable_preview_gui=_resolve_preview_gui_enabled(),
        preview_frame_delay_ms=int(os.environ.get("MPS_SERVER_PREVIEW_DELAY_MS", "1")),
        headless_preview_path=Path(
            os.environ.get("MPS_SERVER_HEADLESS_PREVIEW_PATH", "/tmp/mps_server_preview.jpg")
        ),
        enable_depth_estimation=_parse_env_bool("MPS_SERVER_ENABLE_DEPTH") or False,
        depth_model_id=os.environ.get("MPS_SERVER_DEPTH_MODEL_ID", "depth-anything/DA3-BASE"),
        depth_device=os.environ.get("MPS_SERVER_DEPTH_DEVICE", "auto").strip().lower(),
        depth_process_res=max(224, int(os.environ.get("MPS_SERVER_DEPTH_PROCESS_RES", "504"))),
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
