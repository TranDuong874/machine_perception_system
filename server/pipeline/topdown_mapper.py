from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from proto import perception_pb2


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class DepthEstimate:
    metric_depth: np.ndarray
    confidence_error: np.ndarray | None
    confidence_mask: np.ndarray | None


class TopDownMapper:
    def __init__(self) -> None:
        self._enabled = _env_bool("MPS_SERVER_TOPDOWN_ENABLE", True)
        self._map_id = os.environ.get("MPS_SERVER_TOPDOWN_MAP_ID", "map_main")
        self._width = int(os.environ.get("MPS_SERVER_TOPDOWN_WIDTH", "480"))
        self._height = int(os.environ.get("MPS_SERVER_TOPDOWN_HEIGHT", "480"))
        self._resolution_m = float(os.environ.get("MPS_SERVER_TOPDOWN_RESOLUTION_M", "0.10"))
        self._sample_step = max(2, int(os.environ.get("MPS_SERVER_TOPDOWN_SAMPLE_STEP", "20")))
        self._update_every_n = max(1, int(os.environ.get("MPS_SERVER_TOPDOWN_UPDATE_EVERY_N", "3")))

        # Default to true depth path now (UniDepth).
        self._depth_backend = os.environ.get("MPS_SERVER_TOPDOWN_DEPTH_BACKEND", "unidepth_v2").strip().lower()
        self._allow_depth_fallback = _env_bool("MPS_SERVER_TOPDOWN_ALLOW_DEPTH_FALLBACK", False)
        self._unidepth_model_id = os.environ.get(
            "MPS_SERVER_TOPDOWN_UNIDEPTH_MODEL_ID",
            "lpiccinelli/unidepth-v2-vits14",
        )
        self._unidepth_resolution_level = int(os.environ.get("MPS_SERVER_TOPDOWN_UNIDEPTH_RESOLUTION_LEVEL", "0"))
        self._unidepth_confidence_quantile = float(
            os.environ.get("MPS_SERVER_TOPDOWN_UNIDEPTH_CONFIDENCE_QUANTILE", "0.35")
        )
        self._unidepth_confidence_max_error = float(
            os.environ.get("MPS_SERVER_TOPDOWN_UNIDEPTH_CONFIDENCE_MAX_ERROR", "0.0")
        )

        self._log_odds_hit = float(os.environ.get("MPS_SERVER_TOPDOWN_LOG_ODDS_HIT", "0.85"))
        self._log_odds_miss = float(os.environ.get("MPS_SERVER_TOPDOWN_LOG_ODDS_MISS", "0.10"))
        self._log_odds_min = float(os.environ.get("MPS_SERVER_TOPDOWN_LOG_ODDS_MIN", "-2.0"))
        self._log_odds_max = float(os.environ.get("MPS_SERVER_TOPDOWN_LOG_ODDS_MAX", "3.5"))
        self._occupied_threshold = float(os.environ.get("MPS_SERVER_TOPDOWN_OCCUPIED_THRESHOLD", "0.5"))
        self._depth_min_m = float(os.environ.get("MPS_SERVER_TOPDOWN_DEPTH_MIN_M", "0.1"))
        self._depth_max_m = float(os.environ.get("MPS_SERVER_TOPDOWN_DEPTH_MAX_M", "200.0"))
        self._depth_low_quantile = float(os.environ.get("MPS_SERVER_TOPDOWN_DEPTH_LOW_QUANTILE", "0.05"))
        self._depth_high_quantile = float(os.environ.get("MPS_SERVER_TOPDOWN_DEPTH_HIGH_QUANTILE", "0.95"))

        self._mock_depth_near_m = float(os.environ.get("MPS_SERVER_TOPDOWN_MOCK_DEPTH_NEAR_M", "0.8"))
        self._mock_depth_far_m = float(os.environ.get("MPS_SERVER_TOPDOWN_MOCK_DEPTH_FAR_M", "4.2"))

        self._origin_x_m = -(self._width * self._resolution_m) * 0.5
        self._origin_z_m = (self._height * self._resolution_m) * 0.5

        self._log_odds = np.zeros((self._height, self._width), dtype=np.float32)
        self._revision = 0
        self._frames_seen = 0
        self._latest_png = _encode_empty_map_png(self._width, self._height)

        self._fisheye_k, self._fisheye_d = _load_fisheye_calibration()

        self._unidepth_model = None
        self._unidepth_device = None
        self._init_depth_backend()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process_packet(self, packet: perception_pb2.ServerPerceptionPacket) -> perception_pb2.TopDownMap | None:
        if not self._enabled:
            return None

        self._frames_seen += 1
        tracking_state = packet.orb_tracking.tracking_state if packet.HasField("orb_tracking") else "NO_ORB"
        pose = _pose_from_packet(packet)
        pose_valid = pose is not None and _is_pose_state_usable(tracking_state)

        if pose_valid and (self._frames_seen % self._update_every_n) == 0:
            image_bgr = _decode_jpeg(packet.image_jpeg)
            if image_bgr is not None:
                undistorted_bgr, intrinsics = self._undistort_if_needed(image_bgr)
                depth_estimate = self._predict_depth(undistorted_bgr, intrinsics)
                if depth_estimate is not None:
                    self._integrate_frame(depth_estimate, intrinsics, pose)
                    self._revision += 1
                    self._latest_png = _encode_log_odds_png(
                        self._log_odds,
                        occupied_threshold=self._occupied_threshold,
                    )

        topdown = perception_pb2.TopDownMap(
            map_id=self._map_id,
            revision=self._revision,
            timestamp_ns=int(packet.timestamp_ns),
            tracking_state=tracking_state,
            pose_valid=bool(pose_valid),
            resolution_m=float(self._resolution_m),
            width=int(self._width),
            height=int(self._height),
            origin_x_m=float(self._origin_x_m),
            origin_z_m=float(self._origin_z_m),
            occupancy_png=self._latest_png,
            backend=self._depth_backend,
        )

        if pose is not None:
            translation, _rotation_wc = pose
            topdown.latest_user_translation_xyz[:] = [
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ]

        return topdown

    def _init_depth_backend(self) -> None:
        if self._depth_backend in {"mock", "dummy"}:
            self._depth_backend = "mock"
            print("[topdown] depth backend: mock")
            return
        if self._depth_backend not in {"unidepth", "unidepth_v2"}:
            raise RuntimeError(
                f"Unsupported MPS_SERVER_TOPDOWN_DEPTH_BACKEND={self._depth_backend!r}; "
                "use 'unidepth_v2' or 'mock'."
            )
        self._init_unidepth()
        if self._depth_backend == "mock":
            print("[topdown] depth backend: mock")
            return
        print(
            f"[topdown] depth backend: unidepth_v2 model_id={self._unidepth_model_id} "
            f"device={self._unidepth_device}"
        )

    def _undistort_if_needed(self, image_bgr: np.ndarray) -> tuple[np.ndarray, CameraIntrinsics]:
        h, w = image_bgr.shape[:2]
        if self._fisheye_k is None or self._fisheye_d is None:
            intrinsics = _default_intrinsics(width=w, height=h)
            return image_bgr, intrinsics

        K = self._fisheye_k.copy()
        D = self._fisheye_d.copy()
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            D,
            (w, h),
            np.eye(3, dtype=np.float64),
            balance=0.0,
        )
        undistorted = cv2.fisheye.undistortImage(image_bgr, K, D=D, Knew=new_k, new_size=(w, h))
        intrinsics = CameraIntrinsics(
            fx=float(new_k[0, 0]),
            fy=float(new_k[1, 1]),
            cx=float(new_k[0, 2]),
            cy=float(new_k[1, 2]),
        )
        return undistorted, intrinsics

    def _predict_depth(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics) -> DepthEstimate | None:
        if self._depth_backend in {"unidepth", "unidepth_v2"}:
            return self._predict_depth_unidepth(image_bgr, intrinsics)
        return self._predict_depth_mock(image_bgr)

    def _predict_depth_mock(self, image_bgr: np.ndarray) -> DepthEstimate:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        depth = self._mock_depth_near_m + (1.0 - gray) * (self._mock_depth_far_m - self._mock_depth_near_m)
        return DepthEstimate(metric_depth=depth.astype(np.float32), confidence_error=None, confidence_mask=None)

    def _predict_depth_unidepth(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics) -> DepthEstimate:
        if self._unidepth_model is None or self._unidepth_device is None:
            raise RuntimeError("UniDepth backend selected but model is not initialized.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"torch unavailable for UniDepth inference: {exc}") from exc

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).to(self._unidepth_device)
        camera = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        camera_tensor = torch.from_numpy(camera).to(self._unidepth_device)

        with torch.inference_mode():
            outputs = self._unidepth_model.infer(rgb, camera_tensor)

        metric_depth = outputs["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        confidence_tensor = outputs.get("confidence")
        confidence_error = None
        confidence_mask = None
        if confidence_tensor is not None:
            confidence_error = confidence_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            confidence_mask = self._build_confidence_mask(confidence_error)

        return DepthEstimate(
            metric_depth=metric_depth,
            confidence_error=confidence_error,
            confidence_mask=confidence_mask,
        )

    def _build_confidence_mask(self, confidence_error: np.ndarray) -> np.ndarray | None:
        valid = np.isfinite(confidence_error) & (confidence_error > 0)
        if not np.any(valid):
            return None
        threshold = float(np.quantile(confidence_error[valid], self._unidepth_confidence_quantile))
        if self._unidepth_confidence_max_error > 0:
            threshold = min(threshold, self._unidepth_confidence_max_error)
        return valid & (confidence_error <= threshold)

    def _integrate_frame(
        self,
        depth_estimate: DepthEstimate,
        intrinsics: CameraIntrinsics,
        pose: tuple[np.ndarray, np.ndarray],
    ) -> None:
        translation, rotation_wc = pose
        depth_m = depth_estimate.metric_depth
        height, width = depth_m.shape

        ys, xs = np.mgrid[0:height:self._sample_step, 0:width:self._sample_step]
        sampled_depth = depth_m[::self._sample_step, ::self._sample_step]
        finite = np.isfinite(sampled_depth)
        if not np.any(finite):
            return
        finite_values = sampled_depth[finite]
        low_q = float(np.quantile(finite_values, self._depth_low_quantile))
        high_q = float(np.quantile(finite_values, self._depth_high_quantile))
        valid = (
            finite
            & (sampled_depth >= max(self._depth_min_m, low_q))
            & (sampled_depth <= min(self._depth_max_m, high_q))
        )
        if depth_estimate.confidence_mask is not None:
            sampled_conf = depth_estimate.confidence_mask[::self._sample_step, ::self._sample_step]
            candidate = valid & sampled_conf
            if np.any(candidate):
                valid = candidate
        if not np.any(valid):
            return

        u = xs[valid].astype(np.float64)
        v = ys[valid].astype(np.float64)
        z = sampled_depth[valid].astype(np.float64)

        x_cam = ((u - intrinsics.cx) / intrinsics.fx) * z
        y_cam = ((v - intrinsics.cy) / intrinsics.fy) * z
        xyz_cam = np.stack([x_cam, y_cam, z], axis=1)
        xyz_world = (rotation_wc @ xyz_cam.T).T + translation

        cam_x, cam_z = float(translation[0]), float(translation[2])
        cam_col, cam_row = self._world_to_grid(cam_x, cam_z)
        if cam_col is None or cam_row is None:
            return

        for point in xyz_world:
            end_col, end_row = self._world_to_grid(float(point[0]), float(point[2]))
            if end_col is None or end_row is None:
                continue
            self._apply_ray(cam_col, cam_row, end_col, end_row)

    def _world_to_grid(self, x_m: float, z_m: float) -> tuple[int | None, int | None]:
        col = int(math.floor((x_m - self._origin_x_m) / self._resolution_m))
        row = int(math.floor((self._origin_z_m - z_m) / self._resolution_m))
        if col < 0 or row < 0 or col >= self._width or row >= self._height:
            return None, None
        return col, row

    def _apply_ray(self, start_col: int, start_row: int, end_col: int, end_row: int) -> None:
        dc = end_col - start_col
        dr = end_row - start_row
        steps = max(abs(dc), abs(dr), 1)

        for step in range(steps):
            alpha = step / steps
            col = int(round(start_col + alpha * dc))
            row = int(round(start_row + alpha * dr))
            if 0 <= col < self._width and 0 <= row < self._height:
                self._log_odds[row, col] = np.clip(
                    self._log_odds[row, col] - self._log_odds_miss,
                    self._log_odds_min,
                    self._log_odds_max,
                )

        if 0 <= end_col < self._width and 0 <= end_row < self._height:
            self._log_odds[end_row, end_col] = np.clip(
                self._log_odds[end_row, end_col] + self._log_odds_hit,
                self._log_odds_min,
                self._log_odds_max,
            )

    def _init_unidepth(self) -> None:
        try:
            import torch
        except Exception as exc:
            self._handle_backend_init_failure(f"torch unavailable: {exc}")
            return

        try:
            from unidepth.models import UniDepthV2
        except Exception:
            unidepth_repo = Path(__file__).resolve().parents[3] / "spatial_mapping_demo" / "UniDepth"
            if unidepth_repo.exists() and str(unidepth_repo) not in sys.path:
                sys.path.insert(0, str(unidepth_repo))
            try:
                from unidepth.models import UniDepthV2  # type: ignore
            except Exception as exc:
                self._handle_backend_init_failure(f"unidepth import failed: {exc}")
                return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = UniDepthV2.from_pretrained(self._unidepth_model_id).to(device)
            model.eval()
            model.resolution_level = self._unidepth_resolution_level
        except Exception as exc:
            self._handle_backend_init_failure(f"unidepth model init failed: {exc}")
            return

        self._unidepth_model = model
        self._unidepth_device = device

    def _handle_backend_init_failure(self, reason: str) -> None:
        if self._allow_depth_fallback:
            print(f"[topdown] UniDepth unavailable ({reason}); fallback to mock depth.")
            self._depth_backend = "mock"
            self._unidepth_model = None
            self._unidepth_device = None
            return
        raise RuntimeError(
            "UniDepth initialization failed and fallback is disabled. "
            f"Reason: {reason}. "
            "Set MPS_SERVER_TOPDOWN_ALLOW_DEPTH_FALLBACK=1 to use mock depth fallback."
        )


def _decode_jpeg(encoded: bytes) -> np.ndarray | None:
    if not encoded:
        return None
    arr = np.frombuffer(encoded, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _pose_from_packet(
    packet: perception_pb2.ServerPerceptionPacket,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not packet.HasField("orb_tracking"):
        return None
    orb = packet.orb_tracking
    if len(orb.camera_translation_xyz) != 3 or len(orb.camera_quaternion_wxyz) != 4:
        return None

    translation = np.asarray(orb.camera_translation_xyz, dtype=np.float64)
    q = np.asarray(orb.camera_quaternion_wxyz, dtype=np.float64)
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(q)):
        return None
    rotation = _quaternion_to_rotation_matrix(q)
    if rotation is None:
        return None
    return translation, rotation


def _quaternion_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray | None:
    w, x, y, z = [float(v) for v in q_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return None
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _default_intrinsics(width: int, height: int) -> CameraIntrinsics:
    focal = max(width, height) * 0.90
    return CameraIntrinsics(
        fx=float(focal),
        fy=float(focal),
        cx=float(width * 0.5),
        cy=float(height * 0.5),
    )


def _load_fisheye_calibration() -> tuple[np.ndarray | None, np.ndarray | None]:
    k_values = _parse_csv_floats(os.environ.get("MPS_SERVER_TOPDOWN_FISHEYE_K", ""))
    d_values = _parse_csv_floats(os.environ.get("MPS_SERVER_TOPDOWN_FISHEYE_D", ""))
    if len(k_values) != 4 or len(d_values) < 4:
        return None, None

    fx, fy, cx, cy = k_values[:4]
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.asarray(d_values[:4], dtype=np.float64).reshape(4, 1)
    return K, D


def _parse_csv_floats(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _encode_empty_map_png(width: int, height: int) -> bytes:
    base = np.full((height, width), 20, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", base)
    if not ok:
        return b""
    return encoded.tobytes()


def _encode_log_odds_png(log_odds: np.ndarray, occupied_threshold: float) -> bytes:
    image = np.full(log_odds.shape, 20, dtype=np.uint8)
    image[log_odds >= occupied_threshold] = 230
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return b""
    return encoded.tobytes()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _is_pose_state_usable(tracking_state: str | None) -> bool:
    if tracking_state is None:
        return False
    state = tracking_state.strip().upper()
    return state in {"OK", "TRACKING", "RECENTLY_LOST"}
