from __future__ import annotations

from dataclasses import dataclass
import math
import threading

from server.runtime.processing_models import ProcessedPacket


VALID_TRACKING_STATES = {"TRACKING", "TRACKING_KLT", "RECENTLY_LOST"}


@dataclass(frozen=True)
class TaskSelectionConfig:
    depth_every_n_frames: int
    depth_translation_m: float
    depth_rotation_deg: float
    memory_every_n_frames: int
    memory_translation_m: float
    memory_rotation_deg: float
    memory_schedule_on_detection_change: bool


@dataclass(frozen=True)
class TaskSelectionDecision:
    frame_index: int
    schedule_depth: bool
    schedule_memory: bool


@dataclass
class _SelectionState:
    frame_index: int
    camera_position_xyz: tuple[float, float, float] | None
    raw_quaternion_wxyz: tuple[float, float, float, float] | None
    yolo_class_names: tuple[str, ...]


class FrameSelectionService:
    def __init__(self, *, config: TaskSelectionConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._frame_index = 0
        self._last_depth_state: _SelectionState | None = None
        self._last_memory_state: _SelectionState | None = None

    def decide(self, processed_packet: ProcessedPacket) -> TaskSelectionDecision:
        with self._lock:
            self._frame_index += 1
            frame_index = self._frame_index
            current_state = _SelectionState(
                frame_index=frame_index,
                camera_position_xyz=processed_packet.pose.camera_position_xyz,
                raw_quaternion_wxyz=processed_packet.pose.raw_quaternion_wxyz,
                yolo_class_names=_extract_yolo_class_names(processed_packet),
            )

            valid_pose = (
                processed_packet.pose.pose_valid
                and processed_packet.pose.tracking_state in VALID_TRACKING_STATES
            )
            schedule_depth = valid_pose and self._should_schedule(
                current_state,
                self._last_depth_state,
                every_n_frames=self._config.depth_every_n_frames,
                translation_m=self._config.depth_translation_m,
                rotation_deg=self._config.depth_rotation_deg,
                detection_change=False,
            )
            if schedule_depth:
                self._last_depth_state = current_state

            schedule_memory = self._should_schedule(
                current_state,
                self._last_memory_state,
                every_n_frames=self._config.memory_every_n_frames,
                translation_m=self._config.memory_translation_m,
                rotation_deg=self._config.memory_rotation_deg,
                detection_change=self._config.memory_schedule_on_detection_change,
            )
            if schedule_memory:
                self._last_memory_state = current_state

            return TaskSelectionDecision(
                frame_index=frame_index,
                schedule_depth=schedule_depth,
                schedule_memory=schedule_memory,
            )

    def _should_schedule(
        self,
        current_state: _SelectionState,
        previous_state: _SelectionState | None,
        *,
        every_n_frames: int,
        translation_m: float,
        rotation_deg: float,
        detection_change: bool,
    ) -> bool:
        if previous_state is None:
            return True
        if current_state.frame_index - previous_state.frame_index >= max(1, int(every_n_frames)):
            return True
        if detection_change and current_state.yolo_class_names != previous_state.yolo_class_names:
            return True
        translation_delta = _translation_delta_m(
            current_state.camera_position_xyz,
            previous_state.camera_position_xyz,
        )
        if translation_delta >= max(0.0, float(translation_m)):
            return True
        rotation_delta = _rotation_delta_deg(
            current_state.raw_quaternion_wxyz,
            previous_state.raw_quaternion_wxyz,
        )
        return rotation_delta >= max(0.0, float(rotation_deg))


def _extract_yolo_class_names(processed_packet: ProcessedPacket) -> tuple[str, ...]:
    packet = processed_packet.packet
    if not packet.HasField("yolo_output"):
        return ()
    return tuple(
        sorted(
            {
                str(detection.class_name).strip()
                for detection in packet.yolo_output.detections
                if detection.class_name
            }
        )
    )


def _translation_delta_m(
    current_position: tuple[float, float, float] | None,
    previous_position: tuple[float, float, float] | None,
) -> float:
    if current_position is None or previous_position is None:
        return 0.0
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    dz = current_position[2] - previous_position[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _rotation_delta_deg(
    current_quaternion: tuple[float, float, float, float] | None,
    previous_quaternion: tuple[float, float, float, float] | None,
) -> float:
    if current_quaternion is None or previous_quaternion is None:
        return 0.0

    def _normalize(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        norm = math.sqrt(sum(component * component for component in values))
        if norm <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(component / norm for component in values)

    current = _normalize(current_quaternion)
    previous = _normalize(previous_quaternion)
    dot = abs(sum(lhs * rhs for lhs, rhs in zip(current, previous)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))
