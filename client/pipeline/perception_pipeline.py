from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time

from config import ClientConfig
from input_source.euroc_input_source import EurocInputSource
from input_source.stream_runner import StreamRunner
from local_services import DetectionService, SlamService
from schema.sensor_schema import EnrichedPerceptionPacket, InputSample, OrbResult, SynchronizedSensorPacket, YoloResult
from state import PerceptionSnapshot, SharedClientState
from sync import SensorPacketSynchronizer
from transport import create_client_from_env


@dataclass
class _PipelineRuntime:
    runner: StreamRunner
    synchronizer: SensorPacketSynchronizer
    detection_service: DetectionService
    slam_service: SlamService


class PerceptionPipeline:
    def __init__(self, config: ClientConfig, shared_state: SharedClientState) -> None:
        self._config = config
        self._shared_state = shared_state
        self._runtime = self._build_runtime(config)
        self._perception_client = None
        self._last_transport_error_log_s = 0.0
        self._transport_error_log_interval_s = float(
            os.environ.get("MPS_SERVER_BRIDGE_LOG_INTERVAL_S", "2.0")
        )
        self._last_logged_map_revision = 0

    def run(self) -> None:
        try:
            self._start_runtime()
            self._print_startup_summary()
            self._run_stream_loop()
        finally:
            self._stop_runtime()

    def _build_runtime(self, config: ClientConfig) -> _PipelineRuntime:
        source = EurocInputSource(
            dataset_root=config.dataset_root,
            monocular=True,
            inertial=True,
            replay_speed=config.replay_speed,
        )
        runner = StreamRunner(
            source=source,
            queue_size=config.runner_queue_size,
            poll_timeout_s=config.runner_poll_timeout_s,
            drop_oldest_when_full=config.runner_drop_oldest_when_full,
        )
        return _PipelineRuntime(
            runner=runner,
            synchronizer=SensorPacketSynchronizer(max_pending_imu=config.max_pending_imu),
            detection_service=DetectionService(model_path=config.yolo_model_path, device=config.yolo_device),
            slam_service=SlamService(
                host=config.slam_host,
                port=config.slam_port,
                result_timeout_s=config.slam_timeout_s,
            ),
        )

    def _start_runtime(self) -> None:
        self._runtime.detection_service.start()
        self._runtime.slam_service.start()
        self._runtime.runner.start()

    def _stop_runtime(self) -> None:
        self._runtime.runner.stop()
        self._runtime.detection_service.stop()
        self._runtime.slam_service.stop()
        self._reset_perception_client()

    def _print_startup_summary(self) -> None:
        print("Producer: StreamRunner thread")
        print(f"Dataset: {self._config.dataset_root}")
        print(f"YOLO model: {self._config.yolo_model_path}")
        print(f"Replay speed: {self._config.replay_speed}")
        print(f"SLAM adapter: {self._config.slam_target} timeout_s={self._config.slam_timeout_s}")
        print(f"Server transport: {self._config.server_transport} target={self._config.server_target}")
        if self._config.max_frames is not None:
            print(f"Max frames: {self._config.max_frames}")
        print(f"Synchronizer max pending IMU: {self._config.max_pending_imu}")
        print(
            "Stages: StreamRunner -> SensorPacketSynchronizer -> "
            "DetectionService + SlamService -> PerceptionPipeline -> server"
        )
        if self._config.user_gui_enabled:
            print(
                "UI: click the Assistant Chat input box to ask questions; "
                "press Tab there to switch current/rag mode."
            )
        print("Press 'q' in the preview window to quit.")

    def _run_stream_loop(self) -> None:
        frames_processed = 0
        while not self._shared_state.stop_requested:
            sample = self._runtime.runner.read(timeout_s=self._config.poll_timeout_s)
            if sample is None:
                break

            if not self._process_sample(sample):
                continue

            frames_processed += 1
            if self._config.max_frames is not None and frames_processed >= self._config.max_frames:
                break

    def _process_sample(self, sample: InputSample) -> bool:
        synchronized_packet = self._runtime.synchronizer.push(sample)
        if synchronized_packet is None:
            return False

        enriched_packet = self._build_enriched_perception_packet(synchronized_packet)
        self._shared_state.publish_snapshot(
            PerceptionSnapshot(
                synchronized_packet=synchronized_packet,
                enriched_packet=enriched_packet,
            )
        )
        self._send_to_server(enriched_packet)
        return True

    def _build_enriched_perception_packet(
        self,
        synchronized_packet: SynchronizedSensorPacket,
    ) -> EnrichedPerceptionPacket:
        self._runtime.detection_service.submit(synchronized_packet)
        self._runtime.slam_service.submit(synchronized_packet)

        yolo_output = self._runtime.detection_service.await_result(
            synchronized_packet.timestamp_ns,
            timeout_s=self._config.yolo_timeout_s,
        )
        orb_tracking = self._runtime.slam_service.await_result(
            synchronized_packet.timestamp_ns,
            timeout_s=self._config.slam_timeout_s,
        )

        if yolo_output is None:
            yolo_output = YoloResult(error="timeout")
        if orb_tracking is None:
            orb_tracking = OrbResult(tracking_state="TIMEOUT", error="timeout")

        return EnrichedPerceptionPacket(
            timestamp_ns=synchronized_packet.timestamp_ns,
            image_bgr=synchronized_packet.image_bgr,
            imu_samples=synchronized_packet.imu_samples,
            orb_tracking=orb_tracking,
            yolo_output=yolo_output,
            segmentation_mask=None,
        )

    def _send_to_server(self, packet: EnrichedPerceptionPacket) -> None:
        if self._config.server_transport != "grpc":
            return

        client = self._get_perception_client()
        if client is None:
            return
        try:
            reply = client.send_packet(packet)
        except Exception as exc:
            self._log_transport_warning(f"grpc send failed (will retry): {exc}")
            self._reset_perception_client()
            return

        if reply.status != "ok":
            print(
                f"[PerceptionPipeline] server rejected packet ts={packet.timestamp_ns}: "
                f"status={reply.status} message={reply.message}",
                file=sys.stderr,
            )
            return

        if reply.HasField("topdown_map"):
            map_revision = int(reply.topdown_map.revision)
            if map_revision > self._last_logged_map_revision and (
                map_revision == 1 or map_revision % 20 == 0
            ):
                pose = tuple(reply.topdown_map.latest_user_translation_xyz)
                pose_text = pose if pose else ()
                print(
                    f"[PerceptionPipeline] map revision={map_revision} "
                    f"tracking={reply.topdown_map.tracking_state} "
                    f"pose_valid={reply.topdown_map.pose_valid} "
                    f"user_xyz={pose_text}"
                )
                self._last_logged_map_revision = map_revision

    def _get_perception_client(self):
        if self._perception_client is not None:
            return self._perception_client
        try:
            self._perception_client = create_client_from_env()
        except Exception as exc:
            self._log_transport_warning(f"failed to initialize grpc bridge (will retry): {exc}")
            self._perception_client = None
            return None
        return self._perception_client

    def _reset_perception_client(self) -> None:
        if self._perception_client is None:
            return
        try:
            self._perception_client.close()
        except Exception:
            pass
        finally:
            self._perception_client = None

    def _log_transport_warning(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_transport_error_log_s < self._transport_error_log_interval_s:
            return
        print(f"[PerceptionPipeline] {message}", file=sys.stderr)
        self._last_transport_error_log_s = now
