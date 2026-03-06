from __future__ import annotations

import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import cv2

from schema.SensorSchema import LocalSensorPacket, OrbResult


class SlamService:
    def __init__(
        self,
        image: str = "orb_slam3_ros:noetic",
        container_name: str = "mps_orb_slam3",
        queue_size: int = 4,
        result_timeout_s: float = 1.5,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.result_timeout_s = result_timeout_s

        self.input_queue: queue.Queue[LocalSensorPacket] = queue.Queue(maxsize=queue_size)
        self._results: dict[int, OrbResult] = {}
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self._repo_root = Path(__file__).resolve().parents[2]
        self._bridge_root = Path("/tmp") / self.container_name
        self._inbox_dir = self._bridge_root / "inbox"
        self._outbox_dir = self._bridge_root / "outbox"
        self._images_in_dir = self._bridge_root / "images_in"
        self._images_out_dir = self._bridge_root / "images_out"
        self._bridge_script = (
            self._repo_root / "client" / "local_services" / "orb_bridge" / "ros_orb_bridge.py"
        )
        self._service_state = "NOT_STARTED"
        self._container_started = False

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._prepare_bridge_dirs()
        self._stop_container()
        self._service_state = self._start_container()
        self._container_started = self._service_state == "RUNNING"

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run, name="slam-service", daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._stop_container()
        self._container_started = False

    def submit(self, packet: LocalSensorPacket) -> None:
        if not self.input_queue.full():
            self.input_queue.put_nowait(packet)
            return
        try:
            self.input_queue.get_nowait()
        except queue.Empty:
            pass
        self.input_queue.put_nowait(packet)

    def await_result(self, timestamp_ns: int, timeout_s: float = 1.0) -> OrbResult | None:
        with self._condition:
            if timestamp_ns in self._results:
                return self._results.pop(timestamp_ns)
            self._condition.wait_for(lambda: timestamp_ns in self._results, timeout=timeout_s)
            return self._results.pop(timestamp_ns, None)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._container_started:
                result = OrbResult(tracking_state=self._service_state, error=self._service_state)
            else:
                result = self._process_packet(packet)

            with self._condition:
                self._results[packet.timestamp_ns] = result
                self._condition.notify_all()

    def _process_packet(self, packet: LocalSensorPacket) -> OrbResult:
        try:
            self._write_request(packet)
        except Exception as exc:
            return OrbResult(tracking_state="BRIDGE_WRITE_FAILED", error=str(exc))

        result_path = self._outbox_dir / f"{packet.timestamp_ns}.json"
        deadline = time.monotonic() + self.result_timeout_s
        while time.monotonic() < deadline and not self._stop_event.is_set():
            if result_path.exists():
                return self._read_result(result_path)
            time.sleep(0.01)

        return OrbResult(
            tracking_state="NO_OUTPUT",
            error=f"timeout waiting for ORB output from {self.container_name}",
        )

    def _write_request(self, packet: LocalSensorPacket) -> None:
        image_path = self._images_in_dir / f"{packet.timestamp_ns}.png"
        if not cv2.imwrite(str(image_path), packet.image_bgr):
            raise RuntimeError(f"failed to write request image: {image_path}")

        payload = {
            "timestamp_ns": packet.timestamp_ns,
            "image_path": image_path.name,
            "imu_samples": [
                {
                    "timestamp_ns": imu.timestamp_ns,
                    "angular_velocity_rad_s": list(imu.angular_velocity_rad_s),
                    "linear_acceleration_m_s2": list(imu.linear_acceleration_m_s2),
                }
                for imu in packet.imu_samples
            ],
        }
        tmp_path = self._inbox_dir / f".{packet.timestamp_ns}.json.tmp"
        target_path = self._inbox_dir / f"{packet.timestamp_ns}.json"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(target_path)

    def _read_result(self, result_path: Path) -> OrbResult:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        finally:
            result_path.unlink(missing_ok=True)

        tracking_image_bgr = None
        tracking_image_name = payload.get("tracking_image_path")
        if tracking_image_name:
            image_path = self._images_out_dir / tracking_image_name
            if image_path.exists():
                tracking_image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        return OrbResult(
            tracking_state=payload.get("tracking_state"),
            camera_translation_xyz=_tuple3(payload.get("camera_translation_xyz")),
            camera_quaternion_wxyz=_tuple4(payload.get("camera_quaternion_wxyz")),
            body_translation_xyz=_tuple3(payload.get("body_translation_xyz")),
            body_quaternion_wxyz=_tuple4(payload.get("body_quaternion_wxyz")),
            tracking_image_bgr=tracking_image_bgr,
            error=payload.get("error"),
        )

    def _prepare_bridge_dirs(self) -> None:
        for path in (
            self._bridge_root,
            self._inbox_dir,
            self._outbox_dir,
            self._images_in_dir,
            self._images_out_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        for path in (
            self._inbox_dir,
            self._outbox_dir,
            self._images_in_dir,
            self._images_out_dir,
        ):
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)

    def _start_container(self) -> str:
        command = (
            "source /opt/ros/noetic/setup.bash && "
            "source /root/catkin_ws/devel/setup.bash && "
            "roscore >/bridge/roscore.log 2>&1 & "
            "sleep 3 && "
            "python3 /bridge_scripts/ros_orb_bridge.py >/bridge/bridge.log 2>&1 & "
            "sleep 1 && "
            "rosrun orb_slam3_ros ros_mono_inertial "
            "_voc_file:=/root/catkin_ws/src/orb_slam3_ros/orb_slam3/Vocabulary/ORBvoc.txt.bin "
            "_settings_file:=/root/catkin_ws/src/orb_slam3_ros/config/Monocular-Inertial/EuRoC.yaml "
            "_world_frame_id:=world "
            "_cam_frame_id:=camera "
            "_imu_frame_id:=imu "
            "_enable_pangolin:=false "
            "/camera/image_raw:=/cam0/image_raw "
            "/imu:=/imu0"
        )
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    self.container_name,
                    "-v",
                    f"{self._bridge_root}:/bridge",
                    "-v",
                    f"{self._bridge_script.parent}:/bridge_scripts:ro",
                    self.image,
                    "/bin/bash",
                    "-lc",
                    command,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "DOCKER_UNAVAILABLE"

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker run failed"
            return f"START_FAILED:{detail}"

        ready_file = self._bridge_root / "bridge.ready"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if ready_file.exists():
                return "RUNNING"
            time.sleep(0.1)

        return "BRIDGE_NOT_READY"

    def _stop_container(self) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            pass


def _tuple3(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float] | None:
    if values is None or len(values) != 3:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple4(values: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float] | None:
    if values is None or len(values) != 4:
        return None
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
