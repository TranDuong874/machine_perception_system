from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu


BRIDGE_ROOT = Path("/bridge")
INBOX_DIR = BRIDGE_ROOT / "inbox"
OUTBOX_DIR = BRIDGE_ROOT / "outbox"
IMAGES_IN_DIR = BRIDGE_ROOT / "images_in"
IMAGES_OUT_DIR = BRIDGE_ROOT / "images_out"
READY_FILE = BRIDGE_ROOT / "bridge.ready"

CAMERA_TOPIC = "/cam0/image_raw"
IMU_TOPIC = "/imu0"
CAMERA_POSE_TOPIC = "/orb_slam3/camera_pose"
BODY_ODOM_TOPIC = "/orb_slam3/body_odom"
TRACKING_IMAGE_TOPIC = "/orb_slam3/tracking_image"


class OrbBridge:
    def __init__(self) -> None:
        self.camera_pub = rospy.Publisher(CAMERA_TOPIC, Image, queue_size=1)
        self.imu_pub = rospy.Publisher(IMU_TOPIC, Imu, queue_size=200)

        self._camera_pose_by_stamp: dict[int, PoseStamped] = {}
        self._body_odom_by_stamp: dict[int, Odometry] = {}
        self._tracking_image_by_stamp: dict[int, Image] = {}

        rospy.Subscriber(CAMERA_POSE_TOPIC, PoseStamped, self._on_camera_pose, queue_size=20)
        rospy.Subscriber(BODY_ODOM_TOPIC, Odometry, self._on_body_odom, queue_size=20)
        rospy.Subscriber(TRACKING_IMAGE_TOPIC, Image, self._on_tracking_image, queue_size=20)

    def _on_camera_pose(self, msg: PoseStamped) -> None:
        self._camera_pose_by_stamp[msg.header.stamp.to_nsec()] = msg
        self._trim(self._camera_pose_by_stamp)

    def _on_body_odom(self, msg: Odometry) -> None:
        self._body_odom_by_stamp[msg.header.stamp.to_nsec()] = msg
        self._trim(self._body_odom_by_stamp)

    def _on_tracking_image(self, msg: Image) -> None:
        self._tracking_image_by_stamp[msg.header.stamp.to_nsec()] = msg
        self._trim(self._tracking_image_by_stamp)

    def run(self) -> None:
        READY_FILE.write_text("ready\n", encoding="utf-8")
        while not rospy.is_shutdown():
            pending = sorted(INBOX_DIR.glob("*.json"))
            if not pending:
                time.sleep(0.01)
                continue

            request_path = pending[0]
            processing_path = request_path.with_suffix(".processing")
            try:
                request_path.replace(processing_path)
            except FileNotFoundError:
                continue

            try:
                payload = json.loads(processing_path.read_text(encoding="utf-8"))
                self._handle_packet(payload)
            except Exception as exc:
                timestamp_ns = int(processing_path.stem) if processing_path.stem.isdigit() else 0
                self._write_result(
                    timestamp_ns=timestamp_ns,
                    tracking_state="BRIDGE_ERROR",
                    error=str(exc),
                )
            finally:
                processing_path.unlink(missing_ok=True)

    def _handle_packet(self, payload: dict) -> None:
        timestamp_ns = int(payload["timestamp_ns"])
        image_path = IMAGES_IN_DIR / payload["image_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"missing bridge image: {image_path}")

        for imu_payload in payload.get("imu_samples", ()):
            self.imu_pub.publish(self._build_imu_message(imu_payload))

        self.camera_pub.publish(self._build_image_message(timestamp_ns, image_bgr))
        result = self._await_result(timestamp_ns)
        self._write_result(timestamp_ns=timestamp_ns, **result)

    def _await_result(self, timestamp_ns: int) -> dict:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            camera_pose = self._pick_message(self._camera_pose_by_stamp, timestamp_ns)
            body_odom = self._pick_message(self._body_odom_by_stamp, timestamp_ns)
            tracking_image = self._pick_message(self._tracking_image_by_stamp, timestamp_ns)

            if camera_pose is not None or body_odom is not None or tracking_image is not None:
                tracking_image_path = None
                if tracking_image is not None:
                    tracking_image_path = f"{timestamp_ns}.png"
                    cv2.imwrite(
                        str(IMAGES_OUT_DIR / tracking_image_path),
                        image_message_to_bgr(tracking_image),
                    )

                result = {
                    "tracking_state": infer_tracking_state(camera_pose, body_odom, tracking_image),
                    "camera_translation_xyz": pose_translation(camera_pose.pose) if camera_pose else None,
                    "camera_quaternion_wxyz": pose_quaternion(camera_pose.pose) if camera_pose else None,
                    "body_translation_xyz": odom_translation(body_odom) if body_odom else None,
                    "body_quaternion_wxyz": odom_quaternion(body_odom) if body_odom else None,
                    "tracking_image_path": tracking_image_path,
                    "error": None,
                }
                self._discard_older(self._camera_pose_by_stamp, timestamp_ns)
                self._discard_older(self._body_odom_by_stamp, timestamp_ns)
                self._discard_older(self._tracking_image_by_stamp, timestamp_ns)
                return result
            time.sleep(0.01)

        return {
            "tracking_state": "NO_OUTPUT",
            "camera_translation_xyz": None,
            "camera_quaternion_wxyz": None,
            "body_translation_xyz": None,
            "body_quaternion_wxyz": None,
            "tracking_image_path": None,
            "error": f"timeout waiting for ORB topics for {timestamp_ns}",
        }

    def _build_image_message(self, timestamp_ns: int, image_bgr: np.ndarray) -> Image:
        message = Image()
        message.header.stamp = rospy.Time.from_sec(timestamp_ns / 1_000_000_000.0)
        message.header.frame_id = "camera"
        message.height = int(image_bgr.shape[0])
        message.width = int(image_bgr.shape[1])
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = int(image_bgr.shape[1] * image_bgr.shape[2])
        message.data = image_bgr.tobytes()
        return message

    def _build_imu_message(self, imu_payload: dict) -> Imu:
        message = Imu()
        timestamp_ns = int(imu_payload["timestamp_ns"])
        message.header.stamp = rospy.Time.from_sec(timestamp_ns / 1_000_000_000.0)
        message.header.frame_id = "imu"
        gx, gy, gz = imu_payload["angular_velocity_rad_s"]
        ax, ay, az = imu_payload["linear_acceleration_m_s2"]
        message.angular_velocity.x = float(gx)
        message.angular_velocity.y = float(gy)
        message.angular_velocity.z = float(gz)
        message.linear_acceleration.x = float(ax)
        message.linear_acceleration.y = float(ay)
        message.linear_acceleration.z = float(az)
        return message

    def _pick_message(self, messages_by_stamp: dict[int, object], timestamp_ns: int):
        if timestamp_ns in messages_by_stamp:
            return messages_by_stamp[timestamp_ns]
        future_stamps = [stamp for stamp in messages_by_stamp if stamp >= timestamp_ns]
        if not future_stamps:
            return None
        return messages_by_stamp[min(future_stamps)]

    def _discard_older(self, messages_by_stamp: dict[int, object], timestamp_ns: int) -> None:
        for stamp in list(messages_by_stamp):
            if stamp <= timestamp_ns:
                messages_by_stamp.pop(stamp, None)

    def _trim(self, messages_by_stamp: dict[int, object], keep: int = 256) -> None:
        if len(messages_by_stamp) <= keep:
            return
        for stamp in sorted(messages_by_stamp)[:-keep]:
            messages_by_stamp.pop(stamp, None)

    def _write_result(
        self,
        timestamp_ns: int,
        tracking_state: str | None,
        camera_translation_xyz=None,
        camera_quaternion_wxyz=None,
        body_translation_xyz=None,
        body_quaternion_wxyz=None,
        tracking_image_path: str | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "timestamp_ns": timestamp_ns,
            "tracking_state": tracking_state,
            "camera_translation_xyz": camera_translation_xyz,
            "camera_quaternion_wxyz": camera_quaternion_wxyz,
            "body_translation_xyz": body_translation_xyz,
            "body_quaternion_wxyz": body_quaternion_wxyz,
            "tracking_image_path": tracking_image_path,
            "error": error,
        }
        target_path = OUTBOX_DIR / f"{timestamp_ns}.json"
        tmp_path = OUTBOX_DIR / f".{timestamp_ns}.json.tmp"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(target_path)


def infer_tracking_state(camera_pose: PoseStamped | None, body_odom: Odometry | None, tracking_image: Image | None) -> str:
    if camera_pose is not None:
        return "TRACKING"
    if body_odom is not None:
        return "BODY_ONLY"
    if tracking_image is not None:
        return "IMAGE_ONLY"
    return "NO_OUTPUT"


def pose_translation(pose) -> list[float]:
    return [pose.position.x, pose.position.y, pose.position.z]


def pose_quaternion(pose) -> list[float]:
    return [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]


def odom_translation(odom: Odometry) -> list[float]:
    pose = odom.pose.pose
    return [pose.position.x, pose.position.y, pose.position.z]


def odom_quaternion(odom: Odometry) -> list[float]:
    pose = odom.pose.pose
    return [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]


def image_message_to_bgr(message: Image) -> np.ndarray:
    if message.encoding in {"bgr8", "rgb8"}:
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3)
        if message.encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
    if message.encoding == "mono8":
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported tracking image encoding: {message.encoding}")


def main() -> None:
    for path in (BRIDGE_ROOT, INBOX_DIR, OUTBOX_DIR, IMAGES_IN_DIR, IMAGES_OUT_DIR):
        path.mkdir(parents=True, exist_ok=True)

    rospy.init_node("mps_orb_bridge", anonymous=False, disable_signals=True)
    bridge = OrbBridge()
    bridge.run()


if __name__ == "__main__":
    main()
