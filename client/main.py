from __future__ import annotations

from pathlib import Path

import cv2

from input_stream.EurocInputStream import EurocInputStream
from input_stream.StreamRunner import StreamRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "dataset-corridor1_512_16"
WINDOW_NAME = "Input Stream Preview"
POLL_TIMEOUT_S = 0.5
FRAME_DELAY_MS = 1


def _overlay_text(image_bgr, lines: list[str]):
    out = image_bgr.copy()
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    stream = EurocInputStream(dataset_root=DATASET_ROOT, monocular=True, inertial=True)
    runner = StreamRunner(stream=stream, queue_size=8, poll_timeout_s=0.1, drop_oldest_when_full=True)
    runner.start()

    print("Producer: StreamRunner thread")
    print("Consumer: main render loop")
    print("Press 'q' to quit.")

    try:
        while True:
            packet = runner.read(timeout_s=POLL_TIMEOUT_S)
            if packet is None:
                break

            lines = [
                f"frame={packet.frame_index}",
                f"timestamp_ns={packet.timestamp_ns}",
                "gyro(rad/s)="
                f"({packet.angular_velocity_rad_s[0]:+.3f}, "
                f"{packet.angular_velocity_rad_s[1]:+.3f}, "
                f"{packet.angular_velocity_rad_s[2]:+.3f})",
                "accel(m/s^2)="
                f"({packet.linear_acceleration_m_s2[0]:+.3f}, "
                f"{packet.linear_acceleration_m_s2[1]:+.3f}, "
                f"{packet.linear_acceleration_m_s2[2]:+.3f})",
            ]
            frame = _overlay_text(packet.image_bgr, lines)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF
            if key == ord("q"):
                break
    finally:
        runner.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
