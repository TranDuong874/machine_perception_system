from __future__ import annotations

import os
from pathlib import Path
import queue
import threading

import cv2
import numpy as np

from schema.sensor_schema import SynchronizedSensorPacket, YoloDetection, YoloResult


class DetectionService:
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        queue_size: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.input_queue: queue.Queue[SynchronizedSensorPacket] = queue.Queue(maxsize=queue_size)
        self._results: dict[int, YoloResult] = {}
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._model = None
        self._model_error: str | None = None
        self._settings_dir = Path("/tmp/machine_perception_system_ultralytics")

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._settings_dir.mkdir(parents=True, exist_ok=True)
        self._worker_thread = threading.Thread(target=self._run, name="detection-service", daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def submit(self, packet: SynchronizedSensorPacket) -> None:
        if not self.input_queue.full():
            self.input_queue.put_nowait(packet)
            return
        try:
            self.input_queue.get_nowait()
        except queue.Empty:
            pass
        self.input_queue.put_nowait(packet)

    def await_result(self, timestamp_ns: int, timeout_s: float = 1.0) -> YoloResult | None:
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
            result = self._detect(packet)
            with self._condition:
                self._results[packet.timestamp_ns] = result
                self._condition.notify_all()

    def _detect(self, packet: SynchronizedSensorPacket) -> YoloResult:
        model = self._ensure_model()
        if model is None:
            return YoloResult(error=self._model_error)

        try:
            results = model.predict(packet.image_bgr, device=self.device, conf=self.confidence_threshold, verbose=False)
        except Exception as exc:
            return YoloResult(error=f"prediction failed: {exc}")
        if not results:
            return YoloResult(detections=(), annotated_image_bgr=packet.image_bgr)

        result = results[0]
        names = result.names
        detections: list[YoloDetection] = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    YoloDetection(
                        class_id=cls_id,
                        class_name=str(names.get(cls_id, cls_id)),
                        confidence=conf,
                        xyxy=(x1, y1, x2, y2),
                    )
                )
        annotated = result.plot()
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR) if annotated.ndim == 3 else packet.image_bgr
        return YoloResult(detections=tuple(detections), annotated_image_bgr=annotated_bgr)

    def _ensure_model(self):
        if self._model is not None or self._model_error is not None:
            return self._model
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self._settings_dir))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self._model_error = f"ultralytics unavailable: {exc}"
            return None
        if not self.model_path.exists():
            self._model_error = f"missing model: {self.model_path}"
            return None
        self._model = YOLO(str(self.model_path))
        return self._model
