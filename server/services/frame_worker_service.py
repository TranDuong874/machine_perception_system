from __future__ import annotations

import queue
import threading


class FrameWorkerService:
    def __init__(
        self,
        *,
        name: str,
        queue_size: int,
        handler,
        drop_oldest_when_full: bool,
    ) -> None:
        self._name = name
        self._queue: queue.Queue[int] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._handler = handler
        self._drop_oldest_when_full = drop_oldest_when_full
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._submitted = 0
        self._dropped = 0
        self._processed = 0
        self._errors = 0

    @property
    def submitted(self) -> int:
        with self._lock:
            return self._submitted

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.join()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        self._thread = None

    def submit(self, timestamp_ns: int) -> bool:
        if self._drop_oldest_when_full and self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped += 1
        try:
            self._queue.put_nowait(int(timestamp_ns))
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        with self._lock:
            self._submitted += 1
        return True

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                timestamp_ns = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handler(int(timestamp_ns))
                with self._lock:
                    self._processed += 1
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                print(f"[{self._name}] task error: {exc}")
            finally:
                self._queue.task_done()
