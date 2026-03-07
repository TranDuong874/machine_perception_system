from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

from .InputSource import InputSource


T = TypeVar("T")


class StreamRunner(Generic[T]):
    def __init__(
        self,
        source: InputSource[T],
        queue_size: int = 16,
        poll_timeout_s: float = 0.1,
        drop_oldest_when_full: bool = True,
    ) -> None:
        self.source = source
        self.queue_size = queue_size
        self.poll_timeout_s = poll_timeout_s
        self.drop_oldest_when_full = drop_oldest_when_full

        self.output_queue: queue.Queue[T] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            raise RuntimeError("StreamRunner is already running.")

        self.stop_event.clear()
        self.done_event.clear()
        self.source.open()
        self._worker_thread = threading.Thread(
            target=self._run,
            name="stream-runner",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    def read(self, timeout_s: float | None = None) -> T | None:
        timeout = self.poll_timeout_s if timeout_s is None else timeout_s
        while True:
            if self.done_event.is_set() and self.output_queue.empty():
                return None
            try:
                return self.output_queue.get(timeout=timeout)
            except queue.Empty:
                if self.done_event.is_set():
                    return None

    def is_done(self) -> bool:
        return self.done_event.is_set() and self.output_queue.empty()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                item = self.source.read_next()
                if item is None:
                    break
                self._push_item(item)
        finally:
            self.source.close()
            self.done_event.set()
            self.stop_event.set()

    def _push_item(self, item: T) -> None:
        if not self.output_queue.full():
            self.output_queue.put_nowait(item)
            return

        if not self.drop_oldest_when_full:
            while not self.stop_event.is_set():
                try:
                    self.output_queue.put(item, timeout=self.poll_timeout_s)
                    return
                except queue.Full:
                    continue
            return

        try:
            self.output_queue.get_nowait()
        except queue.Empty:
            pass
        self.output_queue.put_nowait(item)
