from __future__ import annotations

import threading

from config import ClientConfig
from pipeline import PerceptionPipeline
from state import SharedClientState
from ui import UserViewPipeline


class ClientRuntime:
    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._shared_state = SharedClientState()
        self._perception_pipeline = PerceptionPipeline(config, self._shared_state)
        self._user_view_pipeline = UserViewPipeline(config, self._shared_state)
        self._pipeline_thread: threading.Thread | None = None
        self._pipeline_error: BaseException | None = None

    def run(self) -> None:
        if not self._config.any_gui_enabled:
            self._perception_pipeline.run()
            return
        try:
            self._start_pipeline_thread()
            self._user_view_pipeline.run_foreground()
        finally:
            self._shared_state.request_stop()
            self._user_view_pipeline.stop()
            self._join_pipeline_thread()
        if self._pipeline_error is not None:
            raise RuntimeError("Perception pipeline failed") from self._pipeline_error

    def _start_pipeline_thread(self) -> None:
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            return
        self._pipeline_error = None
        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline_thread,
            name="client-perception-pipeline",
            daemon=False,
        )
        self._pipeline_thread.start()

    def _join_pipeline_thread(self) -> None:
        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=10.0)
        self._pipeline_thread = None

    def _run_pipeline_thread(self) -> None:
        try:
            self._perception_pipeline.run()
        except BaseException as exc:
            self._pipeline_error = exc
            self._shared_state.request_stop()
        else:
            self._shared_state.request_stop()
