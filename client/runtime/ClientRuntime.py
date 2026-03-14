from __future__ import annotations

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

    def run(self) -> None:
        try:
            self._user_view_pipeline.start()
            self._perception_pipeline.run()
        finally:
            self._shared_state.request_stop()
            self._user_view_pipeline.stop()
