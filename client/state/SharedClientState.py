from __future__ import annotations

from dataclasses import dataclass
import threading

from schema.SensorSchema import EnrichedPerceptionPacket, SynchronizedSensorPacket


@dataclass(frozen=True)
class PerceptionSnapshot:
    synchronized_packet: SynchronizedSensorPacket
    enriched_packet: EnrichedPerceptionPacket


class SharedClientState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_snapshot: PerceptionSnapshot | None = None
        self._stop_event = threading.Event()

    def publish_snapshot(self, snapshot: PerceptionSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot

    def get_latest_snapshot(self) -> PerceptionSnapshot | None:
        with self._lock:
            return self._latest_snapshot

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()
