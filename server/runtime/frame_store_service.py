from __future__ import annotations

from collections import OrderedDict
import threading

from server.runtime.processing_models import ProcessedPacket


class FrameStoreService:
    def __init__(self, *, max_packets: int) -> None:
        self._max_packets = max(8, int(max_packets))
        self._lock = threading.Lock()
        self._packets: OrderedDict[int, ProcessedPacket] = OrderedDict()

    def put(self, processed_packet: ProcessedPacket) -> None:
        timestamp_ns = int(processed_packet.packet.timestamp_ns)
        with self._lock:
            self._packets[timestamp_ns] = processed_packet
            self._packets.move_to_end(timestamp_ns)
            while len(self._packets) > self._max_packets:
                self._packets.popitem(last=False)

    def get(self, timestamp_ns: int) -> ProcessedPacket | None:
        with self._lock:
            return self._packets.get(int(timestamp_ns))

    def latest(self) -> ProcessedPacket | None:
        with self._lock:
            if not self._packets:
                return None
            latest_key = next(reversed(self._packets))
            return self._packets[latest_key]

    def recent(self, *, limit: int) -> list[ProcessedPacket]:
        with self._lock:
            values = list(self._packets.values())
        return values[-max(1, int(limit)) :]
