from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
import time

import cv2
import numpy as np

from server.runtime.processing_models import MemoryIndexRecord, ProcessedPacket


@dataclass(frozen=True)
class StoredFrameRecord:
    timestamp_ns: int
    received_unix_s: float
    client_peer: str
    tracking_state: str
    pose_valid: bool
    camera_position_xyz: tuple[float, float, float] | None
    yolo_detection_count: int
    yolo_class_names: tuple[str, ...]
    map_revision: int | None
    source_frame_path: str | None
    rectified_frame_path: str | None
    depth_preview_path: str | None
    created_unix_s: float


class PerceptionStorageService:
    def __init__(
        self,
        *,
        db_path: Path,
        artifact_dir: Path,
        save_debug_artifacts: bool,
        save_map_every_n_updates: int,
    ) -> None:
        self._db_path = Path(db_path)
        self._artifact_dir = Path(artifact_dir)
        self._save_debug_artifacts = save_debug_artifacts
        self._save_map_every_n_updates = max(1, int(save_map_every_n_updates))
        self._lock = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._create_schema()
            self._migrate_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def record_ingested_frame(self, processed_packet: ProcessedPacket) -> None:
        frame_paths = self._persist_frame_artifacts(processed_packet)
        pose = processed_packet.pose
        position = pose.camera_position_xyz
        now_unix_s = time.time()
        yolo_class_names = ",".join(_extract_yolo_class_names(processed_packet))

        with self._lock:
            self._connection.execute(
                """
                INSERT INTO frames (
                    timestamp_ns,
                    received_unix_s,
                    client_peer,
                    tracking_state,
                    pose_valid,
                    camera_x_m,
                    camera_y_m,
                    camera_z_m,
                    yolo_detection_count,
                    yolo_class_names,
                    rectified,
                    rectification_calibration_id,
                    depth_inference_ms,
                    reprojected_point_count,
                    reprojection_confidence_threshold,
                    map_revision,
                    source_frame_path,
                    rectified_frame_path,
                    depth_preview_path,
                    created_unix_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp_ns) DO UPDATE SET
                    received_unix_s=excluded.received_unix_s,
                    client_peer=excluded.client_peer,
                    tracking_state=excluded.tracking_state,
                    pose_valid=excluded.pose_valid,
                    camera_x_m=excluded.camera_x_m,
                    camera_y_m=excluded.camera_y_m,
                    camera_z_m=excluded.camera_z_m,
                    yolo_detection_count=excluded.yolo_detection_count,
                    yolo_class_names=excluded.yolo_class_names,
                    rectified=excluded.rectified,
                    rectification_calibration_id=excluded.rectification_calibration_id,
                    source_frame_path=excluded.source_frame_path,
                    rectified_frame_path=excluded.rectified_frame_path
                """,
                (
                    int(processed_packet.packet.timestamp_ns),
                    float(processed_packet.envelope.received_unix_s),
                    processed_packet.envelope.client_peer,
                    pose.tracking_state,
                    1 if pose.pose_valid else 0,
                    position[0] if position is not None else None,
                    position[1] if position is not None else None,
                    position[2] if position is not None else None,
                    int(processed_packet.yolo_detection_count),
                    yolo_class_names,
                    1 if processed_packet.rectified_frame.rectified else 0,
                    processed_packet.rectified_frame.calibration_id,
                    None,
                    0,
                    None,
                    None,
                    frame_paths["source_frame_path"],
                    frame_paths["rectified_frame_path"],
                    None,
                    now_unix_s,
                ),
            )
            self._connection.commit()

    def record_map_update(self, processed_packet: ProcessedPacket) -> None:
        map_snapshot = processed_packet.map_snapshot
        map_path = self._persist_map_artifact(map_snapshot) if map_snapshot is not None else None
        depth_preview_path = self._persist_depth_artifact(processed_packet)
        depth_result = processed_packet.depth_result
        now_unix_s = time.time()

        with self._lock:
            self._connection.execute(
                """
                UPDATE frames
                SET depth_inference_ms=?,
                    reprojected_point_count=?,
                    reprojection_confidence_threshold=?,
                    map_revision=?,
                    depth_preview_path=?
                WHERE timestamp_ns=?
                """,
                (
                    depth_result.inference_ms if depth_result is not None else None,
                    int(processed_packet.projected_points_world.shape[0]),
                    processed_packet.reprojection_confidence_threshold,
                    map_snapshot.revision if map_snapshot is not None else None,
                    depth_preview_path,
                    int(processed_packet.packet.timestamp_ns),
                ),
            )

            if map_snapshot is not None:
                latest_position = map_snapshot.latest_user_translation_xyz
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO map_snapshots (
                        revision,
                        timestamp_ns,
                        tracking_state,
                        pose_valid,
                        latest_user_x_m,
                        latest_user_y_m,
                        latest_user_z_m,
                        occupancy_png_path,
                        width,
                        height,
                        resolution_m,
                        origin_x_m,
                        origin_z_m,
                        backend,
                        created_unix_s
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(map_snapshot.revision),
                        int(map_snapshot.timestamp_ns),
                        map_snapshot.tracking_state,
                        1 if map_snapshot.pose_valid else 0,
                        latest_position[0] if latest_position is not None else None,
                        latest_position[1] if latest_position is not None else None,
                        latest_position[2] if latest_position is not None else None,
                        map_path,
                        int(map_snapshot.width),
                        int(map_snapshot.height),
                        float(map_snapshot.resolution_m),
                        float(map_snapshot.origin_x_m),
                        float(map_snapshot.origin_z_m),
                        map_snapshot.backend,
                        now_unix_s,
                    ),
                )

            self._connection.commit()

    def record_memory_entry(
        self,
        processed_packet: ProcessedPacket,
        *,
        memory_record: MemoryIndexRecord,
    ) -> None:
        pose = processed_packet.pose
        position = pose.camera_position_xyz
        now_unix_s = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO memory_entries (
                    timestamp_ns,
                    index_position,
                    image_path,
                    metadata_text,
                    embedding_model_name,
                    embedding_pretrained,
                    embedding_device,
                    index_backend,
                    tracking_state,
                    pose_valid,
                    camera_x_m,
                    camera_y_m,
                    camera_z_m,
                    yolo_class_names,
                    created_unix_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(memory_record.timestamp_ns),
                    int(memory_record.index_position),
                    memory_record.image_path,
                    memory_record.metadata_text,
                    memory_record.embedding_model_name,
                    memory_record.embedding_pretrained,
                    memory_record.embedding_device,
                    memory_record.index_backend,
                    pose.tracking_state,
                    1 if pose.pose_valid else 0,
                    position[0] if position is not None else None,
                    position[1] if position is not None else None,
                    position[2] if position is not None else None,
                    ",".join(memory_record.yolo_class_names),
                    now_unix_s,
                ),
            )
            self._connection.commit()

    def record_packet(
        self,
        processed_packet: ProcessedPacket,
        *,
        memory_record: MemoryIndexRecord | None = None,
    ) -> None:
        self.record_ingested_frame(processed_packet)
        self.record_map_update(processed_packet)
        if memory_record is not None:
            self.record_memory_entry(processed_packet, memory_record=memory_record)

    def get_latest_frame_record(self) -> StoredFrameRecord | None:
        return self.get_frame_record(timestamp_ns=None)

    def get_frame_record(self, timestamp_ns: int | None) -> StoredFrameRecord | None:
        if timestamp_ns is None:
            sql = "SELECT * FROM frames ORDER BY timestamp_ns DESC LIMIT 1"
            params: tuple[object, ...] = ()
        else:
            sql = "SELECT * FROM frames WHERE timestamp_ns = ?"
            params = (int(timestamp_ns),)

        with self._lock:
            row = self._connection.execute(sql, params).fetchone()
        if row is None:
            return None
        return _row_to_frame_record(row)

    def get_recent_frame_records(self, *, limit: int) -> list[StoredFrameRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM frames ORDER BY timestamp_ns DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [_row_to_frame_record(row) for row in rows]

    def _persist_map_artifact(self, map_snapshot) -> str | None:
        if map_snapshot is None:
            return None
        if map_snapshot.revision % self._save_map_every_n_updates != 0 and map_snapshot.revision != 1:
            return None

        relative_path = Path("maps") / f"map_rev_{map_snapshot.revision:06d}.png"
        absolute_path = self._artifact_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_bytes(map_snapshot.occupancy_png)
        return str(absolute_path)

    def _persist_frame_artifacts(self, processed_packet: ProcessedPacket) -> dict[str, str]:
        timestamp_ns = int(processed_packet.packet.timestamp_ns)
        frame_dir = self._artifact_dir / "frames" / f"{timestamp_ns}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        source_path = frame_dir / "source.jpg"
        rectified_path = frame_dir / "rectified.jpg"
        cv2.imwrite(str(source_path), processed_packet.source_frame_bgr)
        cv2.imwrite(str(rectified_path), processed_packet.rectified_frame.image_bgr)
        return {
            "source_frame_path": str(source_path),
            "rectified_frame_path": str(rectified_path),
        }

    def _persist_depth_artifact(self, processed_packet: ProcessedPacket) -> str | None:
        if processed_packet.depth_result is None:
            return None

        timestamp_ns = int(processed_packet.packet.timestamp_ns)
        frame_dir = self._artifact_dir / "frames" / f"{timestamp_ns}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        depth_path = frame_dir / "depth.jpg"
        cv2.imwrite(str(depth_path), _colorize_depth(processed_packet.depth_result.depth))
        return str(depth_path)

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS frames (
                timestamp_ns INTEGER PRIMARY KEY,
                received_unix_s REAL NOT NULL,
                client_peer TEXT NOT NULL,
                tracking_state TEXT NOT NULL,
                pose_valid INTEGER NOT NULL,
                camera_x_m REAL,
                camera_y_m REAL,
                camera_z_m REAL,
                yolo_detection_count INTEGER NOT NULL,
                yolo_class_names TEXT NOT NULL DEFAULT '',
                rectified INTEGER NOT NULL,
                rectification_calibration_id TEXT NOT NULL,
                depth_inference_ms REAL,
                reprojected_point_count INTEGER NOT NULL,
                reprojection_confidence_threshold REAL,
                map_revision INTEGER,
                source_frame_path TEXT,
                rectified_frame_path TEXT,
                depth_preview_path TEXT,
                created_unix_s REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS map_snapshots (
                revision INTEGER PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                tracking_state TEXT NOT NULL,
                pose_valid INTEGER NOT NULL,
                latest_user_x_m REAL,
                latest_user_y_m REAL,
                latest_user_z_m REAL,
                occupancy_png_path TEXT,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                resolution_m REAL NOT NULL,
                origin_x_m REAL NOT NULL,
                origin_z_m REAL NOT NULL,
                backend TEXT NOT NULL,
                created_unix_s REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_entries (
                timestamp_ns INTEGER PRIMARY KEY,
                index_position INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                metadata_text TEXT NOT NULL,
                embedding_model_name TEXT NOT NULL,
                embedding_pretrained TEXT NOT NULL,
                embedding_device TEXT NOT NULL,
                index_backend TEXT NOT NULL,
                tracking_state TEXT NOT NULL,
                pose_valid INTEGER NOT NULL,
                camera_x_m REAL,
                camera_y_m REAL,
                camera_z_m REAL,
                yolo_class_names TEXT NOT NULL,
                created_unix_s REAL NOT NULL
            );
            """
        )

    def _migrate_schema(self) -> None:
        self._ensure_column("frames", "yolo_class_names", "TEXT NOT NULL DEFAULT ''")

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        rows = self._connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = {row[1] for row in rows}
        if column_name in columns:
            return
        self._connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
        )
        self._connection.commit()


def _extract_yolo_class_names(processed_packet: ProcessedPacket) -> tuple[str, ...]:
    packet = processed_packet.packet
    if not packet.HasField("yolo_output"):
        return ()
    return tuple(
        sorted(
            {
                str(detection.class_name).strip()
                for detection in packet.yolo_output.detections
                if detection.class_name
            }
        )
    )


def _row_to_frame_record(row: sqlite3.Row) -> StoredFrameRecord:
    x = row["camera_x_m"]
    y = row["camera_y_m"]
    z = row["camera_z_m"]
    position = None
    if x is not None and y is not None and z is not None:
        position = (float(x), float(y), float(z))
    yolo_class_names_raw = row["yolo_class_names"] or ""
    yolo_class_names = tuple(value for value in yolo_class_names_raw.split(",") if value)
    return StoredFrameRecord(
        timestamp_ns=int(row["timestamp_ns"]),
        received_unix_s=float(row["received_unix_s"]),
        client_peer=str(row["client_peer"]),
        tracking_state=str(row["tracking_state"]),
        pose_valid=bool(row["pose_valid"]),
        camera_position_xyz=position,
        yolo_detection_count=int(row["yolo_detection_count"]),
        yolo_class_names=yolo_class_names,
        map_revision=int(row["map_revision"]) if row["map_revision"] is not None else None,
        source_frame_path=str(row["source_frame_path"]) if row["source_frame_path"] else None,
        rectified_frame_path=str(row["rectified_frame_path"]) if row["rectified_frame_path"] else None,
        depth_preview_path=str(row["depth_preview_path"]) if row["depth_preview_path"] else None,
        created_unix_s=float(row["created_unix_s"]),
    )


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid = depth[finite_mask]
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(depth, lo, hi)
    normalized = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
