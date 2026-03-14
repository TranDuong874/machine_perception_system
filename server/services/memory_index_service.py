from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch

from server.runtime.processing_models import MemoryIndexRecord, ProcessedPacket


def _try_load_faiss():
    try:
        import faiss

        return faiss
    except Exception:
        return None


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return vectors / norms


def _chunked(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    visual_score: float
    metadata_score: float
    record: MemoryIndexRecord


@dataclass(frozen=True)
class MemoryIndexEntry:
    timestamp_ns: int
    image_path: str
    metadata_text: str
    tracking_state: str
    pose_valid: bool
    client_peer: str
    camera_translation_xyz: tuple[float, float, float] | None
    yolo_class_names: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["camera_translation_xyz"] = (
            list(self.camera_translation_xyz) if self.camera_translation_xyz is not None else None
        )
        payload["yolo_class_names"] = list(self.yolo_class_names)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "MemoryIndexEntry":
        translation = payload.get("camera_translation_xyz")
        return cls(
            timestamp_ns=int(payload["timestamp_ns"]),
            image_path=str(payload["image_path"]),
            metadata_text=str(payload["metadata_text"]),
            tracking_state=str(payload["tracking_state"]),
            pose_valid=bool(payload["pose_valid"]),
            client_peer=str(payload["client_peer"]),
            camera_translation_xyz=tuple(float(value) for value in translation)
            if translation is not None
            else None,
            yolo_class_names=tuple(str(value) for value in payload.get("yolo_class_names", [])),
        )


class VectorIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._faiss = _try_load_faiss()
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        self._index = self._faiss.IndexFlatIP(self.dimension) if self._faiss is not None else None

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    @property
    def size(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        vectors = _normalize_rows(vectors)
        if self._index is not None:
            self._index.add(vectors)
            return
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vector = _normalize_rows(query_vector)
        if self.size == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        top_k = max(1, min(int(top_k), self.size))
        if self._index is not None:
            scores, indices = self._index.search(query_vector, top_k)
            return scores[0], indices[0]
        scores = self._vectors @ query_vector[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return scores[indices], indices.astype(np.int64)

    def save(self, index_path: Path, vectors_path: Path, vectors: np.ndarray) -> None:
        if self._index is not None and self._faiss is not None:
            self._faiss.write_index(self._index, str(index_path))
        np.save(vectors_path, vectors.astype(np.float32, copy=False))

    @classmethod
    def load(
        cls,
        *,
        dimension: int,
        index_path: Path,
        vectors_path: Path,
    ) -> tuple["VectorIndex", np.ndarray]:
        instance = cls(dimension=dimension)
        vectors = np.load(vectors_path) if vectors_path.exists() else np.empty((0, dimension), dtype=np.float32)
        vectors = vectors.astype(np.float32, copy=False)
        if instance._index is not None and index_path.exists():
            instance._index = instance._faiss.read_index(str(index_path))
            return instance, vectors
        if instance._index is not None:
            if vectors.size != 0:
                instance._index.add(_normalize_rows(vectors))
            return instance, vectors
        instance._vectors = vectors
        return instance, instance._vectors


class OpenClipEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        pretrained: str,
        device: str,
        batch_size: int,
    ) -> None:
        import open_clip

        self._open_clip = open_clip
        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        precision = "fp16" if self.device.startswith("cuda") else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
            precision=precision,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.image_dtype = self._infer_image_dtype()
        self._use_cuda_autocast = self.device.startswith("cuda") and self.image_dtype in {
            self._torch.float16,
            self._torch.bfloat16,
        }

    @property
    def dimension(self) -> int:
        text_projection = getattr(self.model, "text_projection", None)
        if text_projection is not None:
            return int(text_projection.shape[-1])
        visual = getattr(self.model, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if output_dim is not None:
            return int(output_dim)
        raise RuntimeError("unable to infer OpenCLIP embedding dimension")

    def encode_image_arrays(self, images_bgr: Iterable[np.ndarray]) -> np.ndarray:
        images_bgr = list(images_bgr)
        if not images_bgr:
            return np.empty((0, self.dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        with self._torch.no_grad():
            for batch_images in _chunked(images_bgr, self.batch_size):
                tensors = []
                for image_bgr in batch_images:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    tensors.append(self.preprocess(Image.fromarray(image_rgb)))
                pixel_values = self._torch.stack(tensors).to(device=self.device, dtype=self.image_dtype)
                if self._use_cuda_autocast:
                    with self._torch.autocast(device_type="cuda", dtype=self.image_dtype):
                        features = self.model.encode_image(pixel_values)
                else:
                    features = self.model.encode_image(pixel_values)
                features = features / features.norm(dim=-1, keepdim=True)
                batches.append(features.float().cpu().numpy())
                if self.device.startswith("cuda"):
                    self._torch.cuda.empty_cache()
        return np.concatenate(batches, axis=0)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        with self._torch.no_grad():
            for batch_texts in _chunked(texts, self.batch_size):
                tokens = self.tokenizer(batch_texts).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                batches.append(features.float().cpu().numpy())
                if self.device.startswith("cuda"):
                    self._torch.cuda.empty_cache()
        return np.concatenate(batches, axis=0)

    def _infer_image_dtype(self):
        visual = getattr(self.model, "visual", None)
        if visual is not None:
            preferred_weights = [
                getattr(getattr(visual, "conv1", None), "weight", None),
                getattr(getattr(getattr(visual, "patch_embed", None), "proj", None), "weight", None),
            ]
            for weight in preferred_weights:
                if weight is not None and getattr(weight, "is_floating_point", lambda: False)():
                    return weight.dtype

        visual = getattr(self.model, "visual", None)
        if visual is not None:
            for param in visual.parameters():
                if getattr(param, "is_floating_point", lambda: False)():
                    return param.dtype
        for param in self.model.parameters():
            if getattr(param, "is_floating_point", lambda: False)():
                return param.dtype
        return self._torch.float32


class MemoryIndexService:
    def __init__(
        self,
        *,
        enabled: bool,
        index_dir: Path,
        model_name: str,
        pretrained: str,
        device: str,
        batch_size: int,
        save_every_n_updates: int,
        visual_weight: float,
        metadata_weight: float,
        encoder: OpenClipEncoder | None = None,
    ) -> None:
        self._enabled = enabled
        self._index_dir = Path(index_dir)
        self._frames_dir = self._index_dir / "frames"
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._batch_size = max(1, int(batch_size))
        self._save_every_n_updates = max(1, int(save_every_n_updates))
        self._visual_weight = float(visual_weight)
        self._metadata_weight = float(metadata_weight)
        self._encoder = encoder
        self._encoder_lock = threading.Lock()
        self._data_lock = threading.Lock()

        self._entries: list[MemoryIndexEntry] = []
        self._records: list[MemoryIndexRecord] = []
        self._timestamp_to_index: dict[int, int] = {}
        self._visual_vectors = np.empty((0, 0), dtype=np.float32)
        self._metadata_vectors = np.empty((0, 0), dtype=np.float32)
        self._visual_index: VectorIndex | None = None
        self._metadata_index: VectorIndex | None = None
        self._backend = "disabled"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def entry_count(self) -> int:
        with self._data_lock:
            return len(self._entries)

    def preflight(self) -> bool:
        if not self._enabled:
            return False

        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        if self._encoder is None:
            try:
                self._encoder = OpenClipEncoder(
                    model_name=self._model_name,
                    pretrained=self._pretrained,
                    device=self._device,
                    batch_size=self._batch_size,
                )
            except Exception as exc:
                self._enabled = False
                self._backend = f"disabled:{exc}"
                print(f"[memory] disabled: {exc}")
                return False

        if self._visual_index is None or self._metadata_index is None:
            self._load_existing_bundle()
        return True

    def add_packet(self, processed_packet: ProcessedPacket) -> MemoryIndexRecord | None:
        if not self.preflight():
            return None
        try:
            timestamp_ns = int(processed_packet.packet.timestamp_ns)
            with self._data_lock:
                existing_index = self._timestamp_to_index.get(timestamp_ns)
                if existing_index is not None:
                    return self._records[existing_index]

            image_path = self._persist_frame_image(processed_packet)
            entry = _build_memory_entry(processed_packet, image_path=image_path)

            with self._encoder_lock:
                assert self._encoder is not None
                visual_vector = self._encoder.encode_image_arrays([processed_packet.rectified_frame.image_bgr])
                metadata_vector = self._encoder.encode_texts([entry.metadata_text])

            with self._data_lock:
                index_position = len(self._entries)
                self._entries.append(entry)
                self._timestamp_to_index[entry.timestamp_ns] = index_position
                self._records.append(
                    MemoryIndexRecord(
                        timestamp_ns=entry.timestamp_ns,
                        index_position=index_position,
                        image_path=entry.image_path,
                        metadata_text=entry.metadata_text,
                        embedding_model_name=self._model_name,
                        embedding_pretrained=self._pretrained,
                        embedding_device=self._encoder.device,
                        index_backend=self._backend,
                        yolo_class_names=entry.yolo_class_names,
                    )
                )
                self._visual_vectors = _append_vector(self._visual_vectors, visual_vector[0])
                self._metadata_vectors = _append_vector(self._metadata_vectors, metadata_vector[0])
                assert self._visual_index is not None and self._metadata_index is not None
                self._visual_index.add(visual_vector)
                self._metadata_index.add(metadata_vector)
                record = self._records[-1]

                if len(self._entries) == 1 or len(self._entries) % self._save_every_n_updates == 0:
                    self._save_bundle_locked()
                return record
        except Exception as exc:
            print(f"[memory] packet skipped: {exc}")
            return None

    def query(self, *, query_text: str, top_k: int) -> list[SearchHit]:
        if not self.preflight():
            return []

        with self._data_lock:
            if not self._entries:
                return []

        with self._encoder_lock:
            assert self._encoder is not None
            query_vector = self._encoder.encode_texts([query_text])

        with self._data_lock:
            assert self._visual_index is not None and self._metadata_index is not None
            visual_scores, visual_indices = self._visual_index.search(query_vector, max(top_k * 4, 8))
            metadata_scores, metadata_indices = self._metadata_index.search(query_vector, max(top_k * 4, 8))

            merged: dict[int, dict[str, float]] = {}
            for score, index in zip(visual_scores.tolist(), visual_indices.tolist()):
                if index < 0:
                    continue
                merged.setdefault(int(index), {"visual_score": 0.0, "metadata_score": 0.0})
                merged[int(index)]["visual_score"] = float(score)
            for score, index in zip(metadata_scores.tolist(), metadata_indices.tolist()):
                if index < 0:
                    continue
                merged.setdefault(int(index), {"visual_score": 0.0, "metadata_score": 0.0})
                merged[int(index)]["metadata_score"] = float(score)

            ranked = sorted(
                merged.items(),
                key=lambda item: (
                    item[1]["visual_score"] * self._visual_weight
                    + item[1]["metadata_score"] * self._metadata_weight
                ),
                reverse=True,
            )[: max(1, top_k)]

            hits: list[SearchHit] = []
            for rank, (index_position, scores) in enumerate(ranked, start=1):
                combined = (
                    scores["visual_score"] * self._visual_weight
                    + scores["metadata_score"] * self._metadata_weight
                )
                hits.append(
                    SearchHit(
                        rank=rank,
                        score=float(combined),
                        visual_score=float(scores["visual_score"]),
                        metadata_score=float(scores["metadata_score"]),
                        record=self._records[index_position],
                    )
                )
            return hits

    def close(self) -> None:
        if not self._enabled:
            return
        with self._data_lock:
            if self._entries:
                self._save_bundle_locked()

    def _load_existing_bundle(self) -> None:
        config_path = self._index_dir / "config.json"
        entries_path = self._index_dir / "entries.jsonl"
        visual_vectors_path = self._index_dir / "visual_vectors.npy"
        metadata_vectors_path = self._index_dir / "metadata_vectors.npy"
        visual_index_path = self._index_dir / "visual.index"
        metadata_index_path = self._index_dir / "metadata.index"

        assert self._encoder is not None
        dimension = self._encoder.dimension
        if config_path.exists() and entries_path.exists():
            self._entries = [
                MemoryIndexEntry.from_json(json.loads(line))
                for line in entries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._records = [
                MemoryIndexRecord(
                    timestamp_ns=entry.timestamp_ns,
                    index_position=index_position,
                    image_path=entry.image_path,
                    metadata_text=entry.metadata_text,
                    embedding_model_name=self._model_name,
                    embedding_pretrained=self._pretrained,
                    embedding_device=self._encoder.device,
                    index_backend="faiss" if visual_index_path.exists() else "numpy",
                    yolo_class_names=entry.yolo_class_names,
                )
                for index_position, entry in enumerate(self._entries)
            ]
            self._timestamp_to_index = {
                entry.timestamp_ns: index_position for index_position, entry in enumerate(self._entries)
            }

        self._visual_index, self._visual_vectors = VectorIndex.load(
            dimension=dimension,
            index_path=visual_index_path,
            vectors_path=visual_vectors_path,
        )
        self._metadata_index, self._metadata_vectors = VectorIndex.load(
            dimension=dimension,
            index_path=metadata_index_path,
            vectors_path=metadata_vectors_path,
        )
        self._backend = self._visual_index.backend
        print(
            f"[memory] ready model={self._model_name} pretrained={self._pretrained} "
            f"device={self._encoder.device} backend={self._backend} entries={len(self._entries)}"
        )

    def _persist_frame_image(self, processed_packet: ProcessedPacket) -> str:
        timestamp_ns = int(processed_packet.packet.timestamp_ns)
        image_path = self._frames_dir / f"frame_{timestamp_ns}.jpg"
        cv2.imwrite(str(image_path), processed_packet.rectified_frame.image_bgr)
        return str(image_path)

    def _save_bundle_locked(self) -> None:
        assert self._visual_index is not None and self._metadata_index is not None
        config = {
            "model_name": self._model_name,
            "pretrained": self._pretrained,
            "device": self._encoder.device if self._encoder is not None else self._device,
            "entry_count": len(self._entries),
            "backend": self._backend,
            "visual_weight": self._visual_weight,
            "metadata_weight": self._metadata_weight,
        }
        (self._index_dir / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        with (self._index_dir / "entries.jsonl").open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(json.dumps(entry.to_json(), ensure_ascii=True))
                handle.write("\n")

        self._visual_index.save(
            self._index_dir / "visual.index",
            self._index_dir / "visual_vectors.npy",
            self._visual_vectors,
        )
        self._metadata_index.save(
            self._index_dir / "metadata.index",
            self._index_dir / "metadata_vectors.npy",
            self._metadata_vectors,
        )


def _append_vector(existing_vectors: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector = _normalize_rows(vector)[0]
    if existing_vectors.size == 0:
        return vector.reshape(1, -1).astype(np.float32, copy=False)
    return np.vstack([existing_vectors, vector.reshape(1, -1)]).astype(np.float32, copy=False)


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized != "auto":
        return normalized
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_memory_entry(processed_packet: ProcessedPacket, *, image_path: str) -> MemoryIndexEntry:
    packet = processed_packet.packet
    yolo_output = packet.yolo_output if packet.HasField("yolo_output") else None
    class_names = tuple(
        sorted({str(detection.class_name).strip() for detection in yolo_output.detections if detection.class_name})
    ) if yolo_output is not None else ()
    metadata_lines = [
        f"timestamp_ns: {packet.timestamp_ns}",
        f"tracking_state: {processed_packet.pose.tracking_state}",
        f"pose_valid: {processed_packet.pose.pose_valid}",
        f"detections: {processed_packet.yolo_detection_count}",
    ]
    if class_names:
        metadata_lines.append("objects: " + ", ".join(class_names))
    if processed_packet.pose.camera_position_xyz is not None:
        x, y, z = processed_packet.pose.camera_position_xyz
        metadata_lines.append(f"camera_xyz_m: {x:.3f}, {y:.3f}, {z:.3f}")
    if processed_packet.map_snapshot is not None:
        metadata_lines.append(f"map_revision: {processed_packet.map_snapshot.revision}")

    return MemoryIndexEntry(
        timestamp_ns=int(packet.timestamp_ns),
        image_path=image_path,
        metadata_text="\n".join(metadata_lines),
        tracking_state=processed_packet.pose.tracking_state,
        pose_valid=processed_packet.pose.pose_valid,
        client_peer=processed_packet.envelope.client_peer,
        camera_translation_xyz=processed_packet.pose.camera_position_xyz,
        yolo_class_names=class_names,
    )
