from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.services.memory_index_service import MemoryIndexService
from server.settings.server_settings import load_server_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the server memory index.")
    parser.add_argument("query", type=str, help="Text query to search against the indexed memory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits to return.")
    args = parser.parse_args()

    config = load_server_config()
    service = MemoryIndexService(
        enabled=config.enable_memory_index,
        index_dir=config.memory_index_dir,
        model_name=config.memory_model_name,
        pretrained=config.memory_pretrained,
        device=config.memory_device,
        batch_size=config.memory_batch_size,
        save_every_n_updates=config.memory_save_every_n_updates,
        visual_weight=config.memory_visual_weight,
        metadata_weight=config.memory_metadata_weight,
    )
    if not service.preflight():
        print("Memory index is disabled or unavailable.", file=sys.stderr)
        return 1

    hits = service.query(query_text=args.query, top_k=args.top_k)
    if not hits:
        print("No hits found.")
        return 0

    for hit in hits:
        print(f"[{hit.rank}] score={hit.score:.4f} visual={hit.visual_score:.4f} metadata={hit.metadata_score:.4f}")
        print(f"timestamp_ns={hit.record.timestamp_ns}")
        print(f"image_path={hit.record.image_path}")
        print(f"objects={', '.join(hit.record.yolo_class_names) if hit.record.yolo_class_names else '-'}")
        print(hit.record.metadata_text)
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
