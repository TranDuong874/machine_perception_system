from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request


@dataclass(frozen=True)
class VlmQueryResult:
    question: str
    retrieval_query: str
    answer: str
    latest_timestamp_ns: int | None
    latest_frame_path: str | None
    hits: list[dict]


class VlmHttpClient:
    def __init__(self, *, host: str, port: int, timeout_s: float) -> None:
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}"

    def query(self, *, mode: str, question: str, top_k: int) -> VlmQueryResult:
        if mode not in {"current", "rag"}:
            raise ValueError(f"Unsupported VLM query mode: {mode}")
        endpoint = "current-scene" if mode == "current" else "video-rag"
        request = urllib.request.Request(
            f"{self.endpoint}/query/{endpoint}",
            data=json.dumps(
                {
                    "question": question,
                    "top_k": max(1, int(top_k)),
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM query failed ({exc.code}): {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM endpoint unavailable: {exc.reason}") from exc

        return VlmQueryResult(
            question=str(payload.get("question", question)),
            retrieval_query=str(payload.get("retrieval_query", "")),
            answer=str(payload.get("answer", "")),
            latest_timestamp_ns=payload.get("latest_timestamp_ns"),
            latest_frame_path=payload.get("latest_frame_path"),
            hits=list(payload.get("hits", [])),
        )
