from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from server.services.video_rag_service import VideoRagResult, VideoRagService


class SceneQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=10)


class MemoryHitModel(BaseModel):
    rank: int
    score: float
    timestamp_ns: int
    image_path: str
    metadata_text: str


class SceneQueryResponse(BaseModel):
    question: str
    retrieval_query: str
    answer: str
    latest_timestamp_ns: int | None
    latest_frame_path: str | None
    hits: list[MemoryHitModel]


class VlmHttpService:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        video_rag_service: VideoRagService,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._video_rag_service = video_rag_service
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        app = self._build_app()
        config = uvicorn.Config(app=app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="vlm-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Machine Perception VLM API", version="0.1.0")

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/query/current-scene", response_model=SceneQueryResponse)
        def query_current_scene(request: SceneQueryRequest) -> SceneQueryResponse:
            return _to_response(self._execute_query("current", request))

        @app.post("/query/video-rag", response_model=SceneQueryResponse)
        def query_video_rag(request: SceneQueryRequest) -> SceneQueryResponse:
            return _to_response(self._execute_query("rag", request))

        return app

    def _execute_query(self, mode: str, request: SceneQueryRequest) -> VideoRagResult:
        try:
            if mode == "current":
                return self._video_rag_service.query_current_scene(
                    question=request.question,
                    top_k=request.top_k,
                )
            return self._video_rag_service.query_video_rag(
                question=request.question,
                top_k=request.top_k,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def _to_response(result: VideoRagResult) -> SceneQueryResponse:
    return SceneQueryResponse(
        question=result.question,
        retrieval_query=result.retrieval_query,
        answer=result.answer,
        latest_timestamp_ns=result.latest_timestamp_ns,
        latest_frame_path=result.latest_frame_path,
        hits=[
            MemoryHitModel(
                rank=hit.rank,
                score=hit.score,
                timestamp_ns=hit.record.timestamp_ns,
                image_path=hit.record.image_path,
                metadata_text=hit.record.metadata_text,
            )
            for hit in result.hits
        ],
    )
