from __future__ import annotations

import base64
from pathlib import Path

import cv2
import httpx
import numpy as np


class GroqVLMService:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        text_model: str,
        vision_model: str,
        timeout_s: float,
    ) -> None:
        self._api_key = api_key.strip() if api_key else ""
        self._base_url = base_url.rstrip("/")
        self._text_model = text_model
        self._vision_model = vision_model
        self._timeout_s = float(timeout_s)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def generate_search_query(self, *, question: str, latest_metadata: str) -> str:
        self._require_api_key()
        prompt = (
            "Rewrite the user question into a short retrieval query for egocentric video memory. "
            "Focus on visible objects, place cues, and recent scene context. Output one line only.\n\n"
            f"Latest scene metadata:\n{latest_metadata}\n\n"
            f"Question: {question}"
        )
        response = self._post_chat_completion(
            model=self._text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You create concise retrieval queries for visual memory search.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=64,
        )
        return _extract_text_response(response).strip()

    def answer_question(
        self,
        *,
        question: str,
        retrieval_query: str,
        context_blocks: list[str],
        images_bgr: list[np.ndarray],
    ) -> str:
        self._require_api_key()
        message_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "Answer the user's question using the provided recent-scene context and retrieved memory. "
                    "If context is weak, say so briefly. Mention relevant timestamps when useful.\n\n"
                    f"Retrieval query: {retrieval_query}\n\n"
                    "Context:\n"
                    + ("\n\n".join(context_blocks) if context_blocks else "No retrieved memory context.")
                    + f"\n\nQuestion: {question}"
                ),
            }
        ]
        for image_bgr in images_bgr[:3]:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _to_data_uri(image_bgr),
                    },
                }
            )
        response = self._post_chat_completion(
            model=self._vision_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a vision-language assistant for egocentric video memory. "
                        "Use the images and retrieved context to answer concretely and briefly."
                    ),
                },
                {"role": "user", "content": message_content},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return _extract_text_response(response).strip()

    def _post_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            return response.json()

    def _require_api_key(self) -> None:
        if not self.enabled:
            raise RuntimeError("Groq API key is not configured")


def _extract_text_response(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Groq response did not include choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(part for part in text_parts if part)
    raise RuntimeError("Groq response did not include message content")


def _to_data_uri(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode image for Groq VLM request")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def load_image_bgr(path: str | Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image if image is not None else None
