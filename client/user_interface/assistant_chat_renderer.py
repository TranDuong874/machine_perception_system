from __future__ import annotations

from dataclasses import dataclass
import textwrap
import threading

import cv2
import numpy as np

from transport import VlmHttpClient, VlmQueryResult


WINDOW_WIDTH = 720
WINDOW_HEIGHT = 680
HEADER_HEIGHT = 72
FOOTER_HEIGHT = 140
PADDING_X = 18
LINE_HEIGHT = 22
INPUT_RECT = (18, WINDOW_HEIGHT - 100, WINDOW_WIDTH - 36, 54)


@dataclass
class ChatMessage:
    role: str
    text: str
    pending: bool = False
    request_id: int | None = None


class AssistantChatRenderer:
    def __init__(
        self,
        *,
        client: VlmHttpClient,
        top_k: int,
        default_mode: str,
        window_name: str = "Assistant Chat",
    ) -> None:
        self._client = client
        self._top_k = max(1, int(top_k))
        self._mode = default_mode if default_mode in {"current", "rag"} else "current"
        self._window_name = window_name

        self._lock = threading.Lock()
        self._messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                text=(
                    "Click the input box, type a question, then press Enter. "
                    "Press Tab to switch between current-scene and video-rag mode."
                ),
            )
        ]
        self._input_text = ""
        self._input_active = False
        self._pending_request_id: int | None = None
        self._request_counter = 0
        self._window_initialized = False

    def render(self, *, latest_timestamp_ns: int | None) -> None:
        if not self._window_initialized:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
            cv2.setMouseCallback(self._window_name, self._on_mouse)
            self._window_initialized = True

        frame = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), 18, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (WINDOW_WIDTH, HEADER_HEIGHT), (32, 32, 36), thickness=-1)
        cv2.putText(
            frame,
            "Assistant Chat",
            (PADDING_X, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        status_line = (
            f"mode={self._mode}  top_k={self._top_k}  endpoint={self._client.endpoint}  "
            f"latest_ts={latest_timestamp_ns if latest_timestamp_ns is not None else 'none'}"
        )
        cv2.putText(
            frame,
            status_line,
            (PADDING_X, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (188, 188, 188),
            1,
            cv2.LINE_AA,
        )

        with self._lock:
            messages = list(self._messages)
            input_text = self._input_text
            input_active = self._input_active
            pending = self._pending_request_id is not None

        history_top = HEADER_HEIGHT + 14
        history_bottom = WINDOW_HEIGHT - FOOTER_HEIGHT - 12
        cv2.rectangle(
            frame,
            (PADDING_X, history_top),
            (WINDOW_WIDTH - PADDING_X, history_bottom),
            (26, 26, 30),
            thickness=-1,
        )
        self._draw_history(
            frame,
            messages=messages,
            top=history_top + 14,
            bottom=history_bottom - 10,
        )

        footer_top = WINDOW_HEIGHT - FOOTER_HEIGHT
        cv2.rectangle(frame, (0, footer_top), (WINDOW_WIDTH, WINDOW_HEIGHT), (28, 28, 34), thickness=-1)
        help_line = "Enter=send  Tab=toggle mode  Esc=clear input  q=quit from preview window"
        cv2.putText(
            frame,
            help_line,
            (PADDING_X, footer_top + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        if pending:
            cv2.putText(
                frame,
                "assistant status: waiting for response",
                (PADDING_X, footer_top + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (80, 220, 255),
                1,
                cv2.LINE_AA,
            )

        self._draw_input_box(frame, input_text=input_text, active=input_active)
        cv2.imshow(self._window_name, frame)

    def handle_key(self, key: int) -> bool:
        if key < 0:
            return False
        if key in {9}:
            with self._lock:
                if not self._input_active:
                    return False
                self._mode = "rag" if self._mode == "current" else "current"
            return True
        if key in {27}:
            with self._lock:
                if not self._input_active:
                    return False
                self._input_text = ""
            return True
        if key in {8, 127}:
            with self._lock:
                if not self._input_active:
                    return False
                self._input_text = self._input_text[:-1]
            return True
        if key in {10, 13}:
            return self._submit_current_input()

        if 32 <= key <= 126:
            with self._lock:
                if not self._input_active:
                    return False
                self._input_text += chr(key)
            return True
        return False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass

    def _submit_current_input(self) -> bool:
        with self._lock:
            if not self._input_active:
                return False
            question = self._input_text.strip()
            if not question:
                return True
            if self._pending_request_id is not None:
                self._messages.append(
                    ChatMessage(role="system", text="Assistant is busy. Wait for the current response.")
                )
                self._trim_messages_locked()
                self._input_text = ""
                return True
            self._request_counter += 1
            request_id = self._request_counter
            mode = self._mode
            self._pending_request_id = request_id
            self._messages.append(ChatMessage(role="user", text=question))
            self._messages.append(
                ChatMessage(role="assistant", text="thinking...", pending=True, request_id=request_id)
            )
            self._trim_messages_locked()
            self._input_text = ""

        worker = threading.Thread(
            target=self._run_query,
            args=(request_id, mode, question),
            name=f"assistant-chat-{request_id}",
            daemon=True,
        )
        worker.start()
        return True

    def _run_query(self, request_id: int, mode: str, question: str) -> None:
        try:
            result = self._client.query(mode=mode, question=question, top_k=self._top_k)
            answer_text = _format_result(mode=mode, result=result)
        except Exception as exc:
            answer_text = f"request failed: {exc}"
        with self._lock:
            for message in self._messages:
                if message.request_id == request_id:
                    message.text = answer_text
                    message.pending = False
                    break
            self._pending_request_id = None
            self._trim_messages_locked()

    def _draw_history(
        self,
        frame: np.ndarray,
        *,
        messages: list[ChatMessage],
        top: int,
        bottom: int,
    ) -> None:
        rendered_lines: list[tuple[str, str, bool]] = []
        for message in messages:
            prefix = f"{message.role}: "
            wrapped = _wrap_text(message.text, width=72)
            if not wrapped:
                wrapped = [""]
            rendered_lines.append((prefix + wrapped[0], message.role, message.pending))
            for line in wrapped[1:]:
                rendered_lines.append((" " * len(prefix) + line, message.role, message.pending))
            rendered_lines.append(("", message.role, False))

        capacity = max(1, (bottom - top) // LINE_HEIGHT)
        rendered_lines = rendered_lines[-capacity:]
        y = top
        for line, role, pending in rendered_lines:
            color = _role_color(role=role, pending=pending)
            cv2.putText(
                frame,
                line,
                (PADDING_X + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            y += LINE_HEIGHT

    def _draw_input_box(self, frame: np.ndarray, *, input_text: str, active: bool) -> None:
        x, y, width, height = INPUT_RECT
        border_color = (76, 220, 255) if active else (120, 120, 128)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (46, 46, 52), thickness=-1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, thickness=2)
        prompt = "> " + input_text
        display_text = prompt[-84:]
        if active:
            display_text += "_"
        cv2.putText(
            frame,
            display_text,
            (x + 12, y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (244, 244, 244),
            1,
            cv2.LINE_AA,
        )
        hint = "Click here to type"
        cv2.putText(
            frame,
            hint,
            (x + 12, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rect_x, rect_y, rect_w, rect_h = INPUT_RECT
        with self._lock:
            self._input_active = (
                rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h
            )

    def _trim_messages_locked(self) -> None:
        if len(self._messages) <= 30:
            return
        pending_messages = [message for message in self._messages if message.pending]
        retained = self._messages[-28:]
        for pending in pending_messages:
            if pending not in retained:
                retained.append(pending)
        self._messages = retained[-30:]


def _format_result(*, mode: str, result: VlmQueryResult) -> str:
    summary = [result.answer.strip() or "No answer returned."]
    metadata = [f"mode={mode}", f"hits={len(result.hits)}"]
    if result.retrieval_query:
        metadata.append(f"query={result.retrieval_query}")
    if result.latest_timestamp_ns is not None:
        metadata.append(f"latest_ts={result.latest_timestamp_ns}")
    summary.append("[" + " | ".join(metadata) + "]")
    return "\n".join(summary)


def _wrap_text(text: str, *, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(raw_line, width=width, break_long_words=True)
        if not wrapped:
            lines.append("")
        else:
            lines.extend(wrapped)
    return lines


def _role_color(*, role: str, pending: bool) -> tuple[int, int, int]:
    if pending:
        return (80, 220, 255)
    if role == "user":
        return (110, 230, 110)
    if role == "assistant":
        return (240, 240, 240)
    return (180, 180, 180)
