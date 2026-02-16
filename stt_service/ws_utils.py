import json
import re
from json import JSONDecodeError
from typing import Optional

from fastapi import WebSocket

_SPECIAL_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<pad>", re.IGNORECASE),
    re.compile(r"</?s>", re.IGNORECASE),
    re.compile(r"<\|[^|>]+?\|>"),
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sanitize_transcript_text(text: str) -> str:
    cleaned = text
    for pattern in _SPECIAL_TOKEN_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


async def send_event(ws: WebSocket, event_type: str, **payload: object) -> None:
    await ws.send_json({"type": event_type, **payload})


def parse_event_frame(text_frame: str) -> Optional[str]:
    try:
        data = json.loads(text_frame)
    except JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    event = data.get("event")
    if isinstance(event, str):
        return event
    return None
