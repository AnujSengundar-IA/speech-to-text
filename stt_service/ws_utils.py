import json
import re
from json import JSONDecodeError
from typing import Optional

from fastapi import WebSocket


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


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
