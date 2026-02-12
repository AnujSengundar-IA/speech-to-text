from fastapi import WebSocket

from stt_service.config import AppConfig


def is_origin_allowed(ws: WebSocket, cfg: AppConfig) -> bool:
    if not cfg.allowed_origins:
        return True
    origin = ws.headers.get("origin", "")
    return origin in cfg.allowed_origins


def is_authenticated(ws: WebSocket, cfg: AppConfig) -> bool:
    if not cfg.ws_auth_token:
        return True
    token = ws.query_params.get("token") or ws.headers.get("x-api-key", "")
    return token == cfg.ws_auth_token
