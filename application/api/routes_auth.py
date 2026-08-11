"""Session auth — local User ID only."""

from __future__ import annotations

import base64
import json
import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

try:
    from application import utils
except ImportError:
    import utils  # type: ignore

logger = logging.getLogger("routes_auth")

router = APIRouter(prefix="/api/session", tags=["session"])

SESSION_COOKIE = "agent_user_id"
_SIGNED_COOKIE_PREFIX = "v1."
_MAX_PLAIN_USER_ID_LEN = 128


class SessionRequest(BaseModel):
    user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Local user id",
    )


class SessionResponse(BaseModel):
    user_id: str
    name: str | None = None
    picture: str | None = None
    llm_gateway_ready: bool = False
    knowledge_graph_enabled: bool = False
    graph_pattern: str = "pattern1"


class SessionSettingsPatch(BaseModel):
    knowledge_graph_enabled: bool | None = None
    graph_pattern: str | None = None


def _env_bypass_flag() -> bool:
    return os.environ.get("ALLOW_LOCAL_AUTH_BYPASS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_loopback_request(request: Request) -> bool:
    host = (request.headers.get("host") or "").split("%")[0]
    hostname = host.split(":")[0].strip().lower().strip("[]")
    return hostname in {"localhost", "127.0.0.1", "::1"}


def local_auth_bypass_enabled(request: Request) -> bool:
    """Always allow local User ID login for rag-multimodal."""
    return True


def _llm_gateway_ready() -> bool:
    cfg = utils.load_config()
    url = (cfg.get("llm_gateway_url") or "").strip()
    key = (cfg.get("llm_gateway_key") or "").strip()
    return bool(url and key)


def _set_user_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=user_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 365,
    )


def _uid_from_signed_cookie(raw: str) -> str | None:
    parts = raw.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        return None
    try:
        padding = "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + padding))
    except Exception:
        logger.warning("Ignoring undecodable signed session cookie")
        return None
    uid = (payload.get("uid") or "").strip()
    if not uid or len(uid) > _MAX_PLAIN_USER_ID_LEN:
        return None
    return uid


def resolve_cookie_user_id(raw: str | None) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    if value.startswith(_SIGNED_COOKIE_PREFIX):
        return _uid_from_signed_cookie(value)
    if len(value) > _MAX_PLAIN_USER_ID_LEN:
        logger.warning("Ignoring oversized session cookie (%d chars)", len(value))
        return None
    return value


def get_optional_user_id(request: Request) -> str | None:
    return resolve_cookie_user_id(request.cookies.get(SESSION_COOKIE))


def _session_response(
    user_id: str,
    *,
    name: str | None = None,
    picture: str | None = None,
    llm_gateway_ready: bool | None = None,
) -> SessionResponse:
    return SessionResponse(
        user_id=user_id,
        name=name,
        picture=picture,
        llm_gateway_ready=(
            _llm_gateway_ready() if llm_gateway_ready is None else llm_gateway_ready
        ),
        knowledge_graph_enabled=False,
        graph_pattern="pattern1",
    )


@router.post("", response_model=SessionResponse)
def set_session(body: SessionRequest, request: Request, response: Response) -> SessionResponse:
    local_user_id = (body.user_id or "").strip()
    if not local_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    gateway_ready = _llm_gateway_ready()
    _set_user_cookie(response, local_user_id)
    utils.ensure_user_artifacts_dir(local_user_id)
    utils.ensure_user_skills_dir(local_user_id)
    utils.ensure_user_skills_list(local_user_id)
    logger.info(
        "Local session login: %s (llm_gateway_ready=%s)",
        local_user_id,
        gateway_ready,
    )
    return _session_response(local_user_id, llm_gateway_ready=gateway_ready)


@router.get("", response_model=SessionResponse | None)
def get_session(request: Request, response: Response) -> SessionResponse | None:
    raw_cookie = (request.cookies.get(SESSION_COOKIE) or "").strip()
    user_id = resolve_cookie_user_id(raw_cookie)
    if not user_id:
        return None
    if raw_cookie.startswith(_SIGNED_COOKIE_PREFIX) and raw_cookie != user_id:
        _set_user_cookie(response, user_id)
        logger.info("Normalized signed session cookie to user_id=%s", user_id)
    utils.ensure_user_artifacts_dir(user_id)
    utils.ensure_user_skills_dir(user_id)
    utils.ensure_user_skills_list(user_id)
    return _session_response(user_id)


@router.patch("/settings", response_model=SessionResponse)
def patch_session_settings(
    body: SessionSettingsPatch, request: Request
) -> SessionResponse:
    """No-op settings patch for UI compatibility (Knowledge Graph not used)."""
    user_id = require_user_id(request)
    _ = body
    return _session_response(user_id)


@router.delete("", status_code=204, response_model=None)
def clear_session(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, samesite="lax")


def require_user_id(request: Request) -> str:
    user_id = get_optional_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User session required")
    return user_id
