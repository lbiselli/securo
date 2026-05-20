"""Bearer-token verification for MCP requests.

Two auth modes, picked by `AGENTS_MCP_AUTH_PROVIDER`:

* `internal` (default) — HS256 JWTs minted by the in-process agent
  runtime (`AGENTS_MCP_JWT_SECRET`). Keeps the built-in AI Agents
  working without changes.

* `workos` — JWTs issued by WorkOS AuthKit. Signature is verified
  against the AuthKit JWKS; issuer/audience are checked; the caller is
  resolved to a Securo `User` by matching the `email` claim. Used so
  external MCP clients (e.g. Claude) can authenticate over OAuth 2.1.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from sqlalchemy import select

from app.agents.config import get_agent_settings
from app.core.database import async_session_maker
from app.models.user import User


JWT_ISSUER = "securo-backend"
JWT_AUDIENCE = "securo-mcp"
JWT_ALGO = "HS256"

# Soft JWKS cache. Key rotation on AuthKit is rare; 1h keeps validation
# fast and we force-refresh on unknown kid anyway.
_JWKS_CACHE: dict[str, tuple[float, dict]] = {}
_JWKS_TTL_SECONDS = 3600


@dataclass
class CallContext:
    user_id: uuid.UUID
    conversation_id: Optional[uuid.UUID] = None
    agent_id: Optional[uuid.UUID] = None


def _settings():
    return get_agent_settings()


async def verify_request(request: Request) -> CallContext:
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = auth.split(" ", 1)[1].strip()

    provider = (_settings().mcp_auth_provider or "internal").lower()
    if provider == "workos":
        return await _verify_workos(token)
    return _verify_internal(token)


def _verify_internal(token: str) -> CallContext:
    try:
        payload = jwt.decode(
            token,
            _settings().mcp_jwt_secret,
            algorithms=[JWT_ALGO],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {exc}") from exc

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing subject")
    try:
        user_id = uuid.UUID(sub)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad subject") from exc

    conv_raw = payload.get("conv_id")
    agent_raw = payload.get("agent_id")
    return CallContext(
        user_id=user_id,
        conversation_id=uuid.UUID(conv_raw) if conv_raw else None,
        agent_id=uuid.UUID(agent_raw) if agent_raw else None,
    )


async def _fetch_jwks(url: str) -> dict:
    now = time.time()
    cached = _JWKS_CACHE.get(url)
    if cached and now - cached[0] < _JWKS_TTL_SECONDS:
        return cached[1]
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    _JWKS_CACHE[url] = (now, data)
    return data


async def _verify_workos(token: str) -> CallContext:
    s = _settings()
    if not s.workos_jwks_url or not s.workos_issuer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WorkOS auth not configured (AGENTS_WORKOS_JWKS_URL / AGENTS_WORKOS_ISSUER missing)",
        )

    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token header: {exc}") from exc

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token missing kid")

    try:
        jwks = await _fetch_jwks(s.workos_jwks_url)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"jwks fetch failed: {exc}") from exc

    key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if key is None:
        # Force-refresh in case keys rotated between cache write and now.
        _JWKS_CACHE.pop(s.workos_jwks_url, None)
        try:
            jwks = await _fetch_jwks(s.workos_jwks_url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"jwks refresh failed: {exc}") from exc
        key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unknown signing key")

    decode_kwargs: dict = {
        "algorithms": [key.get("alg", "RS256")],
        "issuer": s.workos_issuer,
    }
    if s.workos_audience:
        decode_kwargs["audience"] = s.workos_audience
    else:
        decode_kwargs["options"] = {"verify_aud": False}

    try:
        payload = jwt.decode(token, key, **decode_kwargs)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {exc}") from exc

    email = payload.get("email") or payload.get("preferred_username")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token missing email claim")
    email = email.strip().lower()

    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"no Securo user with email {email}",
        )

    return CallContext(user_id=user.id)
