"""
Supabase JWT verification.

Supabase issues HS256-signed access tokens signed with the project's
JWT secret (Settings > API > JWT Secret) for most projects. We verify
signature, expiry, and audience ("authenticated").

If a project has been migrated to asymmetric JWT signing keys (JWKS),
set SUPABASE_JWT_SECRET to empty and provide SUPABASE_URL; the JWKS
verification path below is used instead (fetches
`${SUPABASE_URL}/auth/v1/.well-known/jwks.json` and caches it).
"""
import logging
import time
from typing import Optional

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

from .config import settings

logger = logging.getLogger("yofilter.eraser.auth")

_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not settings.supabase_url:
            raise HTTPException(status_code=500, detail="Auth is not configured (missing SUPABASE_URL).")
        jwks_url = settings.supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url)
    return _jwks_client


def _decode_token(token: str) -> dict:
    if settings.supabase_jwt_secret:
        return jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
        )
    # Fallback: asymmetric JWKS-based verification.
    client = _get_jwks_client()
    signing_key = client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["ES256", "RS256"],
        audience=settings.jwt_audience,
    )


async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    verify_mode = "hs256-shared-secret" if settings.supabase_jwt_secret else "jwks-asymmetric"

    if not authorization or not authorization.lower().startswith("bearer "):
        logger.warning("Auth rejected: missing bearer token (mode=%s)", verify_mode)
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        logger.warning("Auth rejected: empty bearer token (mode=%s)", verify_mode)
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    try:
        payload = _decode_token(token)
    except jwt.ExpiredSignatureError:
        logger.warning("Auth rejected: token expired (mode=%s)", verify_mode)
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.PyJWTError as exc:
        # Very commonly caused by an unresolved placeholder SUPABASE_JWT_SECRET
        # (or a SUPABASE_URL that doesn't match the project that issued the
        # token) in .env -- every request 401s here and the frontend silently
        # falls back to the offline heuristic eraser. Log loudly so this is
        # never mistaken for "the AI eraser isn't working".
        logger.warning(
            "Auth rejected: %s (mode=%s) -- check SUPABASE_JWT_SECRET / SUPABASE_URL in backend/.env",
            exc, verify_mode,
        )
        raise HTTPException(status_code=401, detail="Invalid token.")

    sub = payload.get("sub")
    if not sub:
        logger.warning("Auth rejected: token missing subject (mode=%s)", verify_mode)
        raise HTTPException(status_code=401, detail="Token missing subject.")

    return {"sub": sub, "email": payload.get("email"), "exp": payload.get("exp", int(time.time()))}
