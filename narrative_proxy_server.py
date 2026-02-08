from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

import narrative_engine as ne


TOKEN_HASH_ENV = "NARRATIVE_TOKEN_HASHES"  # comma-separated SHA256 hex digests


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_allowed_hashes() -> set[str]:
    raw = (os.getenv(TOKEN_HASH_ENV) or "").strip()
    if not raw:
        return set()
    out: set[str] = set()
    for part in raw.split(","):
        h = part.strip().lower()
        if h:
            out.add(h)
    return out


def _require_auth(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization must be Bearer <token>.")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty Bearer token.")

    allowed = _load_allowed_hashes()
    if not allowed:
        raise HTTPException(
            status_code=500,
            detail=f"Server not configured: set {TOKEN_HASH_ENV} to allowed token hashes.",
        )

    token_hash = _sha256_hex(token)
    if token_hash not in allowed:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return token_hash


@dataclass(frozen=True)
class NarrativeRequest:
    data_digest: Mapping[str, Any]
    model: str | None = None


app = FastAPI(title="Schedule Analytics Narrative Proxy", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/narrative")
def generate(
    payload: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _require_auth(authorization)

    if "data_digest" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'data_digest' in request body.")

    data_digest = payload["data_digest"]
    if not isinstance(data_digest, dict):
        raise HTTPException(status_code=400, detail="'data_digest' must be a JSON object.")

    model = payload.get("model") or "gemini-2.5-flash"

    try:
        text = ne.generate_narrative(data_digest, model=str(model))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse({"text": text, "model": str(model)})


def create_user_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    return token, _sha256_hex(token)


def _main() -> int:
    p = argparse.ArgumentParser(description="Narrative proxy server (token-auth) utilities.")
    p.add_argument("--create-token", action="store_true", help="Create a new user token and print token + hash.")
    args = p.parse_args()

    if args.create_token:
        token, token_hash = create_user_token()
        print("USER_TOKEN=" + token)
        print("TOKEN_SHA256=" + token_hash)
        print(f"Add TOKEN_SHA256 to {TOKEN_HASH_ENV} on the server (comma-separated).")
        return 0

    print("This module provides a FastAPI app. Run with uvicorn, e.g.:")
    print("  uvicorn narrative_proxy_server:app --host 0.0.0.0 --port 8080")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

