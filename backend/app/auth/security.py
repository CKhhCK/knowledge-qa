from __future__ import annotations
"""JWT token management and password hashing."""

import hashlib
import hmac
import json
import time
from base64 import urlsafe_b64encode, urlsafe_b64decode

# Simple JWT implementation (no external dependency needed)
# For production, use python-jose or PyJWT

_JWT_SECRET = "helloagents-qa-jwt-secret-change-in-production"
_TOKEN_EXPIRE_SECONDS = 86400 * 7  # 7 days


def set_jwt_secret(secret: str) -> None:
    global _JWT_SECRET
    _JWT_SECRET = secret


def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return urlsafe_b64decode(data.encode("ascii"))


def create_token(payload: dict) -> str:
    """Create a signed JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload["iat"] = int(time.time())
    payload["exp"] = int(time.time()) + _TOKEN_EXPIRE_SECONDS

    header_b64 = _b64encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}"

    sig = hmac.new(_JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    sig_b64 = _b64encode(sig)

    return f"{signing_input}.{sig_b64}"


def verify_token(token: str) -> dict | None:
    """Verify a JWT token and return payload, or None if invalid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"

        expected_sig = hmac.new(_JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
        actual_sig = _b64decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64decode(payload_b64))

        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None

        return payload

    except Exception:
        return None


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 + salt."""
    salt = "helloagents-qa-salt"  # In production, use per-user random salt
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hmac.compare_digest(hash_password(password), hashed)
