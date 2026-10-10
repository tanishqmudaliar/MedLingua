from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext

from .config import get_settings

_legacy_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hashes password using native bcrypt."""
    pw_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verifies password using bcrypt for modern hashes and pbkdf2 for legacy hashes."""
    if password_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
        except Exception:
            return False
    return _legacy_context.verify(password, password_hash)


def create_access_token(user_id: UUID) -> str:
    settings = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode({"sub": str(user_id), "exp": expires}, settings.jwt_secret_key, algorithm="HS256")


def decode_user_id(token: str) -> UUID | None:
    try:
        payload = jwt.decode(token, get_settings().jwt_secret_key, algorithms=["HS256"])
        return UUID(payload["sub"])
    except (InvalidTokenError, KeyError, TypeError, ValueError):
        return None
