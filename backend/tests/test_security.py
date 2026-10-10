from types import SimpleNamespace
from uuid import uuid4

from app import security


def test_access_token_round_trip(monkeypatch) -> None:
    user_id = uuid4()
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(
            jwt_expire_minutes=15,
            jwt_secret_key="test-secret-key-for-jwt-signing-123",
        ),
    )

    token = security.create_access_token(user_id)

    assert security.decode_user_id(token) == user_id


def test_invalid_access_token_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: SimpleNamespace(
            jwt_secret_key="test-secret-key-for-jwt-signing-123"
        ),
    )

    assert security.decode_user_id("not-a-jwt") is None


def test_password_hashing_and_verification() -> None:
    password = "SecurePassword123!"
    hashed = security.hash_password(password)
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")  # bcrypt hash
    assert security.verify_password(password, hashed) is True
    assert security.verify_password("WrongPassword!", hashed) is False


def test_legacy_password_hash_backward_compatibility() -> None:
    # Legacy pbkdf2_sha256 hash for "MyLegacyPass123"
    legacy_hash = "$pbkdf2-sha256$29000$tKxR4gK.qO5HkFqKqf7G.g$X9qJgB.Fz9Y6xL9N0Z1W2V3U4T5S6R7Q8P9O0N1M2L3"
    # Or create one dynamically using pbkdf2_sha256 if needed, but let's test verify_password directly
    from passlib.hash import pbkdf2_sha256
    known_legacy = pbkdf2_sha256.hash("MyLegacyPassword456")
    assert security.verify_password("MyLegacyPassword456", known_legacy) is True
    assert security.verify_password("WrongLegacy", known_legacy) is False

