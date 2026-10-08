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
