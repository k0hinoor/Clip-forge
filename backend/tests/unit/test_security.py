from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.core.errors import AppError, ErrorCode
from clipforge.security.files import (
    clip_filename,
    safe_join,
    sanitize_display_filename,
    slugify,
    validate_extension,
    validate_signature,
    validate_storage_key,
)
from clipforge.security.passwords import hash_password, verify_password
from clipforge.security.ratelimit import RateLimiter
from clipforge.security.tokens import hash_token, new_token, sign_resource, verify_resource_signature


def test_password_hashing():
    h = hash_password("correct-horse-1")
    assert h != "correct-horse-1" and h.startswith("$argon2")
    assert verify_password(h, "correct-horse-1")
    assert not verify_password(h, "wrong")
    assert not verify_password(None, "x")


def test_tokens_are_hashed():
    t = new_token()
    assert len(t) >= 32 and hash_token(t) != t and hash_token(t) == hash_token(t)


def test_signed_urls():
    exp, sig = sign_resource("s" * 32, "clip", "abc", "download:r1", 60, now=1000)
    assert verify_resource_signature("s" * 32, "clip", "abc", "download:r1", exp, sig, now=1010)
    assert not verify_resource_signature("s" * 32, "clip", "abc", "download:r2", exp, sig, now=1010)
    assert not verify_resource_signature("s" * 32, "clip", "abc", "download:r1", exp, sig, now=exp + 1)
    assert not verify_resource_signature("x" * 32, "clip", "abc", "download:r1", exp, sig, now=1010)


def test_filename_sanitising_and_extensions():
    assert "/" not in sanitize_display_filename("../../etc/passwd.mp4")
    assert validate_extension("Talk.MP4", [".mp4", ".mov"]) == ".mp4"
    with pytest.raises(AppError) as exc:
        validate_extension("evil.exe", [".mp4"])
    assert exc.value.code == ErrorCode.UNSUPPORTED_FORMAT


def test_magic_bytes():
    mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20
    assert validate_signature(mp4, ".mp4")
    mkv = b"\x1a\x45\xdf\xa3" + b"\x00" * 30
    assert validate_signature(mkv, ".webm") or validate_signature(mkv, ".mkv")
    with pytest.raises(AppError):
        validate_signature(b"MZ\x90\x00" + b"\x00" * 30, ".mp4")


def test_path_traversal_blocked(tmp_path: Path):
    with pytest.raises((AppError, ValueError)):
        validate_storage_key("../secrets")
    with pytest.raises((AppError, ValueError)):
        safe_join(tmp_path, "uploads/../../x")
    assert safe_join(tmp_path, "uploads/u/a.mp4").is_relative_to(tmp_path)


def test_output_filename_format():
    name = clip_filename("job-1", 3, slugify("Why Most People FAIL!!"))
    assert name.startswith("clip_job-1_") and name.endswith(".mp4")
    assert "why-most-people-fail" in name


def test_rate_limiter():
    rl = RateLimiter()
    assert all(rl.hit("k", 3, 60)[0] for _ in range(3))
    allowed, retry = rl.hit("k", 3, 60)
    assert not allowed and retry > 0
