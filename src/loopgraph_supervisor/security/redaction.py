from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
)


class RedactionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    sensitive_keys: frozenset[str] = Field(
        default_factory=lambda: frozenset(
            {
                "authorization",
                "proxy-authorization",
                "cookie",
                "set-cookie",
                "api_key",
                "apikey",
                "access_token",
                "refresh_token",
                "password",
                "secret",
                "client_secret",
            }
        )
    )
    replacement: str = "[REDACTED]"

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    self.replacement
                    if str(key).casefold() in self.sensitive_keys
                    else self.redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        if isinstance(value, str):
            redacted = value
            for pattern in _SECRET_PATTERNS:
                if pattern.pattern.startswith("(?i)\\bBearer"):
                    redacted = pattern.sub("Bearer " + self.replacement, redacted)
                else:
                    redacted = pattern.sub(self.replacement, redacted)
            return redacted
        return value
