"""Canonical, versioned configuration identity, independent of SQL storage."""

import json
import math
from hashlib import sha256
from numbers import Integral, Real
from typing import Any, overload


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                "Configuration numbers must be finite; encode infinite edges explicitly"
            )
        return int(number) if number.is_integer() else number
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _normalize(value[key]) for key in sorted(value)}
    raise TypeError("Configuration must contain only JSON-compatible values and string keys")


@overload
def normalize_configuration(configuration: dict[str, Any]) -> dict[str, Any]: ...


@overload
def normalize_configuration(configuration: None) -> None: ...


def normalize_configuration(configuration: dict[str, Any] | None) -> dict[str, Any] | None:
    """None denotes legacy/unspecified configuration, never current defaults."""
    if configuration is not None and not isinstance(configuration, dict):
        raise TypeError("Configuration must be a dictionary or None for legacy data")
    return _normalize(configuration)


def serialize_configuration(configuration: dict[str, Any] | None) -> str:
    return json.dumps(
        normalize_configuration(configuration),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def configuration_hash(configuration: dict[str, Any] | None) -> str:
    return sha256(serialize_configuration(configuration).encode("utf-8")).hexdigest()
