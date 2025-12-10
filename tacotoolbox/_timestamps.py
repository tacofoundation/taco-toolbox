"""Timestamp conversion utilities."""

import datetime
from typing import TypeAlias

TimestampLike: TypeAlias = datetime.datetime | int | float


def _to_utc_microseconds(timestamp: TimestampLike) -> int:
    """Convert timestamp to microseconds since Unix epoch."""
    if isinstance(timestamp, datetime.datetime):
        if timestamp.tzinfo is None:
            timestamp_utc = timestamp.replace(tzinfo=datetime.timezone.utc)
        else:
            timestamp_utc = timestamp.astimezone(datetime.timezone.utc)
        return int(timestamp_utc.timestamp() * 1_000_000)
    else:
        return int(timestamp * 1_000_000)
