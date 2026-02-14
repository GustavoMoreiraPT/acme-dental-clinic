"""Thread-safe in-memory LRU cache with a configurable byte-size ceiling.

Design decisions
────────────────
• **OrderedDict** for O(1) LRU eviction and promotion.
• **Size tracking** via ``json.dumps`` byte length — accurate for the JSON-
  serialisable dicts returned by the Calendly API.
• **threading.Lock** for thread safety (FastAPI can handle concurrent
  requests on the same process).
• **Prefix-based invalidation** so a single write operation can clear all
  related entries (e.g. every ``events_by_email:*`` key).
• Purely ephemeral — data is lost on process restart, which is acceptable
  for this use-case.

Usage in CalendlyClient
───────────────────────
>>> cache = LRUCache(max_bytes=20 * 1024 * 1024)  # 20 MB
>>> cache.put("events_by_email:alice@example.com", [event1, event2])
>>> cache.get("events_by_email:alice@example.com")
[event1, event2]
>>> cache.invalidate("events_by_email:alice@example.com")
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# Default ceiling: 20 MB
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


class LRUCache:
    """Least-Recently-Used cache bounded by total estimated byte size."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._current_bytes = 0
        # key → (value, estimated_size_bytes)
        self._store: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._lock = threading.Lock()

    # ── Size estimation ──────────────────────────────────────────────

    @staticmethod
    def _estimate_bytes(value: Any) -> int:
        """Return the estimated in-memory size of *value* in bytes.

        Uses ``json.dumps`` length for JSON-serialisable objects (which is
        what the Calendly API returns) and falls back to ``str()`` length
        for anything else.  This is a *lower-bound* estimate but good
        enough for cache-sizing purposes.
        """
        try:
            return len(json.dumps(value, default=str).encode("utf-8"))
        except (TypeError, ValueError, OverflowError):
            return len(str(value).encode("utf-8"))

    # ── Core operations ──────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return the cached value (promoting it to MRU) or ``None``."""
        with self._lock:
            if key not in self._store:
                return None
            # Promote to most-recently-used
            self._store.move_to_end(key)
            value, _ = self._store[key]
            return value

    def put(self, key: str, value: Any) -> None:
        """Insert or overwrite *key*.  Evicts LRU entries if needed."""
        size = self._estimate_bytes(value)

        # Don't cache if a single entry exceeds the limit
        if size > self._max_bytes:
            logger.debug(
                "Cache: skipping key %s (size %d > max %d)",
                key, size, self._max_bytes,
            )
            return

        with self._lock:
            # Remove old entry for the same key
            if key in self._store:
                _, old_size = self._store.pop(key)
                self._current_bytes -= old_size

            # Evict LRU entries until there is room
            while self._current_bytes + size > self._max_bytes and self._store:
                evicted_key, (_, evicted_size) = self._store.popitem(last=False)
                self._current_bytes -= evicted_size
                logger.debug("Cache: evicted %s (%d bytes)", evicted_key, evicted_size)

            self._store[key] = (value, size)
            self._current_bytes += size

    def invalidate(self, key: str) -> bool:
        """Remove a single key.  Returns ``True`` if the key existed."""
        with self._lock:
            if key in self._store:
                _, size = self._store.pop(key)
                self._current_bytes -= size
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove every key that starts with *prefix*.  Returns count removed."""
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for key in keys:
                _, size = self._store.pop(key)
                self._current_bytes -= size
            return len(keys)

    def clear(self) -> None:
        """Drop all entries."""
        with self._lock:
            self._store.clear()
            self._current_bytes = 0

    # ── Introspection ────────────────────────────────────────────────

    @property
    def current_bytes(self) -> int:
        """Total estimated bytes currently stored."""
        return self._current_bytes

    @property
    def entry_count(self) -> int:
        """Number of entries currently stored."""
        return len(self._store)

    def has(self, key: str) -> bool:
        """Check if a key is present *without* promoting it."""
        return key in self._store

    def find_key_containing_value(
        self, prefix: str, needle: str,
    ) -> str | None:
        """Return the first key starting with *prefix* whose JSON-serialised
        value contains *needle*.  Useful for reverse-lookups, e.g. finding
        which email owns a given ``event_uuid``.

        Does *not* promote the entry (read-only scan).
        """
        with self._lock:
            for key in self._store:
                if not key.startswith(prefix):
                    continue
                value, _ = self._store[key]
                try:
                    if needle in json.dumps(value, default=str):
                        return key
                except (TypeError, ValueError):
                    continue
        return None
