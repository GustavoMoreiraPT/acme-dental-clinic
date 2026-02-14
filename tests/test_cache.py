"""Tests for the in-memory LRU cache."""

from __future__ import annotations

from src.services.cache import LRUCache

# ── Core operations ──────────────────────────────────────────────────


class TestLRUCacheBasics:
    def test_put_and_get(self):
        cache = LRUCache()
        cache.put("key1", {"name": "Alice"})
        assert cache.get("key1") == {"name": "Alice"}

    def test_get_returns_none_for_missing_key(self):
        cache = LRUCache()
        assert cache.get("nonexistent") is None

    def test_put_overwrites_existing_key(self):
        cache = LRUCache()
        cache.put("key1", "old")
        cache.put("key1", "new")
        assert cache.get("key1") == "new"
        assert cache.entry_count == 1

    def test_invalidate_removes_key(self):
        cache = LRUCache()
        cache.put("key1", "value")
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None

    def test_invalidate_returns_false_for_missing_key(self):
        cache = LRUCache()
        assert cache.invalidate("nonexistent") is False

    def test_clear_removes_all_entries(self):
        cache = LRUCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.entry_count == 0
        assert cache.current_bytes == 0

    def test_has_key(self):
        cache = LRUCache()
        cache.put("key1", "value")
        assert cache.has("key1") is True
        assert cache.has("key2") is False


# ── LRU eviction ────────────────────────────────────────────────────


class TestLRUEviction:
    def test_evicts_lru_when_over_limit(self):
        """With a tiny limit, inserting a new entry should evict the oldest."""
        # json.dumps("aaa") → '"aaa"' → 5 bytes.  Limit of 10 fits 2 entries.
        cache = LRUCache(max_bytes=10)
        cache.put("first", "aaa")   # 5 bytes → total 5
        cache.put("second", "bbb")  # 5 bytes → total 10
        # This should evict "first" (LRU) to make room
        cache.put("third", "ccc")   # needs 5 bytes, evicts first → total 10
        assert cache.get("first") is None
        assert cache.get("third") == "ccc"

    def test_access_promotes_to_mru(self):
        """Accessing an entry should move it to MRU, protecting it from eviction."""
        cache = LRUCache(max_bytes=10)
        cache.put("a", "111")  # 5 bytes
        cache.put("b", "222")  # 5 bytes → total 10
        # Access "a" to promote it to MRU
        cache.get("a")
        # Insert "c" — should evict "b" (now LRU), not "a"
        cache.put("c", "333")
        assert cache.get("a") == "111"  # still present (was promoted)
        assert cache.get("b") is None   # evicted (was LRU)

    def test_skips_entry_larger_than_max(self):
        """A single entry that exceeds the limit should not be cached."""
        cache = LRUCache(max_bytes=10)
        cache.put("huge", "x" * 100)
        assert cache.get("huge") is None
        assert cache.entry_count == 0


# ── Size tracking ───────────────────────────────────────────────────


class TestSizeTracking:
    def test_current_bytes_tracks_inserts(self):
        cache = LRUCache()
        assert cache.current_bytes == 0
        cache.put("k", {"data": "hello"})
        assert cache.current_bytes > 0

    def test_current_bytes_decreases_on_invalidate(self):
        cache = LRUCache()
        cache.put("k", "val")
        size_before = cache.current_bytes
        cache.invalidate("k")
        assert cache.current_bytes < size_before
        assert cache.current_bytes == 0

    def test_overwrite_adjusts_size(self):
        cache = LRUCache()
        cache.put("k", "short")
        size_short = cache.current_bytes
        cache.put("k", "a much longer value string")
        assert cache.current_bytes > size_short
        assert cache.entry_count == 1


# ── Prefix invalidation ────────────────────────────────────────────


class TestPrefixInvalidation:
    def test_invalidates_matching_prefix(self):
        cache = LRUCache()
        cache.put("events_by_email:alice@test.com", [{"event": 1}])
        cache.put("events_by_email:bob@test.com", [{"event": 2}])
        cache.put("invitees:EVT1", [{"name": "Alice"}])

        removed = cache.invalidate_prefix("events_by_email:")
        assert removed == 2
        assert cache.get("events_by_email:alice@test.com") is None
        assert cache.get("events_by_email:bob@test.com") is None
        # Other prefixes untouched
        assert cache.get("invitees:EVT1") is not None

    def test_returns_zero_when_no_match(self):
        cache = LRUCache()
        cache.put("foo", "bar")
        assert cache.invalidate_prefix("zzz") == 0


# ── Reverse value lookup ────────────────────────────────────────────


class TestFindKeyContainingValue:
    def test_finds_key_with_matching_value(self):
        cache = LRUCache()
        cache.put(
            "events_by_email:alice@test.com",
            [{"uri": "https://api.calendly.com/scheduled_events/EVT1"}],
        )
        cache.put(
            "events_by_email:bob@test.com",
            [{"uri": "https://api.calendly.com/scheduled_events/EVT2"}],
        )

        key = cache.find_key_containing_value("events_by_email:", "EVT1")
        assert key == "events_by_email:alice@test.com"

    def test_returns_none_when_no_match(self):
        cache = LRUCache()
        cache.put("events_by_email:alice@test.com", [{"uri": "EVT1"}])
        key = cache.find_key_containing_value("events_by_email:", "EVT999")
        assert key is None

    def test_ignores_non_matching_prefix(self):
        cache = LRUCache()
        cache.put("invitees:EVT1", [{"name": "Alice"}])
        key = cache.find_key_containing_value("events_by_email:", "EVT1")
        assert key is None


# ── 20 MB default limit ────────────────────────────────────────────


class TestDefaultLimit:
    def test_default_max_is_20mb(self):
        cache = LRUCache()
        assert cache._max_bytes == 20 * 1024 * 1024
