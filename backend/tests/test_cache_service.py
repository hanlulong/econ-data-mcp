from __future__ import annotations

import unittest

from backend.services.cache import CacheService


class CacheServiceTests(unittest.TestCase):
    def test_delete_removes_existing_key(self) -> None:
        cache = CacheService()
        cache.set("demo-key", {"value": 1})

        deleted = cache.delete("demo-key")

        self.assertTrue(deleted)
        self.assertIsNone(cache.get("demo-key"))

    def test_stats_alias_matches_get_stats(self) -> None:
        cache = CacheService()
        cache.set("demo-key", 1)
        cache.get("demo-key")

        self.assertEqual(cache.stats(), cache.get_stats())


if __name__ == "__main__":
    unittest.main()
