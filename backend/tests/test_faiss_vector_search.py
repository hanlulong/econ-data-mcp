from __future__ import annotations

import numpy as np

from backend.services.faiss_vector_search import FAISSVectorSearch


class _FakeIndex:
    ntotal = 3

    def search(self, query_np, k):
        distances = np.array([[0.11, 0.22, 0.33]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)
        return distances, indices


def test_search_uses_raw_rank_distance_when_provider_filter_skips_items():
    searcher = object.__new__(FAISSVectorSearch)
    searcher.index = _FakeIndex()
    searcher.metadata_list = [
        {"code": "A", "name": "WorldBank A", "provider": "WORLDBANK"},
        {"code": "B", "name": "FRED B", "provider": "FRED"},
        {"code": "C", "name": "WorldBank C", "provider": "WORLDBANK"},
    ]
    searcher.embed_text = lambda text: [0.0] * 384

    results = searcher.search("imports to gdp", limit=2, provider_filter="WORLDBANK")

    assert len(results) == 2
    assert results[0].code == "A"
    assert results[1].code == "C"
    # Second kept result should keep its original FAISS rank distance (0.33), not 0.22.
    assert float(results[1].distance) == float(np.float32(0.33))
