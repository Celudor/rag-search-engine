from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        combined_results = {}
        for i, nscore in enumerate(
            min_max_normalization([score[1] for score in bm25_results])
        ):
            combined_results[bm25_results[i][0]] = {
                "document": next(
                    (doc for doc in self.documents if doc["id"] == bm25_results[i][0])
                ),
                "bm25_score": nscore,
                "semantic_score": 0.0,
            }

        css_results = self.semantic_search.search_chunks(query, limit * 500)
        for i, nscore in enumerate(
            min_max_normalization([score["score"] for score in css_results])
        ):
            if css_results[i]["id"] in combined_results:
                combined_results[css_results[i]["id"]]["semantic_score"] = nscore
            else:
                combined_results[css_results[i]["id"]] = {
                    "document": next(
                        (
                            doc
                            for doc in self.documents
                            if doc["id"] == css_results[i]["id"]
                        )
                    ),
                    "bm25_score": 0.0,
                    "semantic_score": nscore,
                }
        for id in combined_results:
            combined_results[id]["hybrid_score"] = hybrid_score(
                combined_results[id]["bm25_score"],
                combined_results[id]["semantic_score"],
                alpha,
            )

        return sorted(
            combined_results.items(),
            key=lambda item: item[1]["hybrid_score"],
            reverse=True,
        )[:limit]

    def rrf_search(self, query: str, k: int, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = {}
        for i, doc in enumerate(bm25_results, start=1):
            combined_results[doc[0]] = {
                "document": next((d for d in self.documents if d["id"] == doc[0])),
                "bm25_rank": i,
                "bm25_score": rrf_score(i, k),
                "semantic_score": 0,
                "semantic_rank": -1,
                "rrf_score": rrf_score(i, k),
            }
        for i, doc in enumerate(semantic_result, start=1):
            if doc["id"] in combined_results:
                combined_results[doc["id"]]["semantic_score"] = rrf_score(i, k)
                combined_results[doc["id"]]["semantic_rank"] = i
                combined_results[doc["id"]]["rrf_score"] = (
                    combined_results[doc["id"]]["bm25_score"]
                    + combined_results[doc["id"]]["semantic_score"]
                )
            else:
                combined_results[doc["id"]] = {
                    "document": next(
                        (d for d in self.documents if d["id"] == doc["id"])
                    ),
                    "bm25_score": 0,
                    "bm25_rank": -1,
                    "semantic_score": rrf_score(i, k),
                    "semantic_rank": i,
                    "rrf_score": rrf_score(i, k),
                }
        return sorted(
            combined_results.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )[:limit]


def rrf_score(rank: int, k=60):
    return 1 / (k + rank)


def hybrid_score(bm25_score: float, semantic_score: float, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def min_max_normalization(scores: list[int | float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for score in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]
