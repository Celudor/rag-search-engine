import argparse

from lib.constants import RRF_K
from lib.gemini import spell_enhancment
from lib.hybrid_search import HybridSearch, min_max_normalization
from lib.utils import load_movies


def normalize(scores: list[int | float]):
    normalized_scores = min_max_normalization(scores)
    for score in normalized_scores:
        print(f"* {score:.4f}")


def weighted_search(query: str, alpha: float, limit: int):
    hs = HybridSearch(load_movies())
    results = hs.weighted_search(query, alpha, limit)

    for i, (doc_id, doc) in enumerate(results, start=1):
        print(f"{i}. {doc['document']['title']}")
        print(f"   Hybrid Score: {doc['hybrid_score']:.3f}")
        print(
            f"   BM25: {doc['bm25_score']:.3f}, Semantic: {doc['semantic_score']:.3f}"
        )
        print(f"   {doc['document']['description'][:100]}...")


def rrf_search(query: str, k: int, limit: int, enhance_method: str):
    if enhance_method:
        enhanced_query = spell_enhancment(query)
        print(f"Enhanced query ({enhance_method}): '{query}' -> '{enhanced_query}'\n")
    else:
        enhanced_query = query
    hs = HybridSearch(load_movies())
    results = hs.rrf_search(enhanced_query, k, limit)

    for i, (doc_id, doc) in enumerate(results, start=1):
        print(f"{i}. {doc['document']['title']}")
        print(f"   RRF Score: {doc['rrf_score']:.3f}")
        print(
            f"   BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}"
        )
        print(f"   {doc['document']['description'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Score normalize using min-max method"
    )
    normalize_parser.add_argument(dest="scores", type=float, nargs="+")

    weighted_search_parser = subparsers.add_parser("weighted-search")
    weighted_search_parser.add_argument("query", type=str)
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    rrf_search_parser = subparsers.add_parser("rrf-search")
    rrf_search_parser.add_argument("query", type=str)
    rrf_search_parser.add_argument("-k", type=int, default=RRF_K)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        help="Query enhancment method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
