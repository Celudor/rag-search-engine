#!/usr/bin/env python3

import argparse
import json
import math

from lib.keyword_search import InvertedIndex, sanitize
from lib.utils import BM25_B, BM25_K1


def load_db() -> dict:
    with open("./data/movies.json", "r") as f:
        return json.load(f)


def is_match(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for qtoken in query_tokens:
        for ttoken in title_tokens:
            if qtoken in ttoken:
                return True
    return False


def search_command(query: str):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as err:
        print(err)
        return
    docs = set()
    for token in sanitize(query):
        docs.update(index.get_documents(token))
        if len(docs) >= 5:
            break
    for doc in list(sorted(docs))[:5]:
        print(f"{doc} {index.docmap[doc]['title']}")


def build_command():
    invert_index = InvertedIndex()
    invert_index.build()
    invert_index.save()


def tf_command(doc_id, term):
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, sanitize(term)[0])


def idf_command(term):
    index = InvertedIndex()
    index.load()
    token = sanitize(term)[0]
    return math.log((len(index.docmap) + 1) / (len(index.index[token]) + 1))


def tfidf_command(doc_id, term):
    index = InvertedIndex()
    index.load()
    token = sanitize(term)[0]
    idf = math.log((len(index.docmap) + 1) / (len(index.index[token]) + 1))
    tf = index.get_tf(doc_id, term)
    return tf * idf


def bm25idf_command(term):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1)


def bm25search_command(query, limit=5):
    index = InvertedIndex()
    index.load()
    results = index.bm25_search(query, limit)
    for i, (doc_id, score) in enumerate(results):
        print(
            f"{i + 1}. ({doc_id}) {index.docmap[doc_id]['title']} - Score: {score:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build reverse index")

    tf_parser = subparsers.add_parser("tf", help="Check term frequencies")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", help="Term")

    tf_idf_parser = subparsers.add_parser("tfidf")
    tf_idf_parser.add_argument("doc_id", type=int)
    tf_idf_parser.add_argument("term", type=str)

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, help="Limit search result (default 5).", default=5
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(tf)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            bm25search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
