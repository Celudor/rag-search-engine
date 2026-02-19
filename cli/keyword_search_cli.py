#!/usr/bin/env python3

import argparse
import json
import math
import pickle
import string
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

        self.cache_dir = Path("./cache")
        self.index_path = self.cache_dir / "index.pkl"
        self.docmap_path = self.cache_dir / "docmap.pkl"
        self.term_frequencies_path = self.cache_dir / "term_frequencies.pkl"
        self.doc_lengths_path = self.cache_dir / "doc_lengths.pkl"

    def __add_document(self, doc_id, text):
        """Tokenize the input text, then add each token to the index with the document ID."""
        sanitized_text = sanitize(text)
        for token in sanitized_text:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {
                    doc_id,
                }
        self.term_frequencies[doc_id] = Counter(sanitized_text)
        self.doc_lengths[doc_id] = len(sanitized_text)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        clean_term = sanitize(term)
        if len(clean_term) > 1:
            raise Exception("too many terms")
        return self.term_frequencies[doc_id][clean_term[0]]

    def get_documents(self, term) -> list[int]:
        if term.lower() in self.index:
            return sorted(self.index[term.lower()])
        return []

    def build(self):
        movies = load_db()
        for movie in movies["movies"]:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        self.cache_dir.mkdir(exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_bm25_idf(self, term: str) -> float:
        token = sanitize(term)[0]
        df = len(self.get_documents(token))
        return math.log((len(self.docmap) - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        lenght_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * lenght_norm)

    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit):
        sanitized_query = sanitize(query)
        scores = {}
        for doc_id in self.docmap:
            total = 0
            for token in sanitized_query:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total

        return list(sorted(scores.items(), key=lambda items: items[1], reverse=True))[
            :limit
        ]


def load_db() -> dict:
    with open("./data/movies.json", "r") as f:
        return json.load(f)


def load_stopwords() -> list[str]:
    with open("./data/stopwords.txt", "r") as f:
        return f.read().splitlines()


def remove_puncation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    clean = text.translate(table)
    return clean


def tokenization(text: str) -> list[str]:
    tokens = text.split()
    return [token for token in tokens if token]


def remove_stopwords_and_stemm(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stopwords = load_stopwords()
    return [stemmer.stem(token) for token in tokens if token not in stopwords]


def is_match(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for qtoken in query_tokens:
        for ttoken in title_tokens:
            if qtoken in ttoken:
                return True
    return False


def sanitize(text: str) -> list[str]:
    return remove_stopwords_and_stemm(tokenization(remove_puncation(text.lower())))


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
