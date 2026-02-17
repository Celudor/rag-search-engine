#!/usr/bin/env python3

import argparse
import json
import math
import pickle
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer


class InvertedIndex:
    index: dict = {}
    docmap: dict = {}
    term_frequencies = {}

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
        Path("./cache").mkdir(exist_ok=True)
        with open("./cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("./cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("./cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open("./cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("./cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("./cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)


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


def search(query: str):
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


def build():
    invert_index = InvertedIndex()
    invert_index.build()
    invert_index.save()


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

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search(args.query)
        case "build":
            build()
        case "tf":
            index = InvertedIndex()
            index.load()
            num = index.get_tf(args.doc_id, sanitize(args.term)[0])
            print(num)
        case "idf":
            index = InvertedIndex()
            index.load()
            token = sanitize(args.term)[0]
            idf = math.log((len(index.docmap) + 1) / (len(index.index[token]) + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            index.load()
            token = sanitize(args.term)[0]
            idf = math.log((len(index.docmap) + 1) / (len(index.index[token]) + 1))
            tf = index.get_tf(args.doc_id, args.term)
            tf_idf = tf * idf
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
