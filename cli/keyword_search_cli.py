#!/usr/bin/env python3

import argparse
import json
import string

from nltk.stem import PorterStemmer


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


def search(query: str):
    movies = load_db()
    found = []
    query_tokens = remove_stopwords_and_stemm(
        tokenization(remove_puncation(query.lower()))
    )
    for movie in movies["movies"]:
        title_tokens = remove_stopwords_and_stemm(
            tokenization(remove_puncation(movie["title"].lower()))
        )
        if is_match(query_tokens, title_tokens):
            found.append(movie["title"])
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            for i, title in enumerate(search(args.query)[:5]):
                print(f"{i + 1}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
