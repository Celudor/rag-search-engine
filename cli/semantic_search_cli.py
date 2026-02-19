#!/usr/bin/env python3

import argparse
import json

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def search(query: str, limit: int = 5):
    ss = SemanticSearch()
    with open("./data/movies.json", "r") as f:
        docs = json.load(f)["movies"]
    ss.load_or_create_embedings(docs)
    results = ss.search(query, limit)
    for i, movie in enumerate(results):
        print(f"{i + 1}. {movie['title']} (score: {movie['score']:.4f})")
        print(f"{'':<3}{movie['description'][:100]} ...")


def chunk(text: str, chunk_size: int = 200, overlap=40):
    splited = text.split()
    chunks = []
    for i in range(0, len(splited), chunk_size):
        if i == 0:
            s = " ".join(splited[i : i + chunk_size])
        else:
            if i - overlap > 0:
                s = " ".join(splited[i - overlap : i + chunk_size])
            else:
                s = " ".join(splited[0 : i + chunk_size])
        chunks.append(s)

    print(f"Chunking {len(text)} characters")
    for i, s in enumerate(chunks):
        print(f"{i + 1}. {s}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Verify model")
    subparser.add_parser("verify_embeddings", help="Verify model")

    embed_text_parser = subparser.add_parser("embed_text")
    embed_text_parser.add_argument("text", type=str)

    embedquery_parser = subparser.add_parser("embedquery")
    embedquery_parser.add_argument("query", type=str)

    search_parser = subparser.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparser.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=40)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
