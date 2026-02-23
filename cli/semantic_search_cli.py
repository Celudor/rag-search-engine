#!/usr/bin/env python3

import argparse
import json

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    chunking,
    semantic_chunk,
)
from lib.utils import load_movies


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    ss = SemanticSearch()
    with open("./data/movies.json", "r") as f:
        docs = json.load(f)["movies"]
    embeddings = ss.load_or_create_embeddings(docs)
    print(type(embeddings))
    print(f"Number of docs: {len(ss.documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def search(query: str, limit: int = 5):
    ss = SemanticSearch()
    docs = load_movies()
    ss.load_or_create_embeddings(docs)
    results = ss.search(query, limit)
    for i, movie in enumerate(results):
        print(f"{i + 1}. {movie['title']} (score: {movie['score']:.4f})")
        print(f"{'':<3}{movie['description'][:100]} ...")


def chunk(text: str, chunk_size: int = 200, overlap=40):
    chunks = chunking(text.split(), chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, s in enumerate(chunks):
        print(f"{i + 1}. {s}")


def embed_chunks():
    with open("./data/movies.json", "r") as f:
        docs = json.load(f)["movies"]
    chunked_ss = ChunkedSemanticSearch()
    chunk_embeddings = chunked_ss.load_or_create_chunk_embeddings(docs)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int = 5):
    docs = load_movies()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(docs)
    results = css.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"\n{i + 1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['description']}...")


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Verify model")
    subparser.add_parser("verify_embeddings", help="Verify embedings")

    embed_text_parser = subparser.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Input text")

    embedquery_parser = subparser.add_parser("embedquery", help="Embed query")
    embedquery_parser.add_argument("query", type=str, help="Query")

    search_parser = subparser.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str, help="Input query")
    search_parser.add_argument("--limit", type=int, default=5, help="Limit (default 5)")

    chunk_parser = subparser.add_parser("chunk", help="Chunking text")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=40)

    semantic_chunk_parser = subparser.add_parser(
        "semantic_chunk", help="Symantic chunking text"
    )
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    subparser.add_parser("embed_chunks", help="Embed chunks")

    search_chunked_parser = subparser.add_parser(
        "search_chunked", help="Semantic chunked search"
    )
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)

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
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            if not chunks:
                print("1. ")
            else:
                for i, s in enumerate(chunks):
                    print(f"{i + 1}. {s}")

        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
