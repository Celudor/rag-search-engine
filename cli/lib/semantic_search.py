import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.cache_dir = Path("./cache")
        self.embeddings_path = self.cache_dir / "movie_embedings.npy"

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Empty string")
        [embedding] = self.model.encode([text])
        return embedding

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_list = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}:{doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        self.save()
        return self.embeddings

    def save(self):
        self.cache_dir.mkdir(exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.embeddings_path.exists():
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        scores = []
        for i in range(len(self.embeddings)):
            similarity_score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((similarity_score, self.documents[i]))
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for (score, doc) in sorted(
                scores, key=lambda items: items[0], reverse=True
            )[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embedings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = self.cache_dir / "chunk_embeddings.npy"
        self.chunk_metadata_path = self.cache_dir / "chunk_metadata.json"

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        all_chunks: list[str] = []
        chunks_metadata: list[dict] = []
        for doc in self.documents:
            if not doc["description"]:
                continue
            chunks = semantic_chunk(doc["description"], 4, 1)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunks_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embedings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        self.cache_dir.mkdir(exist_ok=True)
        with open(self.chunk_embeddings_path, "wb") as f:
            np.save(f, self.chunk_embedings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                f,
                indent=2,
            )
        return self.chunk_embedings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            with open(self.chunk_embeddings_path, "rb") as f:
                self.chunk_embedings = np.load(f)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embedings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if len(self.chunk_embedings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        scores: list[dict] = []
        for i in range(len(self.chunk_embedings)):
            similarity_score = cosine_similarity(
                query_embedding, self.chunk_embedings[i]
            )
            scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity_score,
                }
            )
        movie_scores = {}
        for score in scores:
            if (
                score["movie_idx"] not in movie_scores
                or score["score"] > movie_scores[score["movie_idx"]]
            ):
                movie_scores[score["movie_idx"]] = score["score"]
        movie_scores = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]
        return [
            {
                "id": m[0],
                "title": self.document_map[m[0]]["title"],
                "description": self.document_map[m[0]]["description"][:100],
                "score": round(m[1], 6),
                "metadata": {},
            }
            for m in movie_scores
        ]


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def chunking(splited: list[str], chunk_size: int, overlap):
    chunks = []
    n_splited = len(splited)
    i = 0
    while i < n_splited:
        chunk = splited[i : i + chunk_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def semantic_chunk(text: str, chunk_size: int = 4, overlap: int = 0):
    ctext = text.strip()
    if not ctext:
        return []
    else:
        return chunking(
            [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctext) if s.strip()],
            chunk_size,
            overlap,
        )
