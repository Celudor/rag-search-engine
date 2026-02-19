import json
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

    def build_embedings(self, documents: list[dict]):
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

    def load_or_create_embedings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.embeddings_path.exists():
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                return self.build_embedings(documents)
        else:
            return self.build_embedings(documents)

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
    embeddings = ss.load_or_create_embedings(docs)
    print(type(embeddings))
    print(f"Number of docs: {len(ss.documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
