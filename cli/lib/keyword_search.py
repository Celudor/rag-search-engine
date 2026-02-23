import math
import pickle
import string
from collections import Counter, defaultdict
from pathlib import Path

from lib.constants import BM25_B, BM25_K1
from lib.utils import load_movies
from nltk.stem import PorterStemmer


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
        movies = load_movies()
        for movie in movies:
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


def sanitize(text: str) -> list[str]:
    return remove_stopwords_and_stemm(tokenization(remove_puncation(text.lower())))
