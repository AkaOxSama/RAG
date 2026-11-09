import numpy as np
import os

from sentence_transformers import SentenceTransformer
from typing import Any
from lib.search_utils import (
    Movie,
    S_Search_Result,
    read_data,
    CACHE_PATH,
)

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_path: str = os.path.join(CACHE_PATH, "movie_embeddings.npy")


    # Generates and embeding of a single piece of text
    def generate_embedding(self, text: str) -> Any:
        if text == "" or len(text.split()) == 0:
            raise ValueError("Invalid string input")
        embedding_array = self.model.encode([text])
        return embedding_array[0]

    # Build embeddings for the movies dataset
    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents

        string_repr: list[str] = []

        for doc in documents: 
            self.document_map[doc["id"]] = doc
            string_repr.append(f"{doc["title"]}: {doc["description"]}")

        self.embeddings = self.model.encode(string_repr, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    # Load or create the embedding
    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(documents) == len(self.embeddings):
                return self.embeddings
            
        return self.build_embeddings(documents)


    # Search for embedded vectors using a query
    def search(self, query: str, limit:int) -> list[S_Search_Result]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")

        q_embedding = self.generate_embedding(query)
        docs_similarities: list[tuple[float, Movie]] = []

        for i, d_embedding in enumerate(self.embeddings):
            similarity_score: float = cosine_similarity(q_embedding, d_embedding)
            doc: Movie = self.documents[i]

            docs_similarities.append((similarity_score, doc))

        docs_similarities.sort(key=lambda x: x[0], reverse=True)

        top_results: list[S_Search_Result] = []

        for score, doc in docs_similarities[:limit]:
            top_results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return top_results



# Command to verify the model
def verify_model() -> None:
    semantic_model = SemanticSearch()

    print(f"Model loaded: {semantic_model.model}")
    print(f"Max sequence length: {semantic_model.model.max_seq_length}")

# Command to verify the creation of the embeddings
def verify_embeddings():
    semantic_model = SemanticSearch()

    movies: list[Movie] = read_data()

    embeddings = semantic_model.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

# Command to embed a single piece of text
def embed_text(text: str) -> Any:
    semantic_model = SemanticSearch()

    embedding = semantic_model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

# Command to embed a query of text
def embed_query_text(query: str):
    semantic_model = SemanticSearch()

    embedding = semantic_model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

# Comand to use semantic search on a query
def semantic_search(query:str, limit):
    semantic_model = SemanticSearch()

    movies: list[Movie] = read_data()

    semantic_model.load_or_create_embeddings(movies)

    results: list[S_Search_Result] = semantic_model.search(query, limit)

    print(f"Query: '{query}'")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})\n  {result['description'][:100]}...")
        print()

# Command to divide text into chunks
def text_chunk(text: str, chunk_size) -> None:
    characters: int = len(text)
    chunks = []
    words: list[str] = text.split()
    i = 0

    while i < len(words):
        chunk: list[str] = words[i: i + chunk_size]
        chunks.append(chunk)
        i += chunk_size

    print(f"Chunking {characters} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {" ".join(chunk)}")
        


# Helper function to get cosine similarities between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


