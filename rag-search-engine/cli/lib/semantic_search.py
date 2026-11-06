import numpy as np
import os

from sentence_transformers import SentenceTransformer
from typing import Any
from lib.search_utils import (
    Movie,
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


