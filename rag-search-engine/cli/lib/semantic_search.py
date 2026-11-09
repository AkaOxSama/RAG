import numpy as np
import os
import re
import json

from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from typing import Any
from lib.search_utils import (
    Movie,
    S_Search_Result,
    ChunkMetadata,
    read_data,
    CACHE_PATH,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_OVERLAP_SIZE
)

EmbeddingArray = NDArray[Any]

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

        self.embeddings_path: str = os.path.join(CACHE_PATH, "movie_embeddings.npy")


    # Generates and embeding of a single piece of text
    def generate_embedding(self, text: str) -> EmbeddingArray:
        if text == "" or len(text.split()) == 0:
            raise ValueError("Invalid string input")
        embedding_array = self.model.encode([text])
        return embedding_array[0]

    # Build embeddings for the movies dataset
    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents

        string_repr: list[str] = []

        for doc in documents: 
            self.document_map[doc["id"]] = doc
            string_repr.append(f"{doc["title"]}: {doc["description"]}")

        self.embeddings = self.model.encode(string_repr, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    # Load or create the embedding
    def load_or_create_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            assert self.embeddings is not None
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



class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: EmbeddingArray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None
        self.chunk_embeddings_path: str = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path: str = os.path.join(CACHE_PATH, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents: list[Movie] = documents

        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        c_metadata: list[ChunkMetadata] = []

        for idx, doc in enumerate(self.documents):
            desc: str = doc.get("description", "")
            if not desc.strip():
                continue

            chunks: list[str] = semantic_chunk(
                desc,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_OVERLAP_SIZE
            )

            

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                c_metadata.append(
                    {
                        "movie_idx": idx,
                        "chunk_idx": i,
                        "total_chunks": len(chunks)
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = c_metadata

        os.makedirs(os.path.dirname(self.chunk_embeddings_path), exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) ->EmbeddingArray:
        self.documents: list[Movie] = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

            assert self.chunk_embeddings is not None
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)



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
def text_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    words: list[str] = text.split()
    i: int = 0

    while i < len(words):
            chunk_sentences = words[i : i + chunk_size]
            if chunks and len(chunk_sentences) <= overlap:
                break
            chunks.append(" ".join(chunk_sentences))
            i += chunk_size - overlap

    return chunks

# Command to divide text into semantic chunks
def semantic_chunk(text:str, max_chunk_size: int, overlap: int) -> list[str]:
    sentences: list[str | Any] = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    i = 0
    
    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks

# Command to embed chunks
def embed_chunks():
    movies: list[Movie] = read_data()
    semantic_chunk_model = ChunkedSemanticSearch()

    embeddings = semantic_chunk_model.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


# Helper function to get cosine similarities between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
