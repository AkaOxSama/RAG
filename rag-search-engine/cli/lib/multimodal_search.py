import numpy as np

from PIL import Image
from PIL.ImageFile import ImageFile
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from typing import Any

from .search_utils import(
    Movie, 
    read_data,
    ImageSearchResult,
    MAX_RESULTS,
)

EmbeddingArray = NDArray[Any]
movies: list[Movie] = read_data()

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents: list[Movie]=movies) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents: list[Movie] = documents
        self.texts: list[str] = concatenate_docs(self.documents)

        self.text_embeddings: EmbeddingArray = self.model.encode(self.texts, show_progress_bar=True)


    def embed_image(self, image_path: str) -> EmbeddingArray:
        image: ImageFile = Image.open(image_path)
        image_embed_array: EmbeddingArray = self.model.encode([image])
        return image_embed_array[0]
        

    # Method to create search results based on a image
    def search_with_image(self, image_path: str, limit=MAX_RESULTS) -> list[ImageSearchResult]:
        img_embedding = self.embed_image(image_path)

        doc_similarities: list[tuple[float, Movie]] = []

        for i, txt_embed in enumerate(self.text_embeddings):
            cos_sim: float = cosine_similarity(img_embedding, txt_embed)
            doc_similarities.append((cos_sim, self.documents[i]))

        doc_similarities = sorted(doc_similarities, key=lambda x: x[0], reverse=True)[:limit]
        search_results: list[ImageSearchResult] = []

        for c_score, movie in doc_similarities:
            search_results.append({
                "movie_id": movie['id'],
                "title": movie['title'],
                "description": movie['description'],
                "score": c_score,
            })

        return search_results

        


# Command function to create a instance of a MultimodalSearch class and return the embeddings from a image path
def verify_img_embedding_command(image_path: str) -> int:
    multimodal_search = MultimodalSearch()

    embedding: EmbeddingArray = multimodal_search.embed_image(image_path)

    num_embeddings: int = embedding.shape[0]

    return num_embeddings

# Command to create search results from an image path:
def image_search_command(image_path: str) -> list[ImageSearchResult]:
    multimodal_search = MultimodalSearch()

    search_results: list[ImageSearchResult] = multimodal_search.search_with_image(image_path)

    return search_results



# Helper function to concatenate the documents into a title-description format
def concatenate_docs(documents: list[Movie]) -> list[str]:
    formated_docs: list[str] = []
    for doc in documents:
        formated_docs.append(f"{doc['title']}: {doc['description']}")

    return formated_docs

# Helper function to get cosine similarities between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)