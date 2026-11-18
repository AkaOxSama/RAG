from PIL import Image
from PIL.ImageFile import ImageFile
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from typing import Any

EmbeddingArray = NDArray[Any]


class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> EmbeddingArray:
        image: ImageFile = Image.open(image_path)
        image_embed_array: EmbeddingArray = self.model.encode([image])
        return image_embed_array[0]


# Command function to create a instance of a MultimodalSearch class and return the embeddings from a image path
def verify_img_embedding_command(image_path: str) -> int:
    multimodal_search = MultimodalSearch()

    embedding: EmbeddingArray = multimodal_search.embed_image(image_path)

    num_embeddings: int = embedding.shape[0]

    return num_embeddings