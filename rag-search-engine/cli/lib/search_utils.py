import json
import os

from typing import Any, TypedDict

# Movie object
class Movie(TypedDict):
    id: int
    title: str
    description: str

# Semantic Movie Object
class S_Search_Result(TypedDict):
    score: float
    title: str
    description: str

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int
    
# The project root
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MOVIES_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH: str = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH: str = os.path.join(PROJECT_ROOT, "cache")

# Maximum results shown when returning a search
MAX_RESULTS = 5
# Default Chunk Size
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 1 

# Default semantic chunk size
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

# Saturation control
BM25_K1 = 1.5
# Lenght normalization control
BM25_B = 0.75


# Create a list of stop words
def read_stop_words() -> list[str]:
    with open(STOPWORDS_PATH) as stop_words:
        s_words_list: list[str] = stop_words.read().splitlines()
    return s_words_list

# Create a variable dictionary to store the data/json files
def read_data() -> list[Movie]:
    with open(MOVIES_DATA_PATH, "r") as data_file:
        movies_data_dict = json.load(data_file)
    return movies_data_dict["movies"]



