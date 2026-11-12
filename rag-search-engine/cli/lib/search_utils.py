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

class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float

class ChunkMovieResults(TypedDict):
    movie_id: int
    title: str
    document: str
    score: float

class BM25ScoreMovies(TypedDict):
    movie_id: int
    score: float

class HybridScoresData(TypedDict):
    title: str
    description: str
    bm25_sc: float
    sem_sc: float
    hyb_sc: float

class CombScoreData(TypedDict):
    title: str
    description: str
    bm25_score: float
    sem_score: float
    hyb_score: float

class CombRRFData(TypedDict):
    title: str
    description: str
    bm25_rank: int
    sem_rank: int
    rrf_score: float

    
# The project root
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MOVIES_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH: str = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH: str = os.path.join(PROJECT_ROOT, "cache")

# Maximum results shown when returning a search
MAX_RESULTS = 5
# Maximum results from sub searches for hybird search
MAX_SUB_SEARCH = 500

# Default Chunk Size
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 1 

# Default semantic chunk size
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

# Default Alpha Value
ALPHA_CONSTANT = 0.5
# Default RRF K Value
K_CONSTANT = 60

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



