import json
import os

from typing import Any, TypedDict, Literal

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

class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: str | None
    query: str
    k: int
    results: list[CombRRFData]

    
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


# LLM Prompts
spell_prompt = """Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
"""

rewrite_prompt = """Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

"""

expand_prompt = """Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

"""
