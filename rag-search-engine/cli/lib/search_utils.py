import json
import os

from typing import Any, TypedDict

# Movie object

class Movie(TypedDict):
    id: int
    title: str
    description: str
    
# The project root
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MOVIES_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH: str = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH: str = os.path.join(PROJECT_ROOT, "cache")

# Maximum results shown when returning a search
MAX_RESULTS = 5
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



