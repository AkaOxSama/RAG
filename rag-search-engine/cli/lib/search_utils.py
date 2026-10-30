import json
import os

from typing import Any, TypedDict

# The project root
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIES_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH: str = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

# Maximum results shown when returning a search
MAX_RESULTS = 5

with open(STOPWORDS_PATH) as stop_words:
    s_words_list: list[str] = stop_words.read().splitlines()

# Create a variable dictionary to store the data/json files
with open(MOVIES_DATA_PATH, "r") as data_file:
    movies_data_dict = json.load(data_file)