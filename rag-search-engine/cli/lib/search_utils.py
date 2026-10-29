import json
import os

from typing import Any, TypedDict

# The project root
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "movies.json")


# Maximum results shown when returning a search
MAX_RESULTS = 5

# Create a variable dictionary to store the data/json files
with open(DATA_PATH, "r") as data_file:
    movies_data_dict = json.load(data_file)