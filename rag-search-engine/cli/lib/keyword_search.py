from .search_utils import movies_data_dict, MAX_RESULTS

# Helper function to create a list of search results by name (Case Sensitive and non-relational)
def search_command(search_query: str, limit: int = MAX_RESULTS) -> list[str]:
    search_list: list[str] = []

    for movies in movies_data_dict.values():
        for movie in movies:
            _id: int = movie["id"]
            _title: str = movie["title"]
            if search_query in _title:
                search_list.append(f"{_id}. {_title}")
                if len(search_list) >= limit:
                    break

    return search_list