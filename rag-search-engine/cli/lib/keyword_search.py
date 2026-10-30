from .search_utils import movies_data_dict, s_words_list, MAX_RESULTS

import string

# Helper function to create a list of search results by name (Case Sensitive and non-relational)
def search_command(search_query: str, limit: int = MAX_RESULTS) -> list[str]:
    search_list: list[str] = []

    for movies in movies_data_dict.values():
        for movie in movies:
            # Take the id and the title of each movie
            _id: int = movie["id"]
            _title: str = movie["title"]
            # Convert the query and the title into a list of tokens
            query_tokens: list[str] = tokenizer(search_query)
            title_tokens: list[str] = tokenizer(_title)
            
            # Check membership between each  query token and title token
            if check_membership(query_tokens, title_tokens): # Only pass if at least 1 membership
                search_list.append(f"{_id}. {_title}")
                if len(search_list) >= limit:
                    break

    return search_list

# Wrap function to clean the code
def tokenizer(text: str) -> list[str]:
    proc_text: str = string_filter(text)
    tokens: list[str] = token_converter(proc_text)

    stop_words: list[str] = p_stop_words(s_words_list)

    prc_tokens: list[str] = clean_stop_words(tokens, stop_words)
    return prc_tokens

# Helper function to strip the punctuation from strings
def string_filter(text: str) -> str:

    # Make the text case insensitive
    lower_text: str = text.lower()

    # Strip the text of punctuation characters
    punctuation = string.punctuation
    translation_table = str.maketrans("", "", punctuation)

    filter_text: str = lower_text.translate(translation_table)

    return filter_text

# Helper function to tokenize text
def token_converter(text: str) ->list[str]:
    # Clean the text of punctuations
    clean_text = string_filter(text)

    valid_tokens: list[str] = []
    tokens: list[str] = clean_text.split()
    for token in tokens:
        if token:
            valid_tokens.append(token)

    return valid_tokens

# Helper function to filter tokens by stopwords
def clean_stop_words(tokens: list[str], stop_words: list[str]) -> list[str]:
    # TODO: Complete this function mmgvo
    return []


# Helper function for membership logic
def check_membership(query_tokens: list[str], data_token: list[str]) -> bool:
    for query_token in query_tokens:
        if any(query_token in title for title in data_token):
            return True
    return False


# Helper function to preprocess each stop word
def p_stop_words(stop_words: list[str]) -> list[str]:
    for i in range(len(stop_words)):
        stop_words[i] = string_filter(stop_words[i])
    return stop_words