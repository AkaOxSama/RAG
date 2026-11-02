from .search_utils import (
    read_data,
    read_stop_words,
    MAX_RESULTS,
    CACHE_PATH,
    Movie,
)
from nltk.stem import PorterStemmer
from collections import defaultdict

import string
import os
import pickle

# List of movies
movies_data_dict = read_data()
# List of stop words
s_words_list: list[str] = read_stop_words()
# Stemmer instance
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.index_path: str = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path: str = os.path.join(CACHE_PATH, "docmap.pkl")

    # Add each token and id into the index dict
    def __add_document(self, doc_id: int, text: str) -> None:
        t_text: list[str] = tokenizer(text)
        for token in set(t_text):
            self.index[token].add(doc_id)

    # Return each id from a token in ascending order
    def get_documents(self, term: str) -> list[int]:
        results = self.index.get(term, set())
        return sorted(results)

    def build(self) -> None:
        movies: list[Movie] = read_data()

        for movie in movies:
            doc_id: int = movie["id"]
            doc_desc: str = f"{movie["title"]} {movie["description"]}"

            self.__add_document(doc_id, doc_desc)
            self.docmap[doc_id] = movie

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

    

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


# Helper function to create a list of search results by name (Case Sensitive and non-relational)
def search_command(search_query: str, limit: int = MAX_RESULTS) -> list[str]:
    search_list: list[str] = []


    for movie in movies_data_dict:
        # Take the id and the title of each movie
        _id: int = movie["id"]
        _title: str = movie["title"]
        # Convert the query and the title into a list of tokens
        query_tokens: list[str] = tokenizer(search_query)
        title_tokens: list[str] = tokenizer(_title)

        stop_words: list[str] = p_stop_words(s_words_list)
            
        # Check membership between each  query token and title token
        if check_membership(query_tokens, title_tokens): # Only pass if at least 1 membership
            search_list.append(f"{_id}. {_title}")
            if len(search_list) >= limit:
                break

    return search_list

# Wrap function to clean the code
def tokenizer(text: str) -> list[str]:
    # Clean up punctuation
    proc_text: str = string_filter(text)
    # Converts the text into tokens
    tokens: list[str] = token_converter(proc_text)
    # Load the curated stop words list
    stop_words: list[str] = p_stop_words(s_words_list)
    # Filter out the stop words from the stemmed query tokens
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
    clean_up: list[str] = []
    for token in tokens:
        if token not in stop_words:
            # Append the stemmed version of the token
            clean_up.append(stemmer.stem(token))

    return clean_up

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

