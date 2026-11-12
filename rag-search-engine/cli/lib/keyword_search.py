from .search_utils import (
    read_data,
    read_stop_words,
    MAX_RESULTS,
    CACHE_PATH,
    BM25_K1,
    BM25_B,
    Movie,
    BM25ScoreMovies,
)
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

import string
import os
import pickle
import sys
import math

# List of movies
movies_data_dict = read_data()
# List of stop words
s_words_list: list[str] = read_stop_words()
# Stemmer instance
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self) -> None:
        # A dictionary that maps a token to a set of indexes in wich that token appears
        self.index = defaultdict(set)
        # A dictionary that maps an id to a Movie object
        self.docmap: dict[int, Movie] = {}
        # A dictionary mapping IDs to a Counter Objects
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        # Lenght of a document
        self.doc_lengths: dict[int, int] = {}
        # Filepaths
        self.index_path: str = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path: str = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path: str = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path: str = os.path.join(CACHE_PATH, "doc_lengths.pkl")



    # Add each token and id into the index dict
    def __add_document(self, doc_id: int, text: str) -> None:
        t_text: list[str] = tokenizer(text)

        for token in set(t_text):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(t_text)
        self.doc_lengths[doc_id] = len(t_text)

    # Get the calculated average length of a document
    def __get_avg_doc_length(self) -> float:
        if len(self.docmap) == 0:
            return 0.0
        
        sum_doc_length: int = sum(self.doc_lengths.values())
        return sum_doc_length / len(self.docmap)


    # Return the frequency of a single token on a given id
    def get_tf(self, doc_id: int, term: str) -> int:
        token: str = single_tokenizer(term)
        return self.term_frequencies[doc_id][token]

    # Return each id from a token in ascending order
    def get_documents(self, term: str) -> list[int]:
        results = self.index.get(term, set())
        return sorted(results)

    # Return the IDF score of a given term
    def get_idf(self, term: str) -> float:
        token: str = single_tokenizer(term)

        token_index: list[int] = self.get_documents(token)
        term_match_doc_count: int = len(token_index)
        total_doc_count: int = len(self.docmap)

        idf_value: float = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf_value

    # Return the BM25 IDF score from a term
    def get_bm25_idf(self, term: str) -> float:
        token: str = single_tokenizer(term)

        doc_f: int = len(self.get_documents(token))
        tot_d: int = len(self.docmap)

        bm25_value: float = math.log((tot_d - doc_f + 0.5) / (doc_f + 0.5) + 1)
        return bm25_value

    # Return the BM25 TF socre from a term and a document ID
    def get_bm25_tf(self, doc_id: int, term: str, k1:float = BM25_K1, b:float = BM25_B) -> float:
        token: str = single_tokenizer(term)
        tf: int = self.get_tf(doc_id, token)

        doc_length: int = self.doc_lengths.get(doc_id, 0)
        avg_doc_length: float = self.__get_avg_doc_length()

        if avg_doc_length > 0:
            length_norm: float = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        bm25_sat: float = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_sat


    # Returns the BM25 score from a doc ID and a term
    def bm25(self, doc_id: int, term:str) -> float:
        tf: float = self.get_bm25_tf(doc_id, term)
        idf: float = self.get_bm25_idf(term)

        return tf * idf

    # Returns LIMIT documents based on a BM25 score
    def bm25_search(self, query: str, limit=MAX_RESULTS) -> list[BM25ScoreMovies]:
        tokens: list[str] = tokenizer(query)
        scores: dict[int, float] = {}
        return_scores: list[BM25ScoreMovies] = []

        for document in self.docmap.values():
            for token in tokens:
                doc_id: int = document["id"]
                scores[doc_id] = scores.get(doc_id, 0) + self.bm25(doc_id, token)

        sorted_scores: dict[int, float] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit])

        for s_doc_id, s_bm25 in sorted_scores.items():
            return_scores.append({
                "movie_id": s_doc_id,
                "score": s_bm25
            })

        return return_scores

  
    # Method to build the index and docmap dictionaries from the data file
    def build(self) -> None:
        movies: list[Movie] = read_data()

        for movie in movies:
            doc_id: int = movie["id"]
            doc_desc: str = f"{movie["title"]} {movie["description"]}"

            self.__add_document(doc_id, doc_desc)
            self.docmap[doc_id] = movie

    # Method to save the attributes dictionaries in disc
    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    # Method to load the attributes dictionaries from disc
    def load(self) -> None:
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)


# Command to build the cache files
def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

# Command to create a list of search results by name (Case Sensitive and non-relational)
def search_command(search_query: str, limit: int = MAX_RESULTS) -> list[str]:
    result_list: list[str] = []

    seen_result = set()
    index = InvertedIndex()

    # Convert the query and the title into a list of tokens
    query_tokens: list[str] = tokenizer(search_query)
    # Creates an index of every movie
    try:
        index.load()
    except FileNotFoundError:
        print("Error: file(s) not found. Exiting program")
        sys.exit(1)

    # Iterate over every query token
    for query_t in query_tokens:
        # Get a list of index that match the query token
        query_match: list[int] = index.get_documents(query_t)
        # Iterate over every index and return a list of movie id and title
        for q_index in query_match:
            if q_index in seen_result:
                continue

            seen_result.add(q_index)

            m_id: int = index.docmap[q_index]["id"]
            m_title: str = index.docmap[q_index]["title"]

            result_list.append(f"{m_id}. {m_title}")

            if len(result_list) >= limit:
                return result_list

    return result_list

# Command to get the term frequency of a given term
def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: file(s) not found. Exiting program")
        sys.exit(1)

    return index.get_tf(doc_id, term)

# Command to get the inverse document index of a given term
def idf_command(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: file(s) not found. Exiting program")
        sys.exit(1)

    return index.get_idf(term)

# Command to get the TF-IDF score of a given document and term
def tfidf_command(doc_id: int, term:str) -> float:
    tf: int = tf_command(doc_id, term)
    idf: float = idf_command(term)

    return tf * idf

# Command to get the BM25 IDF score of a term
def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: file(s) not found. Exiting program")
        sys.exit(1)

    return index.get_bm25_idf(term)

# Comand to get the BM25 TF score of a term and a doc ID
def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: file(s) not found. Exiting program")
        sys.exit(1)

    return index.get_bm25_tf(doc_id, term, k1, b)

# Comand to get the LIMIT results from a search query using BM25 score method
def bm25_command(search_query: str) -> list[str]:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError:
        print("Error: file(s) not found. Exiting program")
        sys.exit(1)

    format_results: list[str] = []

    document_info: list[BM25ScoreMovies] = index.bm25_search(search_query)
    for i in range(len(document_info)):
        movie_id: int = document_info[i]["movie_id"]
        score = document_info[i]["score"]
        movie_title: str = index.docmap[movie_id]["title"]
        format_results.append(f"{i + 1}. ({movie_id}) {movie_title} - Score: {score:.2f}")

    return format_results


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

# Helper function to preprocess each stop word
def p_stop_words(stop_words: list[str]) -> list[str]:
    for i in range(len(stop_words)):
        stop_words[i] = string_filter(stop_words[i])
    return stop_words

# Helper function to tokenize a single term
def single_tokenizer(term: str) -> str:
    token: list[str] = tokenizer(term)
    if len(token) != 1:
        raise ValueError("Invalid token")
    return token[0]
