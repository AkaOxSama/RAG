import argparse
import json
import sys

from typing import Any

from lib.search_utils import(
    BM25_K1
)

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
)

# Function to create the main search function
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command to build the index and documents in cache
    subparsers.add_parser("build", help="Build the inverted index")

    # Command to return the frequency of a term
    tf_parser = subparsers.add_parser("tf", help="Returns the frequency of a term for a given document ID")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    # Command to get the inverse document frequency of a term
    idf_parser = subparsers.add_parser("idf", help="Returns the inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="Gets the IDF of the term")

    # Command to get the TF-IDF score 
    tfidf_parser = subparsers.add_parser("tfidf", help="Get the score of the TF-IDF from a document and a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get the TF-IDF from")

    # Command to get the BM25_IDF score
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # Command to get the BM25_TF score
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")

    # Command to search a movie
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            # Call the helper function to show the results
            search_result: list[str] = search_command(args.query)
            for result in search_result:
                print(result)

        case "tf":
            print(f"Searching frequency of {args.term} in the doc_id: {args.doc_id}")
            tf: int = tf_command(args.doc_id, args.term)
            print(f"The term was found: {tf} times in the document: {args.doc_id}")

        case "idf":
            print(f"Getting the IDF of the term: {args.term}...")
            idf: float = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            print(f"Getting the TF-IDF score from document ID: {args.doc_id} and the term: '{args.term}'")
            tf_idf: float = tfidf_command(args.doc_id, args.term)
            print(f"The TF-IDF score is: {tf_idf:.2f}")

        case "bm25idf":
            print(f"Getting the BM25 IDF score of: {args.term}")
            bm25_idf: float = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            print(f"Getting the BM25 TF score from document ID: {args.doc_id} and term {args.term}")
            bm25_tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


