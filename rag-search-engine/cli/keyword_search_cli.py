import argparse
import json
import sys

from typing import Any

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


