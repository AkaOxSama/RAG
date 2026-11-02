import argparse
import json
import sys

from typing import Any

from lib.keyword_search import search_command, build_command

# Function to create the main search function
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()


