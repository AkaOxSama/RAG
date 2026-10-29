import argparse
import json
from typing import Any

from lib.keyword_search import search_command

# Function to create the main search function
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            # Call the helper function to show the results
            search_result: list[str] = search_command(args.query)
            for result in search_result:
                print(result)
        case _:
            parser.print_help()

# Helper function to load a list of search results



if __name__ == "__main__":
    main()


