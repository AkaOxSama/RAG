import argparse

from lib.augmented_generation import(
    rag_command,
    summarize_command,
    citations_command,
    question_command,
)

from lib.search_utils import (
    SearchWithAIResponse,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Perform an AI summary of a search result")
    summarize_parser.add_argument("query", type=str, help="Search query for summary")
    # Citations command
    citations_parser = subparsers.add_parser("citations", help="Perform an AI answer with citations")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    # Question command
    question_parser = subparsers.add_parser("question", help="Perform an AI response to a question")
    question_parser.add_argument("question", type=str, help="Question to answer")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            print(f"Searching for: {query}\n")

            search_response: SearchWithAIResponse = rag_command(query)

            print("Search Results:")
            for title in search_response["search_result"]:
                print(f"- {title}")
            print()
            print("RAG Response:")
            print(search_response["llm_response"])
            print()

        case "summarize":
            query = args.query
            print(f"Searching for: {query}\n")

            search_response: SearchWithAIResponse = summarize_command(query)

            print("Search Results:")
            for title in search_response["search_result"]:
                print(f"- {title}")
            print()
            print("LLM Summary")
            print(search_response["llm_response"])
            print()

        case "citations":
            query = args.query
            print(f"Searching for: {query}\n")

            search_response: SearchWithAIResponse = summarize_command(query)

            print("Search Results:")
            for title in search_response["search_result"]:
                print(f"- {title}")
            print()
            print("LLM Summary")
            print(search_response["llm_response"])
            print()

        case "question":
            question = args.question
            print(f"Searching for: {question}\n")

            search_response = question_command(question)

            print("Search Results:")
            for title in search_response["search_result"]:
                print(f"- {title}")
            print()
            print("LLM Summary")
            print(search_response["llm_response"])
            print()
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()