import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model characteristics")

    subparsers.add_parser("verify_embeddings", help="Verify the contruction of the embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Get an embedded respone from a text")
    embed_text_parser.add_argument("text", type=str, help="Text to get an embedded response")

    embed_query_parser = subparsers.add_parser("embed_query", help="Get an embedded respones from a query")
    embed_query_parser.add_argument("query", type=str, help="The query to use for embedding")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Displaying model info...")
            verify_model()

        case "verify_embeddings":
            verify_embeddings()
            
        case "embed_text":
            embed_text(args.text)

        case "embed_query":
            embed_query_text(args.query)


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()