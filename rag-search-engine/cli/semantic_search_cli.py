import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    text_chunk,
    semantic_chunk,
    embed_chunks,
)
from lib.search_utils import (
    MAX_RESULTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_SIZE,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify model command
    subparsers.add_parser("verify", help="Verify the model characteristics")
    # Verify embeddings command
    subparsers.add_parser("verify_embeddings", help="Verify the contruction of the embeddings")

    # Embbed a single text command
    embed_text_parser = subparsers.add_parser("embed_text", help="Get an embedded respone from a text")
    embed_text_parser.add_argument("text", type=str, help="Text to get an embedded response")

    # Embed a query command
    embed_query_parser = subparsers.add_parser("embed_query", help="Get an embedded respones from a query")
    embed_query_parser.add_argument("query", type=str, help="The query to use for embedding")

    # Search Command from a query
    search_parser = subparsers.add_parser("search", help="Search a movie from a query")
    search_parser.add_argument("query", type=str, help="The query used to search a movie")
    search_parser.add_argument("--limit",type=int, nargs="?", default=MAX_RESULTS, help="select the limit of the search")

    # Chunking command
    chunk_parser = subparsers.add_parser("chunk", help="Creates chunks from a text")
    chunk_parser.add_argument("text", type=str, help="Text to divide in chunks")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE, help="The number of chunks the text will be divided into")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=DEFAULT_OVERLAP_SIZE,help="number of words that overlaps on each chunk")

    # Semantic chunk command
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Creates semantic chunks from a text")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to divide into semantic chunks")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Maximum size allowed for the chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Overlap between chunks allowed")

    # Embed chunks command
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Creates chunk embeddings from a document")

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

        case "search":
            semantic_search(args.query, args.limit)

        case "chunk":
            characters = len(args.text)
            chunks: list[str] = text_chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {characters} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {" ".join(chunk)}")

        case "semantic_chunk":
            characters: int = len(args.text)
            chunks: list[str] = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {characters} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            embed_chunks()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()