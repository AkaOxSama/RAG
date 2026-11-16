import argparse

from lib.hybrid_search import (
    normalize,
    weighted_search_comm,
    rrf_search_comm,
)
from lib.search_utils import (
    MAX_RESULTS,
    ALPHA_CONSTANT,
    K_CONSTANT,
    print_base_result,
)

from lib.search_utils import RRFSearchCommandResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Normalize command
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a set of scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="List of scores to normalize")

    # Weighted search command
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Use a combined use of BM25 and semantic search logic to generate movie results")
    weighted_search_parser.add_argument("query", type=str, help="Query to search for movies")
    weighted_search_parser.add_argument("--alpha", nargs="?", type=float, default=ALPHA_CONSTANT, help="Constant to control the weight between the two scores")
    weighted_search_parser.add_argument("--limit", nargs="?", type=int, default=MAX_RESULTS, help="Maximum results returned")

    # RRF Search command
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Uses a RRF ranking to get movie results")
    rrf_search_parser.add_argument("query", type=str, help="Query to search for movies")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=K_CONSTANT, help="Constant to weight the searches")
    # Added support for enhanced queries using LLMs
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=MAX_RESULTS, help="Maximum results returned")
    # Added rerank argument
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank the search for better results")


    args = parser.parse_args()

    match args.command:

        case "normalize":
            scores: list[float] = normalize(args.scores)
            for value in scores:
                print(f"* {value:.4f}")

        case "weighted-search":
            weighted_search_comm(args.query, args.alpha, args.limit)

        case "rrf-search":
            result: RRFSearchCommandResult = rrf_search_comm(args.query, args.k, args.enhance, args.limit, args.rerank_method)

            is_enhanced: bool = result["enhanced_query"] is not None
            is_rerank: bool = result["rerank_method"] is not None
            
            
            if is_enhanced:
                method = result["enhance_method"]
                original = result["original_query"]
                enhanced = result["enhanced_query"]
                print(f"Enhanced query ({method}): '{original}' -> '{enhanced}'\n")
                print()
                for idx, data in enumerate(result["results"], 1):
                    print(f"{idx}. {data['title']}")
                    print_base_result(data)

            if is_rerank:
                print(f"Re-ranking top {args.limit} results using {result['rerank_method']} method")
                print(f"Reciprocal Rank Fusion Results for '{args.query if not is_enhanced else result["enhanced_query"]}' (k={args.k}):")
                print()
                match args.rerank_method:
                    case "individual":
                        for idx, data in enumerate(result["results"], 1):
                            print(f"{idx}. {data['title']}")
                            print(f"Re-rank Score: {data['rerank_score']:.4f}/10")
                            print_base_result(data)
                            print()

                    case "batch":
                        for idx, data in enumerate(result["results"], 1):
                            print(f"{idx}. {data['title']}")
                            print(f"Re-rank Rank: {idx}")
                            print(f"RRF Score: {data['rrf_score']:.4f}")
                            print(f"  BM25 Rank: {data.get("bm25_rank", 0)}, Semantic Rank: {data.get("sem_rank", 0)}")
                            print(f"  {data['description'][:100]}...")
                            print()

                    case "cross_encoder":
                        assert result["cross_scores"] is not None

                        for idx, data in enumerate(result["results"], 1):
                            print(f"{idx}. {data['title']}")
                            print(f"Cross Encoder Score: {result['cross_scores'][idx]:.4f}")
                            print(f"RRF Score: {data['rrf_score']:.4f}")
                            print(f"  BM25 Rank: {data.get("bm25_rank", 0)}, Semantic Rank: {data.get("sem_rank", 0)}")
                            print(f"  {data['description'][:100]}...")
                            print()

                    case _:
                        for idx, data in enumerate(result["results"], 1):
                            print(f"{idx}. {data['title']}")
                            print_base_result(data)
                            print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()