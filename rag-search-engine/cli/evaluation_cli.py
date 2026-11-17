import argparse

from lib.search_utils import(
    PRECISION_K,
    PrecisionKResults,
)
from lib.evaluation import(
    precision_k_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    
    print(f"k={limit}")
    print()
    precision_results: dict[str, PrecisionKResults] = precision_k_command(limit)

    for query, data in precision_results.items():
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {data['precision_k']:.4f}")
        print(f"  - Recall@{limit}: {data['recall']:.4f}")
        print(f"  - F1 Score: {data['f1_score']:.4f}")
        print(f"  - Retrieved: {data['retrieved']}")
        print(f"  - Relevant: {data['relevant']}")
        print()



if __name__ == "__main__":
    main()