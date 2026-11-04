import argparse

from lib.semantic_search import verify_model

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model characteristics")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Displaying model info...")
            verify_model()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()