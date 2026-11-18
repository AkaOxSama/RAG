import argparse

from lib.multimodal_search import(
    verify_img_embedding_command,
    image_search_command,
)
from lib.search_utils import(
    ImageSearchResult,
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify command
    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Checks if the embeddings have been successfully built")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="The path to the image to embed")
    # Image search command
    image_search_parser = subparsers.add_parser("image_search", help="Create search results from an image path")
    image_search_parser.add_argument("image_path_search", type=str, help="The path of the image to create the search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            print(f"Verifying embedding for: {image_path}\n")
            shape = verify_img_embedding_command(image_path)

            print(f"\nEmbedding shape: {shape} dimensions")

        case "image_search":
            image_path_search = args.image_path_search
            print(f"Searching for movies from image path: {image_path_search}\n")

            search_results: list[ImageSearchResult] = image_search_command(image_path_search)
            print()
            for idx, result in enumerate(search_results, 1):
                print(f"{idx}. {result['title']} (similarity: {result['score']:.1f})")
                print(f"{result['description'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

