import argparse

from lib.multimodal_search import(
    verify_img_embedding_command,
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Checks if the embeddings have been successfully built")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="The path to the image to embed")

    args = parser.parse_args()
    image_path = args.image_path

    match args.command:
        case "verify_image_embedding":
            print(f"Verifying embedding for: {image_path}\n")
            shape = verify_img_embedding_command(image_path)

            print(f"\nEmbedding shape: {shape} dimensions")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

