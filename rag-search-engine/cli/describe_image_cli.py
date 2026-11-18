import argparse

from lib.describe_image import(
    describe_img_command,
)
from lib.search_utils import(
    ImageResponse,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="Path to the image to start the search")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()
    image = args.image
    query = args.query

    response: ImageResponse = describe_img_command(image, query)
    print(f"Rewritten query: {response['response']}")
    if response["total_tokens"] > 0:
        print(f"Total tokens:   {response['total_tokens']}")


if __name__ == "__main__":
    main()