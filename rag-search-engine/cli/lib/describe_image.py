import mimetypes
import base64
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from .search_utils import(
    ImageResponse,
)

# API Key loader
load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

# OpenAI chatbot
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
model = "openrouter/free"

instructions = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""

# Command to get an image and create a query with it's description
def describe_img_command(image: str, query: str) -> ImageResponse:
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image file not found: {image}")
    
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        img = f.read()

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions.strip()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": query.strip()},
                ],
            }
        ]
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("No text in API response")

    tokens_used = 0
    if response.usage is not None:
        tokens_used = response.usage.total_tokens

    return {
        "response": content.strip(),
        "total_tokens": tokens_used,
    }    

