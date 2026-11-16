import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

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


# Function to rerank movies result via AI to create a new score based on the query
def rerank_score_results(query: str, title: str, document: str) -> int:
    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=[{
            "role": "user",
            "content": f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
        }]
    )

    content = (response.choices[0].message.content or "").strip()

    match = re.search(r'\b(10|[0-9])\b', content)

    if not match:
        return 0
    return int(match.group(1))