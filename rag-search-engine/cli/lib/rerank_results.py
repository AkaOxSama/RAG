import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from .search_utils import(
    CombRRFData,
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


# Function to create movie results through AI to create a new score based on the query and it's relevance with the title and the description
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

# Function to create movie results through AI to create a new list of scores based on the query and it's relevance through all the results
def rerank_list_score_results(query: str, documents: list[CombRRFData], limit: int) -> list[int]:
    doc_map: list[str] = []

    for doc in documents:
        doc_map.append(f"ID: {doc['movie_id']}, Title: {doc['title']}, Description: {doc['description']}")

    doc_list_str = "\n".join(doc_map)


    response = client.chat.completions.create(
        model=model, messages=[{
            "role": "user",
            "content": f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
        }]
    )

    content: list[int] = json.loads((response.choices[0].message.content) or "[]")
    return content[:limit]