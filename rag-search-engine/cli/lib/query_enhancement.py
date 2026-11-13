import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from lib.search_utils import (
    spell_prompt,
    rewrite_prompt,
    expand_prompt,
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


def query_modification(query: str, method: str | None) -> str:
    prompt = ""
    match method:
        case "spell":
            prompt = spell_prompt
        case "rewrite":
            prompt = rewrite_prompt
        case "expand":
            prompt = expand_prompt
        case _:
            return query

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": f"{prompt}\nUser query: {query}"}]
    )
    modified_query = (response.choices[0].message.content or "").strip().strip('"')

    if method == "expand":
        if modified_query:
            return f"{query} {modified_query}"

    return modified_query if modified_query else query

