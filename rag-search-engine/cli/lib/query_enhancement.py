import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from lib.search_utils import prompt 

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

def spell_correction(query: str) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": f"{prompt}\nUser query: {query}"}]
    )
    correct_query: str = (response.choices[0].message.content or "").strip().strip('"')
    return correct_query if correct_query else query

def enhance_query(query: str, method:str | None = None) -> str:
    match method:
        case "spell":
            return spell_correction(query)
        case _:
            return query