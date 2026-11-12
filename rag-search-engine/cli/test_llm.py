import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

# API Key loader
load_dotenv()
print(os.getcwd())
print("key loaded:", bool(os.environ.get("OPENROUTER_API_KEY")))
api_key = os.environ.get("OPENROUTER_API_KEY")
print(api_key is None)
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")



# OpenAI chatbot
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]

response: ChatCompletion = client.chat.completions.create(messages=messages, model="openrouter/free")

print(response.choices[0].message.content)
print()
assert response.usage
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Response tokens: {response.usage.completion_tokens}")

