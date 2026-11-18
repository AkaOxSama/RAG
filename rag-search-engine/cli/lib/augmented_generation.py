import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

from .hybrid_search import(
    HybridSearch,
)
from .search_utils import(
    Movie,
    read_data,
    CombRRFData,
    SearchWithAIResponse,
    K_CONSTANT,
    MAX_RESULTS,
)
from .semantic_search import SemanticSearch


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


# RAG command to create an AI response to the search results
def rag_command(query: str) -> SearchWithAIResponse:

    # Initialize the movies data
    movies: list[Movie] = read_data()
    
    # Create instances of search classes
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    # Generate results via search
    search_result: list[CombRRFData] = hybrid_search.rrf_search(query, k=K_CONSTANT, limit=MAX_RESULTS)

    # Create the docs for the prompt
    docs: list[str] = []
    for doc in search_result:
        docs.append(doc["title"])

    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=[{
            "role":"user",
            "content": f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{"\n".join(docs)}

Answer:"""
        }]
    )

    content: str = (response.choices[0].message.content or "").strip()

    

    rag_search_result: SearchWithAIResponse = {
        "search_result": docs,
        "llm_response": content,
    }

    return rag_search_result

# Command to summarize the results provided via AI prompt
def summarize_command(query:str) -> SearchWithAIResponse:

    # Initialize the movies data
    movies: list[Movie] = read_data()
    
    # Create instances of search classes
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    # Generate results via search
    search_result: list[CombRRFData] = hybrid_search.rrf_search(query, k=K_CONSTANT, limit=MAX_RESULTS)

    # Create the docs for the prompt
    docs: list[str] = []
    for doc in search_result:
        docs.append(doc["title"])

    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=[{
            "role":"user",
            "content": f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{"\n".join(docs)}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
        }]
    )

    content = (response.choices[0].message.content or "").strip()

    summary_result: SearchWithAIResponse = {
        "search_result": docs,
        "llm_response": content,
    }

    return summary_result

# Command to create an AI response with citations based on a provided query and search result
def citations_command(query: str) -> SearchWithAIResponse:
    
    # Initialize the movies data
    movies: list[Movie] = read_data()
    
    # Create instances of search classes
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    # Generate results via search
    search_result: list[CombRRFData] = hybrid_search.rrf_search(query, k=K_CONSTANT, limit=MAX_RESULTS)

    # Create the docs for the prompt
    docs: list[str] = []
    for doc in search_result:
        docs.append(doc["title"])

    response = client.chat.completions.create(
        model=model, messages=[{
            "role": "user",
            "content": f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{"\n".join(docs)}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
        }]
    )

    content = (response.choices[0].message.content or "").strip()

    summary_result: SearchWithAIResponse = {
        "search_result": docs,
        "llm_response": content,
    }

    return summary_result

# Command to get and answer from a question using search and RAG
def question_command(question: str) -> SearchWithAIResponse:
    
    # Initialize the movies data
    movies: list[Movie] = read_data()
    
    # Create instances of search classes
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    # Generate results via search
    search_result: list[CombRRFData] = hybrid_search.rrf_search(question, k=K_CONSTANT, limit=MAX_RESULTS)

    # Create the docs for the prompt
    docs: list[str] = []
    for doc in search_result:
        docs.append(doc["title"])

    response = client.chat.completions.create(
        model=model, messages=[{
            "role": "user",
            "content": f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{"\n".join(docs)}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
        }]
    )

    content = (response.choices[0].message.content or "").strip()

    summary_result: SearchWithAIResponse = {
        "search_result": docs,
        "llm_response": content,
    }

    return summary_result

