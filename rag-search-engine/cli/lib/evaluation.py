import os
import json

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
    read_golden_data,
    CombRRFData,
    GoldenDataset,
    PrecisionKResults,
    RRFSearchCommandResult,
    K_CONSTANT,
    PRECISION_K
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


# Creates precision, recall and f1 scores comparing search results and golden dataset information
def precision_k_command(limit: int = PRECISION_K) -> dict:
    movies: list[Movie] = read_data()
    golden_dataset: list[GoldenDataset] = read_golden_data()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    relevant_count = 0
    data_by_query: dict[str, PrecisionKResults] = {}
    for data in golden_dataset:
        query: str = data['query']
        relevant_docs: set[str] = set(data['relevant_docs'])
        search_results: list[CombRRFData] = hybrid_search.rrf_search(query=query, k=K_CONSTANT, limit=limit)

        retrieved_docs: list[str] = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)
                if title in relevant_docs:
                    relevant_count += 1

        precision: float = calculate_k_prec(retrieved_docs, relevant_docs, limit)
        recall: float = calculate_k_recall(retrieved_docs, relevant_docs, limit)

        f1: float = 2 * (precision * recall) / (precision + recall)

        data_by_query[query] = {
            "precision_k": precision,
            "f1_score": f1,
            "recall": recall,
            "retrieved": retrieved_docs,
            "relevant": list(relevant_docs),
        }

    return data_by_query

# Creates an evaluation of each search result via AI prompt
def evaluate_command(query: str, search_results: RRFSearchCommandResult) -> list[str]:

    formatted_results: list[str] = []
    for result in search_results["results"]:
        formatted_results.append(result["title"])

    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=[{
            "role": "user",
            "content": f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
        }]
    )

    content: list[int] = json.loads((response.choices[0].message.content) or "[]")

    relevant_scores: list[str] = []
    for i in range(len(formatted_results)):
        relevant_scores.append(f"{i+1}. {formatted_results[i]}: {content[i]}/3")

    return relevant_scores


# Helper function to calculate precision at K
def calculate_k_prec(retrieved_docs: list[str], relevant_docs: set[str], k: int = PRECISION_K) -> float:
    top_k = retrieved_docs[:k]
    relevant: int = 0

    for doc in top_k:
        if doc in relevant_docs:
            relevant += 1

    return relevant / k

def calculate_k_recall(retrieved_docs: list[str], relevant_docs: set[str], k: int = PRECISION_K) -> float:
    top_k = retrieved_docs[:k]
    relevant: int = 0

    for doc in top_k:
        if doc in relevant_docs:
            relevant += 1

    return relevant / len(relevant_docs)