

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
    K_CONSTANT,
    PRECISION_K
)
from .semantic_search import SemanticSearch

def precision_k_command(limit: int = PRECISION_K) -> dict:
    movies: list[Movie] = read_data()
    golden_dataset: list[GoldenDataset] = read_golden_data()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    
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

        precision: float = calculate_k_prec(retrieved_docs, relevant_docs, limit)

        data_by_query[query] = {
            "precision_k": precision,
            "retrieved": retrieved_docs,
            "relevant": list(relevant_docs),
        }

    return data_by_query
            


# Helper function to calculate precision at K
def calculate_k_prec(retrieved_docs: list[str], relevant_docs: set[str], k: int = PRECISION_K) -> float:
    top_k = retrieved_docs[:k]
    relevant: int = 0

    for doc in top_k:
        if doc in relevant_docs:
            relevant += 1

    return relevant / k