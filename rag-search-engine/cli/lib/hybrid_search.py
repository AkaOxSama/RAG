import os

from .keyword_search import InvertedIndex
from .search_utils import ChunkMovieResults
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    read_data,
    Movie,
    BM25ScoreMovies,
    HybridScoresData,
    CombScoreData,
    MAX_RESULTS,
    ALPHA_CONSTANT,
    MAX_SUB_SEARCH,
)
from typing import Any, TypeGuard


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.doc_map: dict[int, Movie] = InvertedIndex().docmap
        self.documents: list[Movie] = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    # Imported method to get bm25 score of a query
    def _bm25_search(self, query: str, limit: int) -> list[BM25ScoreMovies]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    # Creates a weighted search combining keyword and semantic search
    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridScoresData]:
        bm25_search: list[BM25ScoreMovies] = self._bm25_search(query, limit=MAX_SUB_SEARCH)
        semantic_search: list[ChunkMovieResults] = self.semantic_search.search_chunks(query, limit=MAX_SUB_SEARCH)

        bm25_norm= normalize_search_scores(bm25_search)
        semantic_norm = normalize_search_scores(semantic_search)

        comb_results: dict[int, CombScoreData] = {}
        for norm_score, bm25_data in bm25_norm:
            doc_id: int = bm25_data["movie_id"]
            if doc_id not in comb_results:
                comb_results[doc_id] = {
                    "title": "",
                    "description": "",
                    "bm25_score": 0.0,
                    "sem_score": 0.0,
                    "hyb_score": 0.0,
                }
            comb_results[doc_id]["bm25_score"] = max(comb_results[doc_id]["bm25_score"], norm_score)

        for norm_score, semantic_data in semantic_norm:
            doc_id = semantic_data["movie_id"]
            if doc_id not in comb_results:
                comb_results[doc_id] = {
                    "title": semantic_data["title"],
                    "description": semantic_data["document"][:100],
                    "bm25_score": 0.0,
                    "sem_score": 0.0,
                    "hyb_score": 0.0,
                }
            comb_results[doc_id]["sem_score"] = max(comb_results[doc_id]["sem_score"], norm_score)
            comb_results[doc_id]["title"] = semantic_data["title"]
            comb_results[doc_id]["description"] = semantic_data["document"][:100]

        hybrid_results: list[HybridScoresData] = []
        for doc_id, data in comb_results.items():
            hybrid_score: float = _hybrid_score(data["bm25_score"], data["sem_score"], alpha)

            hybrid_results.append(
                {
                    "title": data["title"],
                    "description": data["description"],
                    "bm25_sc": data["bm25_score"],
                    "sem_sc": data["sem_score"],
                    "hyb_sc": hybrid_score
                }
            )

        return sorted(hybrid_results, key=lambda x: x["hyb_sc"], reverse=True)[:limit]




    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


# Command to normalize scores
def normalize(scores:list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_s: float = min(scores)
    max_s: float = max(scores)

    norm_scores = []

    if min_s == max_s:
        norm_scores: list[float] = [1.0] * len(scores)
    else:
        for i in range(len(scores)):
            norm_score: float = (scores[i] - min_s) / (max_s - min_s)
            norm_scores.append(norm_score)

    return norm_scores

# Command to get movie searches based on a hybrid method
def weighted_search_comm(query: str, alpha=ALPHA_CONSTANT, limit=MAX_RESULTS) -> None:
    movies: list[Movie] = read_data()
    hybrid_search = HybridSearch(movies)

    search_results: list[HybridScoresData] = hybrid_search.weighted_search(query, alpha, limit)

    for idx, result in enumerate(search_results, 1):
        print(f"{idx}. {result['title']}")
        print(f"  Hybrid Score: {result['hyb_sc']:.4f}")
        print(f"  BM25: {result['bm25_sc']:.4f}, Semantic: {result['sem_sc']:.4f}")
        print(f"  {result['description']}...")

# Helper function to get the hybrid score from the bm25 and semantic scores
def _hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

# Helper function to return the normalize scores of a search
def normalize_search_scores(results):
    scores: list[float] = []

    for result in results:
        scores.append(result["score"])

    norm_scores: list[float] = normalize(scores)
    return list(zip(norm_scores, results))