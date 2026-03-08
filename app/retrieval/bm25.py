import math
from dataclasses import dataclass
from typing import Dict, List

from app.core.preprocessor import get_preprocessor
from app.indexing.inverted_index import InvertedIndex


@dataclass
class SearchResult:
    doc_id: str
    score: float


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.avgdl = index.get_average_doc_length()
        self.N = index.doc_count

    def _idf(self, term: str) -> float:
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _score_document(self, doc_id: str, query_terms: List[str]) -> float:
        score = 0.0
        dl = self.index.doc_lengths.get(doc_id, 0)

        for term in query_terms:
            tf = self.index.get_term_frequency(term, doc_id)
            if tf == 0:
                continue

            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        preprocessor = get_preprocessor()
        query_terms = preprocessor.process_to_terms(query)

        if not query_terms:
            return []

        candidate_docs = set()
        for term in query_terms:
            candidate_docs.update(self.index.get_docs_containing_term(term))

        results = []
        for doc_id in candidate_docs:
            score = self._score_document(doc_id, query_terms)
            if score > 0:
                results.append(SearchResult(doc_id=doc_id, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]