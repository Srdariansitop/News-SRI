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
    def __init__(self, index_title: InvertedIndex, index_content: InvertedIndex, k1: float = 1.5, b: float = 0.8):
        self.index_title = index_title
        self.index_content = index_content
        self.k1 = k1
        self.b = b
        self.avgdl_title = index_title.get_average_doc_length()
        self.avgdl_content = index_content.get_average_doc_length()
        self.N = max(index_title.doc_count, index_content.doc_count)

    def _idf(self, term: str, index: InvertedIndex) -> float:
        df = index.get_document_frequency(term)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _score_document_on_index(self, doc_id: str, query_term_freqs: Dict[str, int], index: InvertedIndex, avgdl: float) -> float:
        score = 0.0
        dl = index.doc_lengths.get(doc_id, 0)
        if avgdl == 0:
            return 0.0

        for term, qtf in query_term_freqs.items():
            tf = index.get_term_frequency(term, doc_id)
            if tf == 0:
                continue

            idf = self._idf(term, index)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score += (idf * numerator / denominator) * qtf

        return score

    def _score_document(self, doc_id: str, query_term_freqs: Dict[str, int]) -> float:
        score_title = self._score_document_on_index(doc_id, query_term_freqs, self.index_title, self.avgdl_title)
        score_content = self._score_document_on_index(doc_id, query_term_freqs, self.index_content, self.avgdl_content)
        
        # score_total = (score_titulo * 2) + (score_contenido * 1)
        return (score_title * 2.0) + (score_content * 1.0)

    def search(self, query: str, top_k: int = 10, min_match_ratio: float = 0.0) -> List[SearchResult]:
        preprocessor = get_preprocessor()
        original_query_terms = preprocessor.process_to_terms(query)

        if not original_query_terms:
            return []

        # Remove duplicates to count unique matched terms
        query_terms = list(set(original_query_terms))
        
        # Minimum matches required (e.g. >= 2 if query has 2 or more unique terms)
        # It's clamped to the number of unique terms to avoid filtering out 1-word queries
        min_matches = min(2, len(query_terms))

        doc_matches = {}
        for term in query_terms:
            docs_with_term = set(self.index_title.get_docs_containing_term(term)) | \
                             set(self.index_content.get_docs_containing_term(term))
            
            for doc_id in docs_with_term:
                if doc_id not in doc_matches:
                    doc_matches[doc_id] = 0
                doc_matches[doc_id] += 1

        # Filter candidate docs
        candidate_docs = [
            doc_id for doc_id, match_count in doc_matches.items()
            if match_count >= min_matches
        ]

        # Calculate query term frequencies
        query_term_freqs = {}
        for term in original_query_terms:
            query_term_freqs[term] = query_term_freqs.get(term, 0) + 1

        results = []
        for doc_id in candidate_docs:
            score = self._score_document(doc_id, query_term_freqs)
            if score > 0:
                results.append(SearchResult(doc_id=doc_id, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]