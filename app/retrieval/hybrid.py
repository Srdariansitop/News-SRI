class HybridSearcher:
    def __init__(self, bm25, vector_store, embedder, doc_metadata_map):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.doc_metadata_map = doc_metadata_map  

    def search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> list:
        """
        Realiza búsqueda híbrida usando Reciprocal Rank Fusion (RRF).
        rrf_k: Constante estándar para penalizar los rangos más bajos (usualmente 60).
        """
        # 1. Obtener resultados de BM25
        bm25_results = self.bm25.search(query, top_k=top_k)
        
        # 2. Obtener resultados Semánticos
        query_vector = self.embedder.encode(query)
        semantic_results = self.vector_store.search(query_vector, top_k=top_k)

        # 3. Aplicar Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # Procesar rankings de BM25
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.doc_id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)

        # Procesar rankings Semánticos
        for rank, result in enumerate(semantic_results, start=1):
            meta = result["metadata"]
            doc_id = meta if isinstance(meta, str) else meta.get("id")
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)

        # 4. Ordenar resultados finales por el score RRF de mayor a menor
        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        # 5. Formatear la salida
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            doc_data = self.doc_metadata_map.get(doc_id, {})
            final_results.append({
                "doc_id": doc_id,
                "score": score,
                "title": doc_data.get("title", "Sin título"),
                "source": doc_data.get("source", "N/A"),
                "url": doc_data.get("url", "N/A")
            })

        return final_results