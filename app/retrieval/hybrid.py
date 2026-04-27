from app.web import WebSearcher, SufficiencyChecker
import math


class HybridSearcher:
    def __init__(self, bm25, vector_store, embedder, doc_metadata_map, enable_web_search: bool = True):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.doc_metadata_map = doc_metadata_map
        self.enable_web_search = enable_web_search
        
        # Inicializa el buscador web y verificador de suficiencia
        if enable_web_search:
            self.web_searcher = WebSearcher()
            self.sufficiency_checker = SufficiencyChecker(min_results=3, min_avg_score=0.3)  

    def search(self, query: str, top_k: int = 10, rrf_k: int = 60, semantic_threshold: float = 0.3) -> list:
        """
        Realiza búsqueda híbrida usando Reciprocal Rank Fusion (RRF).
        Filtra los resultados devolviendo SOLO aquellos que superan el semantic_threshold.
        Umbral recomendado: 0.5 (balance entre relevancia y cobertura)
        """
        # 1. Obtener resultados de BM25
        bm25_results = self.bm25.search(query, top_k=top_k)
        
        # 2. Obtener resultados Semánticos
        query_vector = self.embedder.encode(query)
        semantic_results = self.vector_store.search(query_vector, top_k=top_k)

        # 2.1 Guardamos los scores semánticos reales en un diccionario para validarlos luego
        semantic_scores = {}
        for result in semantic_results:
            meta = result["metadata"]
            doc_id = meta if isinstance(meta, str) else meta.get("id")
            semantic_scores[doc_id] = result["score"]

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

        # 5. Formatear la salida Y APLICAR EL UMBRAL
        final_results = []
        for doc_id, score in sorted_results:
            # Obtenemos el score semántico del documento (0.0 si solo lo encontró BM25)
            doc_semantic_score = semantic_scores.get(doc_id, 0.0)
            
            # FILTRO CRÍTICO: Solo agregamos si cumple el umbral de similitud semántica
            if doc_semantic_score >= semantic_threshold:
                doc_data = self.doc_metadata_map.get(doc_id, {})
                print(f"✓ Documento aceptado - ID: {doc_id} | Score Semántico: {doc_semantic_score:.4f} >= Umbral: {semantic_threshold} | Título: {doc_data.get('title', 'Sin título')[:50]}")
                
                final_results.append({
                    "doc_id": doc_id,
                    "score": score, # Score RRF
                    "semantic_score": doc_semantic_score, 
                    "title": doc_data.get("title", "Sin título"),
                    "source": doc_data.get("source", "N/A"),
                    "url": doc_data.get("url", "N/A")
                })
                
                # Detenemos si ya llenamos el top_k de documentos VÁLIDOS
                if len(final_results) == top_k:
                    break

        # 6. BÚSQUEDA WEB: Si los resultados locales son insuficientes, busca en la web
        if self.enable_web_search:
            is_sufficient, reason = self.sufficiency_checker.is_sufficient(final_results)
            
            if not is_sufficient:
                print(f"\n⚠️ Resultados insuficientes: {reason}")
                print(f"🌐 Activando búsqueda web para: '{query}'")
                
                web_results = self.web_searcher.search(query, top_k=top_k - len(final_results))
                
                if web_results:
                    print(f"✅ Se encontraron {len(web_results)} resultados en la web\n")
                    
                    for idx, web_result in enumerate(web_results, start=1):
                        # Los resultados web tienen menor confianza que BBC
                        # Usamos fórmula logarítmica para que decrezca suavemente pero NUNCA sea negativo
                        web_score = max(0.0001, 1.0 / (2.0 + idx))  # Asegura positivo: 0.33, 0.25, 0.2, 0.16...
                        final_results.append({
                            "doc_id": f"web_{idx}",
                            "score": web_score,
                            "semantic_score": 0.0,
                            "title": web_result.get("title", "Sin título"),
                            "source": web_result.get("source", "web"),
                            "url": web_result.get("url", "N/A"),
                            "snippet": web_result.get("snippet", ""),
                            "from_web": True
                        })
                else:
                    print("❌ No se encontraron resultados en la web\n")
            else:
                print(f"✅ {reason}\n")

        return final_results