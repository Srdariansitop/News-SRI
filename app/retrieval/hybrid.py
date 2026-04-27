from app.web import WebSearcher, SufficiencyChecker
from app.maintenance.web_document_manager import WebDocumentManager
import math


class HybridSearcher:
    def __init__(self, bm25, vector_store, embedder, doc_metadata_map, enable_web_search: bool = True, save_web_results: bool = True):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.doc_metadata_map = doc_metadata_map
        self.enable_web_search = enable_web_search
        self.save_web_results = save_web_results
        
        # Inicializa el buscador web y verificador de suficiencia
        if enable_web_search:
            self.web_searcher = WebSearcher()
            self.sufficiency_checker = SufficiencyChecker(min_results=3, min_avg_score=0.3)
            
            # Inicializa gestor de documentos web consolidado
            if save_web_results:
                self.web_manager = WebDocumentManager()
            else:
                self.web_manager = None
        else:
            self.web_manager = None  

    def search(self, query: str, top_k: int = 10, rrf_k: int = 60, semantic_threshold: float = 0.3) -> list:
        """
        Realiza búsqueda híbrida usando Reciprocal Rank Fusion (RRF).
        Filtra los resultados devolviendo SOLO aquellos que superan el semantic_threshold.
        Umbral recomendado: 0.5 (balance entre relevancia y cobertura)
        """
        # 1. Obtener resultados de BM25
        bm25_results = self.bm25.search(query, top_k=top_k)
        bm25_doc_ids = {result.doc_id for result in bm25_results}  # Guardamos IDs para validar luego
        
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

        # 5. Formatear la salida Y APLICAR EL UMBRAL (SOLO si vino de búsqueda semántica)
        final_results = []
        for doc_id, score in sorted_results:
            # Obtenemos el score semántico del documento (0.0 si solo lo encontró BM25)
            doc_semantic_score = semantic_scores.get(doc_id, 0.0)
            bm25_found = doc_id in bm25_doc_ids  # ✓ Verificamos si BM25 lo encontró
            
            # LÓGICA MEJORADA: 
            # - Si BM25 lo encontró, SIEMPRE incluirlo (confía en BM25)
            # - Si SOLO vino de semántica, requiere el umbral
            should_include = bm25_found or (doc_semantic_score >= semantic_threshold)
            
            if should_include:
                doc_data = self.doc_metadata_map.get(doc_id, {})
                
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
                    
                    # 🔄 GUARDAR RESULTADOS WEB AUTOMÁTICAMENTE
                    if self.save_web_results and self.web_manager:
                        print("💾 Guardando resultados web en data/raw/...")
                        saved_docs = self.web_manager.save_multiple_web_results(web_results, source="web", auto_index=False)
                        # No printear aquí - el método ya printea un mensaje detallado
                    
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
    
    def get_web_storage_stats(self) -> dict:
        """Retorna estadísticas del almacenamiento web."""
        if self.web_manager:
            return self.web_manager.get_statistics()
        return {"status": "Web storage not enabled"}