import sys
import os

from app.indexing.index_builder import IndexBuilder
from app.retrieval.bm25 import BM25


def evaluate_bm25_params(queries: list[str]):
    builder = IndexBuilder()
    try:
        builder.load()
    except Exception as e:
        print(f"Error cargando el índice: {e}")
        return

    # Valores a probar según sugerencia (k1: 1.2-2.0, b: 0.5-0.9)
    k1_values = [1.5, 1.8, 2.0]
    b_values = [0.75, 0.8, 0.9]

    print("=== Herramienta de Afinamiento de Parámetros BM25 ===")
    stats = builder.get_stats()
    print(f"Documentos probados: {stats['total_documents']}")
    
    for query in queries:
        print(f"\n" + "="*50)
        print(f"Evaluando consulta: '{query}'")
        print("="*50)
        
        for k1 in k1_values:
            for b in b_values:
                print(f"\n--- Parámetros: k1={k1}, b={b} ---")
                bm25 = BM25(builder.index_title, builder.index_content, k1=k1, b=b)
                results = bm25.search(query, top_k=10)
                
                if not results:
                    print("  Sin resultados.")
                    continue
                
                for i, result in enumerate(results, 1):
                    metadata = builder.get_document_metadata(result.doc_id)
                    title = metadata.get("title", "Sin título") if metadata else "Sin título"
                    print(f"  {i}. [Score: {result.score:.4f}] {title} (ID: {result.doc_id[:8]}...)")

if __name__ == "__main__":
    test_queries = [
    # Multi-palabra
    "UK election voting rules",
    "AI threats research Google",
    "climate change effects environment",

    # Morfología
    "vote voting votes election",

    # IDF
    "rare penguin moult behavior",

    # Query larga
    "what are the risks of artificial intelligence in modern society",

    # Ruido
    "AI AI AI technology future",
    ]
    evaluate_bm25_params(test_queries)
