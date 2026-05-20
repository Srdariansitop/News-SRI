import sys
import os
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app.indexing.index_builder import IndexBuilder
from app.retrieval.bm25 import BM25
from app.vector.embeddings import EmbeddingGenerator
from app.vector.vector_store import VectorStore
from app.retrieval.hybrid import HybridSearcher


def load_raw_documents():
    """Carga documentos desde data/raw"""
    documents = []
    raw_path = "data/raw"
    
    if not os.path.exists(raw_path):
        return documents
    
    for filename in os.listdir(raw_path):
        if filename.startswith(".") or not filename.endswith(".json"):
            continue
        
        path = os.path.join(raw_path, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "id" in data:
                documents.append(data)
        except:
            continue
    
    return documents


def execute_hybrid_search(query):
    """
    Ejecuta búsqueda híbrida completa (RAG) para una query
    Retorna top 10 resultados
    """
    try:
        # 1. Cargar documentos
        documents = load_raw_documents()
        doc_map = {doc["id"]: doc for doc in documents}
        
        if not doc_map:
            print("❌ No hay documentos en data/raw/")
            return None
        
        print(f"   ℹ️ {len(doc_map)} documentos cargados")
        
        # 2. Cargar BM25
        builder = IndexBuilder()
        try:
            builder.load()
            bm25 = BM25(builder.index)
            print(f"   ℹ️ BM25 index cargado")
        except Exception as e:
            print(f"❌ Error al cargar BM25: {type(e).__name__}: {e}")
            return None
        
        # 3. Cargar embeddings
        try:
            embedder = EmbeddingGenerator()
            vector_store = VectorStore(embedder.get_dimension())
            vector_store.load("data/vector_db")
            print(f"   ℹ️ Vector store cargado")
        except Exception as e:
            print(f"❌ Error al cargar embeddings: {type(e).__name__}: {e}")
            return None
        
        # 4. Inicializar búsqueda híbrida
        try:
            hybrid_searcher = HybridSearcher(
                bm25=bm25,
                vector_store=vector_store,
                embedder=embedder,
                doc_metadata_map=doc_map,
                enable_web_search=True,
                save_web_results=True
            )
            print(f"   ℹ️ HybridSearcher inicializado")
        except Exception as e:
            print(f"❌ Error al inicializar HybridSearcher: {type(e).__name__}: {e}")
            return None
        
        # 5. Ejecutar búsqueda
        try:
            results = hybrid_searcher.search(query, top_k=10, semantic_threshold=0.7)
        except Exception as e:
            print(f"❌ Error ejecutando búsqueda: {type(e).__name__}: {e}")
            return None
        
        if not results:
            print(f"   ⚠️ No se encontraron resultados para: {query}")
            return []
        
        # 6. Formatear resultados (máximo 10)
        formatted_results = []
        for i, result in enumerate(results[:10], 1):  # Limitar a 10
            formatted_results.append({
                "rank": i,
                "doc_id": result.get("doc_id") or result.get("id"),
                "title": result.get("title", "Sin título"),
                "url": result.get("url", "N/A"),
                "source": result.get("source", "N/A"),
                "judgment": None
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"   ❌ Error general: {type(e).__name__}: {e}")
        return None


def get_predefined_queries():
    return [
    "plantillas del Mundial 2026",
    "últimas actualizaciones sobre la guerra en Irán",
    "PIB de Kuwait 2026",
    "noticias sobre Anthropic y Claude",
    "últimos avances del régimen de Corea del Norte",
    "acusaciones de golpe de Estado contra Evo Morales",
    "fallos de productos de IA de Google en 2026",
    "mejores brokers para operar en 2026",
    "últimas actualizaciones sobre el hantavirus",
    "documentales sobre asesinos en serie"
    ]


def save_test_queries(queries_data, output_path="data/evaluation/test_queries.json"):
    """Guarda queries con resultados en JSON"""
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "queries": queries_data,
        "metadata": {
            "total_queries": len(queries_data),
            "instructions": "Complete el campo 'relevance' con 0 (no relevante), 1 (relevante) o 2 (muy relevante)"
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Queries guardadas en: {output_path}")


def generate_test_queries_interactive():
    """Función principal"""
    
    print("\n" + "="*70)
    print("🔬 GENERADOR DE QUERIES DE PRUEBA - BÚSQUEDA HÍBRIDA COMPLETA (RAG)")
    print("="*70)
    
    # 1. Obtener queries predefinidas
    queries = get_predefined_queries()
    
    print(f"\n📋 QUERIES A PROCESAR ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")
    
    print("\n" + "="*70)
    print(f"🔄 EJECUTANDO BÚSQUEDAS PARA {len(queries)} QUERIES...")
    print("="*70)
    print("(Esto puede tardar un momento, especialmente con búsqueda web activada)\n")
    
    # 2. Ejecutar búsquedas
    queries_data = []
    
    for idx, query in enumerate(queries, 1):
        print(f"[{idx}/{len(queries)}] Procesando: '{query}'")
        
        results = execute_hybrid_search(query)
        
        if results is None:
            print("   ⚠️ Saltando esta query debido a error\n")
            continue
        
        query_data = {
            "id": f"q{idx}",
            "query": query,
            "results": results
        }
        
        queries_data.append(query_data)
        print(f"   ✅ {len(results)} resultados obtenidos\n")
    
    if not queries_data:
        print("❌ No se pudo procesar ninguna query.")
        return
    
    # 3. Guardar en JSON
    save_test_queries(queries_data)

if __name__ == "__main__":
    generate_test_queries_interactive()
