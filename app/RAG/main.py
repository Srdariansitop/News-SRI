import os
import json
from app.RAG.rag import RAGSystem
from app.retrieval.hybrid import HybridSearcher
from app.retrieval.bm25 import BM25
from app.vector.embeddings import EmbeddingGenerator
from app.vector.vector_store import VectorStore
from app.indexing.index_builder import IndexBuilder

def load_raw_documents():
    """Carga documentos desde data/raw"""
    RAW_DATA_PATH = "data/raw"
    documents = []
    
    for filename in os.listdir(RAW_DATA_PATH):
        if filename.startswith(".") or not filename.endswith(".json"):
            continue
            
        path = os.path.join(RAW_DATA_PATH, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "id" in data:
                documents.append(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    return documents

def main():
    print("="*50)
    print("🚀 TEST DEL SISTEMA RAG CON GROQ")
    print("="*50)
    
    # 1. Cargar documentos reales
    print("\n[*] Cargando documentos...")
    documents = load_raw_documents()
    doc_map = {doc["id"]: doc for doc in documents}
    
    if not doc_map:
        print("❌ No hay documentos en data/raw. Ejecuta primero el crawling.")
        return
    
    print(f"✅ Se cargaron {len(doc_map)} documentos")
    
    # 2. Cargar BM25
    print("[*] Cargando índice BM25...")
    try:
        builder = IndexBuilder()
        builder.load()
        bm25 = BM25(builder.index)
        print("✅ Índice BM25 cargado")
    except Exception as e:
        print(f"❌ Error al cargar BM25: {e}")
        return
    
    # 3. Cargar embedder y vector store
    print("[*] Cargando embeddings...")
    try:
        embedder = EmbeddingGenerator()
        vector_store = VectorStore(embedder.get_dimension())
        vector_store.load("data/vector_db")
        print("✅ Vector store cargado")
    except Exception as e:
        print(f"❌ Error al cargar vector store: {e}")
        return
    
    # 4. Inicializar HybridSearcher REAL
    print("[*] Inicializando buscador híbrido...")
    hybrid_searcher = HybridSearcher(
        bm25=bm25, 
        vector_store=vector_store, 
        embedder=embedder, 
        doc_metadata_map=doc_map,
        enable_web_search=False,  # Desactivar búsqueda web para pruebas
        save_web_results=False
    )
    print("✅ Buscador híbrido listo\n")
    
    # 5. Inicialización del RAG
    print("[*] Inicializando sistema RAG...")
    rag = RAGSystem(hybrid_searcher=hybrid_searcher, raw_data_path="data/raw")    
    # 6. Hacer la prueba de pregunta
    pregunta = input("\n👤 Ingrese su pregunta: ")
    if not pregunta.strip():
        print("❌ Pregunta vacía.")
        return
        
    print(f"\n[*] Procesando: '{pregunta}'")
    print("[*] Buscando documentos relevantes...")
    print("[*] Generando respuesta con RAG (esto puede tardar unos segundos)...")
    
    try:
        resultado = rag.answer(pregunta)
        
        print("\n" + "="*50)
        print("🤖 RESPUESTA DE LA IA:")
        print("="*50)
        print(resultado["summary"])
        print("="*50)
        
        print("\n📚 Documentos consultados:")
        for doc in resultado["documents"]:
            print(f" - [{doc['doc_id']}] {doc['title']} (Score: {doc['score']:.4f})")
            print(f"   URL: {doc['url']}")
    
    except Exception as e:
        print(f"\n❌ Error en el RAG: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
