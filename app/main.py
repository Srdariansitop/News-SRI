import sys
import os
import json
from app.crawler.rss_crawler import crawl
from app.indexing.index_builder import IndexBuilder
from app.retrieval.bm25 import BM25
from app.vector.embeddings import EmbeddingGenerator
from app.vector.vector_store import VectorStore
from app.maintenance.cleaner import DataCleaner 
from app.retrieval.hybrid import HybridSearcher

RAW_DATA_PATH = "data/raw"


def load_raw_documents():

    documents = []

    for filename in os.listdir(RAW_DATA_PATH):

        if not filename.endswith(".json"):
            continue

        path = os.path.join(RAW_DATA_PATH, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents.append(data)

    return documents

def run_embeddings():

    print("\n🧠 Generando embeddings desde documentos RAW...\n")

    documents = load_raw_documents()

    if not documents:
        print("❌ No hay documentos en data/raw. Ejecuta primero el crawling.")
        return

    embedder = EmbeddingGenerator()

    dimension = embedder.get_dimension()

    vector_store = VectorStore(dimension)

    texts = []
    ids = []

    for doc in documents:

        doc_id = doc["id"]

        text = f"{doc.get('title','')} {doc.get('content','')}"

        texts.append(text)

        ids.append(doc_id)

    print(f"📄 Generando embeddings para {len(texts)} documentos...")

    vectors = embedder.encode_batch(texts)

    vector_store.add(vectors, ids)

    vector_store.save("data/vector_db")

    print("\n✅ Embeddings generados y guardados.")



def run_semantic_search():

    print("\n🔎 Búsqueda semántica...\n")

    documents = load_raw_documents()

    doc_map = {doc["id"]: doc for doc in documents}

    embedder = EmbeddingGenerator()

    vector_store = VectorStore(embedder.get_dimension())

    vector_store.load("data/vector_db")

    query = input("📝 Ingrese su consulta: ")

    query_vector = embedder.encode(query)

    results = vector_store.search(query_vector, top_k=10)

    print("\n🏆 Resultados:\n")

    for i, result in enumerate(results, 1):

        doc_id = result["metadata"] 
        score = result["score"]     

        doc = doc_map.get(doc_id)

        if not doc:
            continue

        print(f"{i}. {doc.get('title','Sin título')}")
        print(f"   Score: {score:.4f}")
        print(f"   Fuente: {doc.get('source','N/A')}")
        print(f"   URL: {doc.get('url','N/A')}")
        print()

def run_hybrid_search():
    print("\n🤝 Búsqueda Híbrida (BM25 + Semántica)...\n")

    # 1. Cargar metadatos originales de los documentos
    documents = load_raw_documents()
    doc_map = {doc["id"]: doc for doc in documents}

    if not doc_map:
        print("❌ No hay documentos. Ejecuta primero el crawling.")
        return

    # 2. Cargar índice BM25
    builder = IndexBuilder()
    try:
        builder.load()
        bm25 = BM25(builder.index)
    except Exception as e:
        print(f"❌ Error al cargar BM25. ¿Ejecutaste la indexación? ({e})")
        return

    # 3. Cargar Base de Datos Vectorial
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(embedder.get_dimension())
    try:
        vector_store.load("data/vector_db")
    except Exception as e:
        print(f"❌ Error al cargar FAISS. ¿Generaste los embeddings? ({e})")
        return

    # 4. Inicializar buscador híbrido
    hybrid_searcher = HybridSearcher(bm25, vector_store, embedder, doc_map)

    # 5. Ejecutar consulta
    query = input("📝 Ingrese su consulta: ")
    if not query.strip():
        print("Consulta vacía. Volviendo al menú.")
        return

    results = hybrid_searcher.search(query, top_k=10)

    if not results:
        print("❌ No se encontraron resultados.")
        return

    print(f"\n🏆 Top {len(results)} Resultados Híbridos:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['score']:.4f} RRF] {result['title']}")
        print(f"   Fuente: {result['source']}")
        print(f"   URL: {result['url']}\n")

def run_crawling():
    print("\n🚀 Iniciando crawling...\n")
    crawl()
    print("\n✅ Crawling finalizado.\n")


def run_indexing():
    print("\n📚 Iniciando construcción del índice...\n")
    builder = IndexBuilder()
    count = builder.build()
    
    print(f"📄 Se indexaron {count} documentos")
    builder.save()
    
    print("\n✅ Indexación finalizada.\n")


def run_search():
    print("\n🔍 Cargando índice...\n")
    builder = IndexBuilder()
    
    try:
        builder.load()
    except Exception as e:
        print(f"❌ Error al cargar el índice. ¿Ya ejecutaste la indexación? (Detalle: {e})")
        return

    # Mostrar estadísticas como solicitaste
    print(f"📊 Estadísticas del índice:")
    print(f"   - Documentos indexados: {builder.index.doc_count}")
    print(f"   - Vocabulario: {builder.index.get_vocabulary_size()} términos")
    print(f"   - Longitud promedio: {builder.index.get_average_doc_length():.2f}\n")

    bm25 = BM25(builder.index)

    query = input("📝 Ingrese su consulta: ")
    if not query.strip():
        print("Consulta vacía. Volviendo al menú principal.")
        return
        
    results = bm25.search(query)

    if not results:
        print("❌ No se encontraron resultados para la consulta.")
        return

    print(f"\n🏆 Top {len(results)} resultados:\n")
    for i, result in enumerate(results, 1):
        metadata = builder.get_document_metadata(result.doc_id)
        title = metadata.get("title", "Sin título") if metadata else "Sin título"
        print(f"{i}. [{result.score:.4f}] {title}")
        
        if metadata:
            print(f"   Fuente: {metadata.get('source', 'N/A')}")
            print(f"   URL: {metadata.get('url', 'N/A')}")
        print()



def run_cleanup_duplicates():
    print("\n🧹 Eliminando duplicados...\n")

    cleaner = DataCleaner()

    cleaner.remove_duplicate_crawler()
    cleaner.remove_duplicate_embeddings()

    print("\n✅ Limpieza de duplicados completada.\n")


def run_delete_all_data():
    print("\n⚠️ ADVERTENCIA: Esto eliminará TODOS los datos (documentos y embeddings)\n")
    confirm = input("Escriba 'SI' para confirmar: ")

    if confirm != "SI":
        print("❌ Operación cancelada.")
        return

    cleaner = DataCleaner()

    cleaner.delete_crawler_data()
    cleaner.delete_embeddings()

    print("\n🗑️ Base de datos completamente eliminada.\n")


def run_delete_documents():
    print("\n⚠️ ADVERTENCIA: Esto eliminará SOLO los documentos (datos del crawler)\n")
    confirm = input("Escriba 'SI' para confirmar: ")

    if confirm != "SI":
        print("❌ Operación cancelada.")
        return

    cleaner = DataCleaner()

    cleaner.delete_crawler_data()

    print("\n🗑️ Documentos eliminados.\n")


def run_delete_embeddings():
    print("\n⚠️ ADVERTENCIA: Esto eliminará SOLO los embeddings\n")
    confirm = input("Escriba 'SI' para confirmar: ")

    if confirm != "SI":
        print("❌ Operación cancelada.")
        return

    cleaner = DataCleaner()

    cleaner.delete_embeddings()

    print("\n🗑️ Embeddings eliminados.\n")


def main():
    while True:
        print("\n" + "="*50)
        print("    📰 MOTOR DE BÚSQUEDA Y CRAWLER (BBC)")
        print("="*50)
        print("Seleccione una opción:")
        print("  1 - Crawling")
        print("  2 - Crawling + Indexing")
        print("  3 - Generar Embeddings")
        print("  4 - Buscar (BM25)")
        print("  5 - Buscar (Semantic Search)")
        print("  6 - Buscar (Híbrida - BM25 + Semántica) 🔥") # <--- Nueva opción
        print("  7 - Crawling + Indexing + Embeddings")
        print("  8 - Limpiar duplicados")
        print("  9 - Borrar TODA la base de datos")
        print(" 10 - Borrar SOLO documentos")
        print(" 11 - Borrar SOLO embeddings")
        print(" 12 - Salir")
        print("="*50)

        opcion = input("\n👉 Elija una opción: ")

        # Lógica de las opciones ajustada
        if opcion == "1":
            run_crawling()
        elif opcion == "2":
            run_crawling()
            run_indexing()
        elif opcion == "3":
            run_embeddings()
        elif opcion == "4":
            run_search()
        elif opcion == "5":
            run_semantic_search()
        elif opcion == "6":
            run_hybrid_search()  # <--- Llamada a la nueva función
        elif opcion == "7":
            run_crawling()
            run_indexing()
            run_embeddings()
        elif opcion == "8":
            run_cleanup_duplicates()
        elif opcion == "9":
            run_delete_all_data()
        elif opcion == "10":
            run_delete_documents()
        elif opcion == "11":
            run_delete_embeddings()
        elif opcion == "12":
            print("\n👋 Saliendo del programa. ¡Hasta luego!\n")
            sys.exit(0)
        else:
            print("\n❌ Opción no reconocida.")
            print("Por favor, ingrese un número del 1 al 12.")
            
if __name__ == "__main__":
    main()