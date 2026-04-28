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
from app.maintenance.web_document_manager import WebDocumentManager
from app.indexing.incremental_index_builder import IncrementalIndexBuilder

RAW_DATA_PATH = "data/raw"


def load_raw_documents():

    documents = []

    for filename in os.listdir(RAW_DATA_PATH):

        # Filtrar archivos especiales del sistema (comienzan con .)
        if filename.startswith("."):
            continue
            
        if not filename.endswith(".json"):
            continue

        path = os.path.join(RAW_DATA_PATH, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Verificar que el documento tiene estructura válida (tiene "id")
            # y que es un diccionario (no una lista u otro tipo)
            if isinstance(data, dict) and "id" in data:
                documents.append(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Ignorar archivos corruptos o con estructura inválida
            continue

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

    # Convertir IDs a diccionarios de metadata (requerido por vector_store.add)
    metadata_list = [{"id": doc_id} for doc_id in ids]
    vector_store.add(vectors, metadata_list)

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
    initial_doc_count = len(doc_map)

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

    # 4. Inicializar buscador híbrido CON ALMACENAMIENTO WEB
    hybrid_searcher = HybridSearcher(
        bm25=bm25, 
        vector_store=vector_store, 
        embedder=embedder, 
        doc_metadata_map=doc_map,
        enable_web_search=True,
        save_web_results=True  # 👈 Guardar resultados web
    )

    # 5. Ejecutar consulta
    query = input("📝 Ingrese su consulta: ")
    if not query.strip():
        print("Consulta vacía. Volviendo al menú.")
        return

    results = hybrid_searcher.search(query, top_k=10, semantic_threshold=0.7)

    if not results:
        print("❌ No se encontraron resultados.")
        return

    # 6. VERIFICAR si se guardaron nuevos documentos web y reindexar automáticamente
    documents_after = load_raw_documents()
    new_doc_count = len(documents_after)
    
    if new_doc_count > initial_doc_count:
        new_docs_count = new_doc_count - initial_doc_count
        print(f"\n🔄 Se guardaron {new_docs_count} documentos web nuevos.")
        print("🔁 Reindexando automáticamente para futuras búsquedas...\n")
        
        try:
            from app.maintenance.web_document_manager import WebDocumentManager
            from app.indexing.incremental_index_builder import IncrementalIndexBuilder
            
            web_manager = WebDocumentManager()
            incremental_builder = IncrementalIndexBuilder()
            newly_indexed, stats = web_manager.reindex_web_documents(incremental_builder)
            
            print(f"\n✅ {newly_indexed} documentos indexados correctamente")
            print("🔄 Recargando índices para mostrar nuevos resultados...\n")
            
            # 🔑 RECARGAMOS LOS ÍNDICES DESDE DISCO
            try:
                builder.load()  # Recarga BM25 con los datos nuevos
                bm25 = BM25(builder.index)
                vector_store.load("data/vector_db")  # Recarga FAISS con los datos nuevos
                
                # ✅ Cargar documentos frescos y convertir a dict
                documents_refreshed = load_raw_documents()
                doc_map_refreshed = {doc["id"]: doc for doc in documents_refreshed}
                
                # Recreamos el hybrid_searcher con los índices frescos
                hybrid_searcher = HybridSearcher(
                    bm25=bm25, 
                    vector_store=vector_store, 
                    embedder=embedder, 
                    doc_metadata_map=doc_map_refreshed,  # ✓ Pasar dict, no lista
                    enable_web_search=True,
                    save_web_results=True
                )
                
                # 🔍 EJECUTAMOS LA BÚSQUEDA DE NUEVO CON LOS ÍNDICES NUEVOS
                results = hybrid_searcher.search(query, top_k=10, semantic_threshold=0.7)
                print(f"✅ Nueva búsqueda ejecutada con documentos indexados\n")
                
            except Exception as e:
                print(f"⚠️ Advertencia al recargar índices: {e}")
                print("💡 Los resultados mostrados son sin los nuevos documentos\n")
                
        except Exception as e:
            print(f"⚠️ Advertencia al reindexar: {e}")
            print("💡 Ejecuta la opción 9 del menú para reindexar manualmente\n")

    print(f"\n🏆 Top {len(results)} Resultados Híbridos:\n")
    for i, result in enumerate(results, 1):
        # Determina si es LOCAL o WEB
        is_web = result.get("from_web", False)
        source_label = "🌐 [WEB]" if is_web else "📰 [LOCAL]"
        
        print(f"{i}. {source_label} [{result['score']:.4f} RRF] {result['title']}")
        print(f"   Fuente: {result['source']}")
        print(f"   URL: {result['url']}\n")
    
    # 7. Mostrar estadísticas de almacenamiento web
    print("\n" + "="*50)
    print("📊 Estadísticas de Almacenamiento Web:")
    stats = hybrid_searcher.get_web_storage_stats()
    if "status" not in stats:
        print(f"   - Documentos totales: {stats.get('total_documents', 0)}")
        print(f"   - Documentos BBC: {stats.get('bbc_documents', 0)}")
        print(f"   - Documentos web: {stats.get('web_documents', 0)}")
        print(f"   - Ruta: {stats.get('storage_path', 'N/A')}")
    else:
        print(f"   {stats['status']}")
    print("="*50)

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


def run_web_storage_stats():
    """Muestra estadísticas del almacenamiento web."""
    print("\n📊 ESTADÍSTICAS DE ALMACENAMIENTO WEB\n")
    print("="*60)
    
    try:
        web_manager = WebDocumentManager()
        stats = web_manager.get_statistics()
        
        print(f"📁 Ruta de almacenamiento: {stats['storage_path']}")
        print(f"\n📚 Documentos:")
        print(f"   - Total:          {stats['total_documents']}")
        print(f"   - Documentos BBC: {stats['bbc_documents']}")
        print(f"   - Documentos web: {stats['web_documents']}")
        
        urls_stored = web_manager.get_stored_urls()
        print(f"\n🔗 URLs únicas almacenadas: {len(urls_stored)}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")


def run_reindex_web_documents():
    """Reindexación incremental incorporando documentos web."""
    print("\n🔄 REINDEXACIÓN CON DOCUMENTOS WEB\n")
    
    try:
        web_manager = WebDocumentManager()
        incremental_builder = IncrementalIndexBuilder()
        
        # Ejecutar reindexación
        newly_indexed, stats = web_manager.reindex_web_documents(incremental_builder)
        
        if newly_indexed > 0:
            print(f"\n✅ Reindexación completada exitosamente")
            print(f"   Documentos nuevos indexados: {newly_indexed}")
        
    except Exception as e:
        print(f"❌ Error durante la reindexación: {e}")


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
        print("\n" + "="*60)
        print("    📰 MOTOR DE BÚSQUEDA Y CRAWLER (BBC + Web)")
        print("="*60)
        print("Seleccione una opción:")
        print("\n🔍 BÚSQUEDA Y RECUPERACIÓN:")
        print("  1 - Crawling (descargar artículos BBC)")
        print("  2 - Crawling + Indexing")
        print("  3 - Generar Embeddings (búsqueda semántica)")
        print("  4 - Buscar (BM25)")
        print("  5 - Buscar (Semantic Search)")
        print("  6 - Buscar (Híbrida - BM25 + Semántica) 🔥")
        print("  7 - Crawling + Indexing + Embeddings (todo)")
        
        print("\n🌐 GESTIÓN DE DOCUMENTOS WEB:")
        print("  8 - Ver estadísticas de almacenamiento web")
        print("  9 - Reindexar con documentos web nuevos")
        
        print("\n🧹 MANTENIMIENTO:")
        print(" 10 - Limpiar duplicados")
        print(" 11 - Borrar TODA la base de datos")
        print(" 12 - Borrar SOLO documentos")
        print(" 13 - Borrar SOLO embeddings")
        
        print("\n❌ SALIR:")
        print(" 14 - Salir del programa")
        print("="*60)

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
            run_hybrid_search()
        elif opcion == "7":
            run_crawling()
            run_indexing()
            run_embeddings()
        elif opcion == "8":
            run_web_storage_stats()
        elif opcion == "9":
            run_reindex_web_documents()
        elif opcion == "10":
            run_cleanup_duplicates()
        elif opcion == "11":
            run_delete_all_data()
        elif opcion == "12":
            run_delete_documents()
        elif opcion == "13":
            run_delete_embeddings()
        elif opcion == "14":
            print("\n👋 Saliendo del programa. ¡Hasta luego!\n")
            sys.exit(0)
        else:
            print("\n❌ Opción no reconocida.")
            print("Por favor, ingrese un número del 1 al 14.")
            
if __name__ == "__main__":
    main()