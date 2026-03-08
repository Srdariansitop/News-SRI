import sys

from app.crawler.rss_crawler import crawl
from app.indexing.index_builder import IndexBuilder
from app.retrieval.bm25 import BM25


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


def main():
    while True:
        print("\n" + "="*45)
        print("    📰 MOTOR DE BÚSQUEDA Y CRAWLER (BBC)")
        print("="*45)
        print("Seleccione una opción:")
        print("  1 - Crawling")
        print("  2 - Crawling and Indexing")
        print("  3 - Buscar término")
        print("  4 - Salir")
        print("="*45)

        opcion = input("\n👉 Elija una opción (1-4): ")

        if opcion == "1":
            run_crawling()
            
        elif opcion == "2":
            run_crawling()
            run_indexing()
            
        elif opcion == "3":
            run_search()
            
        elif opcion == "4":
            print("\n👋 Saliendo del programa. ¡Hasta luego!\n")
            sys.exit(0)
            
        else:
            print("\n❌ Opción no reconocida. Por favor, introduzca un número del 1 al 4.")


if __name__ == "__main__":
    main()