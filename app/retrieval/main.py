from app.indexing.index_builder import IndexBuilder
from app.retrieval.bm25 import BM25


def main():
    builder = IndexBuilder()
    builder.load()

    print(f"Documentos indexados: {builder.index.doc_count}")
    print(f"Vocabulario: {builder.index.get_vocabulary_size()} términos")
    print(f"Longitud promedio: {builder.index.get_average_doc_length():.2f}\n")

    bm25 = BM25(builder.index)

    query = input("Ingrese su consulta: ")
    results = bm25.search(query, top_k=5)

    if not results:
        print("No se encontraron resultados.")
        return

    print(f"\nTop {len(results)} resultados:\n")
    for i, result in enumerate(results, 1):
        metadata = builder.get_document_metadata(result.doc_id)
        title = metadata.get("title", "Sin título") if metadata else "Sin título"
        print(f"{i}. [{result.score:.4f}] {title}")
        if metadata:
            print(f"   Fuente: {metadata.get('source', 'N/A')}")
            print(f"   URL: {metadata.get('url', 'N/A')}")
        print()


if __name__ == "__main__":
    main()