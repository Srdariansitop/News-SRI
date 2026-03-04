import sys

from app.crawler.rss_crawler import crawl
from app.indexing.index_builder import IndexBuilder


def run_crawling():
    print("\n🚀 Iniciando crawling...\n")
    crawl()
    print("\n✅ Crawling finalizado.\n")


def run_indexing():
    print("\n📚 Iniciando construcción del índice...\n")

    builder = IndexBuilder()
    count = builder.build()

    print(f"📄 Indexed {count} documents")
    builder.save()

    print("\n✅ Indexación finalizada.\n")

def run_full_pipeline():
    run_crawling()
    run_indexing()


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python -m app.main crawl")
        print("  python -m app.main index")
        print("  python -m app.main full")
        return

    command = sys.argv[1]

    if command == "crawl":
        run_crawling()

    elif command == "index":
        run_indexing()

    elif command == "full":
        run_full_pipeline()

    else:
        print("❌ Comando no reconocido.")


if __name__ == "__main__":
    main()