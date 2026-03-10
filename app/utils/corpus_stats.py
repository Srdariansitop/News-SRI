import os
import json
from collections import Counter

DATA_PATH = "data/raw"


def tokenize(text):
    """Tokenización simple por espacios."""
    return text.lower().split()


def compute_corpus_stats():
    documents = 0
    total_words = 0
    total_title_words = 0
    total_chars = 0
    vocabulary = Counter()

    for filename in os.listdir(DATA_PATH):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(DATA_PATH, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)

        documents += 1

        content = doc.get("content", "")
        title = doc.get("title", "")

        words = tokenize(content)
        title_words = tokenize(title)

        total_words += len(words)
        total_title_words += len(title_words)
        total_chars += len(content)

        vocabulary.update(words)

    avg_doc_length = total_words / documents if documents else 0
    avg_title_length = total_title_words / documents if documents else 0
    avg_doc_size = total_chars / documents if documents else 0

    print("\n📊 Estadísticas del Corpus\n")
    print(f"Total de documentos: {documents}")
    print(f"Total de palabras: {total_words}")
    print(f"Vocabulario único: {len(vocabulary)}")
    print(f"Longitud promedio de documentos (palabras): {avg_doc_length:.2f}")
    print(f"Longitud promedio de títulos (palabras): {avg_title_length:.2f}")
    print(f"Tamaño promedio de documentos (caracteres): {avg_doc_size:.2f}")


if __name__ == "__main__":
    compute_corpus_stats()