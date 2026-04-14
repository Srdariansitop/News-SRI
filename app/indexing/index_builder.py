import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from app.core.preprocessor import Preprocessor
from app.indexing.inverted_index import InvertedIndex


DATA_RAW_PATH = "data/raw"
INDEX_PATH = "data/index"
INDEX_FILE = "inverted_index.json"
METADATA_FILE = "documents_metadata.json"


class IndexBuilder:
    def __init__(self, data_path: str = DATA_RAW_PATH, index_path: str = INDEX_PATH):
        self.data_path = data_path
        self.index_path = index_path
        self.preprocessor = Preprocessor()
        self.index_title = InvertedIndex()
        self.index_content = InvertedIndex()
        self.documents_metadata: Dict[str, dict] = {}

    def load_document(self, filepath: str) -> Optional[dict]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def get_document_files(self) -> List[str]:
        if not os.path.exists(self.data_path):
            return []

        files = []
        for filename in os.listdir(self.data_path):
            if filename.endswith(".json"):
                files.append(os.path.join(self.data_path, filename))
        return files

    def extract_indexable_text(self, doc: dict) -> str:
        title = doc.get("title", "")
        content = doc.get("content", "")
        return f"{title} {content}"

    def extract_metadata(self, doc: dict) -> dict:
        return {
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "category": doc.get("category", ""),
            "url": doc.get("url", ""),
            "author": doc.get("author", ""),
            "date": doc.get("date", ""),
        }

    def index_document(self, doc: dict):
        doc_id = doc.get("id")
        if not doc_id:
            return

        title_text = doc.get("title", "")
        content_text = doc.get("content", "")

        title_positions = self.preprocessor.get_term_positions(title_text)
        content_positions = self.preprocessor.get_term_positions(content_text)

        self.index_title.add_document(doc_id, title_positions)
        self.index_content.add_document(doc_id, content_positions)
        self.documents_metadata[doc_id] = self.extract_metadata(doc)

    def build(self) -> int:
        files = self.get_document_files()
        indexed_count = 0

        for filepath in files:
            doc = self.load_document(filepath)
            if doc:
                self.index_document(doc)
                indexed_count += 1

        return indexed_count

    def save(self):
        Path(self.index_path).mkdir(parents=True, exist_ok=True)

        title_filepath = os.path.join(self.index_path, "title_" + INDEX_FILE)
        content_filepath = os.path.join(self.index_path, "content_" + INDEX_FILE)
        
        self.index_title.save(title_filepath)
        self.index_content.save(content_filepath)

        metadata_filepath = os.path.join(self.index_path, METADATA_FILE)
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(self.documents_metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        title_filepath = os.path.join(self.index_path, "title_" + INDEX_FILE)
        content_filepath = os.path.join(self.index_path, "content_" + INDEX_FILE)

        if os.path.exists(title_filepath):
            self.index_title.load(title_filepath)
            
        if os.path.exists(content_filepath):
            self.index_content.load(content_filepath)

        metadata_filepath = os.path.join(self.index_path, METADATA_FILE)
        if os.path.exists(metadata_filepath):
            with open(metadata_filepath, "r", encoding="utf-8") as f:
                self.documents_metadata = json.load(f)

    def get_document_metadata(self, doc_id: str) -> Optional[dict]:
        return self.documents_metadata.get(doc_id)

    def get_stats(self) -> dict:
        return {
            "total_documents": self.index_content.doc_count,
            "vocabulary_size": len(set(self.index_title.get_vocabulary()) | set(self.index_content.get_vocabulary())),
            "avg_title_length": self.index_title.get_average_doc_length(),
            "avg_content_length": self.index_content.get_average_doc_length(),
        }


def build_index():
    print("🔨 Building inverted index...")

    builder = IndexBuilder()
    count = builder.build()

    print(f"📄 Indexed {count} documents")
    print(f"📚 Vocabulary size: {builder.index.get_vocabulary_size()} terms")
    print(f"📏 Average doc length: {builder.index.get_average_doc_length():.2f} terms")

    builder.save()
    print(f"💾 Index saved to {builder.index_path}/")

    return builder


if __name__ == "__main__":
    build_index()
