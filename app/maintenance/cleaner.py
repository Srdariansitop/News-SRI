import os
import json
import pickle
import shutil
from pathlib import Path

import faiss
import numpy as np


class DataCleaner:
    def __init__(self):
        self.base_path = Path("data")

        self.vector_path = self.base_path / "vector_store"
        self.raw_path = self.base_path / "raw"
        self.index_path = self.base_path / "index"

    # -------------------------
    # 🔴 DELETE EMBEDDINGS
    # -------------------------
    def delete_embeddings(self):
        if not self.vector_path.exists():
            print("⚠️ No existe la base de embeddings")
            return

        total_files = sum(1 for _ in self.vector_path.rglob("*") if _.is_file())

        shutil.rmtree(self.vector_path)

        print(f"🗑️ Embeddings eliminados ({total_files} archivos borrados)")

    # -------------------------
    # 🔴 DELETE CRAWLER DATA
    # -------------------------
    def delete_crawler_data(self):
        files_deleted = 0

        if self.raw_path.exists():
            files = list(self.raw_path.glob("*.json"))
            files_deleted += len(files)
            shutil.rmtree(self.raw_path)

        if self.index_path.exists():
            files = list(self.index_path.rglob("*"))
            files_deleted += len([f for f in files if f.is_file()])
            shutil.rmtree(self.index_path)

        print(f"🗑️ Crawler eliminado ({files_deleted} archivos borrados)")

    # -------------------------
    # 🟡 REMOVE DUPLICATE EMBEDDINGS
    # -------------------------
    def remove_duplicate_embeddings(self):
        index_file = self.vector_path / "faiss.index"
        ids_file = self.vector_path / "doc_ids.pkl"

        if not index_file.exists() or not ids_file.exists():
            print("⚠️ No hay embeddings para limpiar")
            return

        index = faiss.read_index(str(index_file))

        with open(ids_file, "rb") as f:
            doc_ids = pickle.load(f)

        seen = set()
        unique_vectors = []
        unique_ids = []
        duplicates = 0

        for i, doc_id in enumerate(doc_ids):
            if doc_id not in seen:
                seen.add(doc_id)
                unique_ids.append(doc_id)
                unique_vectors.append(index.reconstruct(i))
            else:
                duplicates += 1

        unique_vectors = np.array(unique_vectors).astype("float32")

        new_index = faiss.IndexFlatL2(index.d)
        new_index.add(unique_vectors)

        faiss.write_index(new_index, str(index_file))

        with open(ids_file, "wb") as f:
            pickle.dump(unique_ids, f)

        print(f"🧹 Embeddings duplicados eliminados: {duplicates}")

    # -------------------------
    # 🟡 REMOVE DUPLICATE DOCUMENTS
    # -------------------------
    def remove_duplicate_crawler(self):
        if not self.raw_path.exists():
            print("⚠️ No hay documentos para limpiar")
            return

        seen_urls = set()
        duplicates = 0

        files = list(self.raw_path.glob("*.json"))

        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                url = doc.get("url")

                if url in seen_urls:
                    file.unlink()
                    duplicates += 1
                else:
                    seen_urls.add(url)

            except Exception as e:
                print(f"❌ Error procesando {file}: {e}")

        print(f"🧹 Documentos duplicados eliminados: {duplicates}")