import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Vector database using FAISS with:
    - Cosine similarity (Inner Product)
    - HNSW index (efficient ANN)
    - Metadata storage
    """

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexHNSWFlat(dimension, 32)

        self.index.hnsw.efSearch = 50

        self.metadata = []

    def add(self, vectors: np.ndarray, metadata_list: list[dict]):
        """
        Add vectors with metadata to the index.
        """

        vectors = np.array(vectors).astype("float32")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        self.index.add(vectors)

        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 10):
        """
        Search similar vectors using cosine similarity.
        """

        query_vector = np.array([query_vector]).astype("float32")

        query_vector = query_vector / np.linalg.norm(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for i, idx in enumerate(indices[0]):

            if idx == -1:
                continue

            doc_metadata = self.metadata[idx]

            score = float(scores[0][i])

            results.append({
                "metadata": doc_metadata,
                "score": score
            })

        return results

    def save(self, path: str):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):

        path = Path(path)

        self.index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)