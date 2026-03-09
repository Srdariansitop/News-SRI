import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Vector database using FAISS.
    """

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(dimension)

        self.doc_ids = []

    def add(self, vectors: np.ndarray, ids: list[str]):
        """
        Add vectors to the FAISS index.
        """

        vectors = np.array(vectors).astype("float32")

        self.index.add(vectors)

        self.doc_ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 10):

        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for i, idx in enumerate(indices[0]):

            if idx == -1:
                continue

            doc_id = self.doc_ids[idx]

            score = float(distances[0][i])

            results.append((doc_id, score))

        return results

    def save(self, path: str):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "doc_ids.pkl", "wb") as f:
            pickle.dump(self.doc_ids, f)

    def load(self, path: str):

        path = Path(path)

        self.index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "doc_ids.pkl", "rb") as f:
            self.doc_ids = pickle.load(f)