import os
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates normalized embeddings for texts using a SentenceTransformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        local_model_path = os.path.join("modelos", model_name)

        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            print(f"✅ Cargando modelo desde disco: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            print(f"🌐 Modelo local no detectado. Descargando '{model_name}'...")
            self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """
        Generate normalized embedding for a single text.
        """
        vector = self.model.encode(text)

        # 🔥 NORMALIZACIÓN (clave para cosine similarity)
        vector = vector / np.linalg.norm(vector)

        return vector.astype("float32")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate normalized embeddings for multiple texts.
        """
        vectors = self.model.encode(texts)

        # 🔥 NORMALIZACIÓN en batch
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        return np.array(vectors).astype("float32")

    def get_dimension(self) -> int:
        return self.dimension