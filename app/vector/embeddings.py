import os
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """
    Generates embeddings for texts using a SentenceTransformer model.
    Checks for a local version before attempting to download.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Construimos la ruta hacia la carpeta de modelos en la raíz del proyecto
        # Esto asume que el comando se ejecuta desde la carpeta NewsIR
        local_model_path = os.path.join("modelos", model_name)

        # Verificamos si la carpeta local existe y contiene archivos
        if os.path.exists(local_model_path) and os.listdir(local_model_path):
            print(f"✅ Cargando modelo desde disco: {local_model_path}")
            # Al pasar una ruta de carpeta, SentenceTransformer no usa internet
            self.model = SentenceTransformer(local_model_path)
        else:
            print(f"🌐 Modelo local no detectado. Descargando '{model_name}' desde Hugging Face...")
            # Si no existe, se descarga normalmente por nombre
            self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        """
        # FAISS requiere vectores en formato float32
        vector = self.model.encode(text)
        return vector.astype("float32")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        """
        vectors = self.model.encode(texts)
        return np.array(vectors).astype("float32")

    def get_dimension(self) -> int:
        """
        Return embedding dimension.
        """
        return self.dimension