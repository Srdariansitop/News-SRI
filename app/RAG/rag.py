import os
import json
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class RAGSystem:
    def __init__(self, hybrid_searcher, raw_data_path: str = "data/raw"):
        """
        Inicializa el sistema RAG.
        :param hybrid_searcher: Instancia de HybridSearcher (app/retrieval/hybrid.py)
        :param raw_data_path: Ruta a la carpeta que contiene los documentos descargados en crudo.
        """
        self.hybrid_searcher = hybrid_searcher
        self.raw_data_path = raw_data_path

    def _get_raw_document(self, doc_id: str) -> str:
        """
        Recupera el contenido del documento crudo dado su ID.
        NOTA SOBRE LA IMPLEMENTACIÓN: Esta es una abstracción. Ahora mismo asume 
        que los documentos están en local (ej. JSON) dentro de `data/raw/`, pero 
        en el futuro podría conectarse a AWS S3, un contenedor de base de datos 
        externa o cualquier API de almacenamiento sin romper la lógica.
        """
        # Asume formato .json, cambia si es txt o se almacena en BD
        file_path = os.path.join(self.raw_data_path, f"{doc_id}.json")
        
        if not os.path.exists(file_path):
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("content", "")
        except Exception as e:
            print(f"Error cargando documento crudo {doc_id}: {e}")
            return ""

    def _call_llm(self, prompt: str) -> str:
        """
        Invoca a la IA usando Groq.
        Asegúrate de tener la variable de entorno GROQ_API_KEY configurada.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "❌ Error: La variable de entorno GROQ_API_KEY no está configurada."

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.3-70b-versatile", # Modelo más capaz y actualizado de Llama en Groq
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un asistente de lectura de documentos. Responde a la pregunta del usuario usando exclusivamente el contexto proporcionado. Si la respuesta no está en el contexto, di 'No tengo información sobre eso'."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"❌ Error conectando a Groq: {str(e)}"

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Responde a la consulta usando el enfoque Retrieval-Augmented Generation.
        Retorna el resumen de la IA y los 10 documentos recuperados.
        """
        # 1. Recuperar los 10 documentos (links) usando la búsqueda híbrida
        search_results = self.hybrid_searcher.search(query, top_k=10)
        
        # 2. Tomar el contenido real de los 3 mejores resultados
        top_3_results = search_results[:3]
        context_texts = []
        
        for i, result in enumerate(top_3_results, 1):
            doc_id = result.get("doc_id")
            content = self._get_raw_document(doc_id)
            
            if content:
                context_texts.append(f"--- Documento {i} ---\n{content}\n")
            else:
                # Fallback si el documento local no existe o se limpia rápido (solo metadatos)
                context_texts.append(f"--- Documento {i} ---\nTítulo: {result.get('title')}\nURL: {result.get('url')}\n")

        context_str = "\n".join(context_texts)

        # 3. Construir el prompt manual por defecto exigido
        prompt = (
            f"El usuario me pregunta esto: '{query}'\n\n"
            f"Estos son mis documentos:\n{context_str}\n\n"
            f"Dame una respuesta resumen dado mis documentos, no te inventes nada."
        )

        # 4. Obtener la respuesta de la Inteligencia Artificial (resumen)
        summary = self._call_llm(prompt)

        # 5. Entregar la respuesta resumen en conjunto con los 10 documentos
        return {
            "summary": summary,
            "documents": search_results
        }
