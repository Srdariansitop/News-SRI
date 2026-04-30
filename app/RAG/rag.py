import os
import json
import requests
from typing import List, Dict, Any

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
        Invoca a la IA usando OpenRouter.
        Asegúrate de tener la variable de entorno OPENROUTER_API_KEY configurada.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "❌ Error: La variable de entorno OPENROUTER_API_KEY no está configurada."

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:8000",  # Cambia esto por la URL de tu proyecto
                "X-Title": "News-SRI", # Cambia esto por el nombre de tu proyecto
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/llama-3-8b-instruct:free", # Usa el modelo gratuito que prefieras en OpenRouter
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un asistente riguroso para leer documentos. Responde estrictamente la pregunta usando ÚNICAMENTE la información provista en los documentos, no inventes hechos."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"❌ Error conectando a OpenRouter: {str(e)}"

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
