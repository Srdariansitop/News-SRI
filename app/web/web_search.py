"""
Módulo de búsqueda web para ampliar resultados cuando la base de datos es insuficiente.
Busca información en la web y retorna resultados procesados.
"""

from ddgs import DDGS
from typing import List, Dict, Tuple
import time


class WebSearcher:
    """
    Realiza búsquedas en la web cuando la base de datos local no tiene suficientes resultados.
    Usa duckduckgo-search para evitar bloqueos de protección anti-bots.
    """
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Busca información en la web usando DuckDuckGo con manejo automático de rotación.
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de diccionarios con resultados de búsqueda
        """
        try:
            results = []
            
            # Usa la librería duckduckgo-search que maneja rotación automática
            search_results = self.ddgs.text(query, max_results=top_k)
            
            for result in search_results:
                try:
                    results.append({
                        "title": result.get("title", "Sin título"),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": "web"
                    })
                except Exception as e:
                    print(f"⚠️ Error procesando resultado: {e}")
                    continue
            
            return results
        
        except Exception as e:
            print(f"❌ Error en búsqueda web: {e}")
            return []
    
    def scrape_content(self, url: str) -> str:
        """
        Extrae el contenido de texto de una URL.
        
        Args:
            url: URL a scrapear
            
        Returns:
            Texto extraído de la página
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Elimina scripts y estilos
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Obtiene el texto
            text = soup.get_text(separator="\n", strip=True)
            
            # Limpia espacios en blanco excesivos
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return "\n".join(lines[:500])  # Limita a 500 líneas
        
        except Exception as e:
            print(f"❌ Error scrapando contenido: {e}")
            return ""


class SufficiencyChecker:
    """
    Determina si los resultados locales son suficientes o si es necesario buscar en la web.
    """
    
    def __init__(self, min_results: int = 3, min_avg_score: float = 0.3, min_semantic_score: float = 0.5):
        """
        Args:
            min_results: Mínimo número de resultados para considerar suficiente
            min_avg_score: Score promedio mínimo requerido
            min_semantic_score: Score semántico MÍNIMO individual requerido (umbral de relevancia)
        """
        self.min_results = min_results
        self.min_avg_score = min_avg_score
        self.min_semantic_score = min_semantic_score
    
    def is_sufficient(self, results: List[Dict]) -> Tuple[bool, str]:
        """
        Evalúa si los resultados locales son suficientes.
        
        Lógica:
        - Si NO hay resultados, necesita web
        - Si TODOS tienen semantic_score=0.0, necesita web (sin embeddings)
        - Si hay resultados con semantic_score > 0.0, son suficientes
        
        Args:
            results: Lista de resultados de búsqueda local
            
        Returns:
            Tupla (es_suficiente, razón)
        """
        
        # Si no hay resultados, necesitamos búsqueda web
        if not results:
            return False, "No hay resultados locales"
        
        # Contar resultados locales (no WEB)
        local_results = [r for r in results if not r.get("from_web", False)]
        
        # Si NO hay resultados locales en absoluto, necesita web
        if not local_results:
            return False, "No hay resultados locales"
        
        # CLAVE: Si TODOS tienen semantic_score=0.0, NO hay embeddings generados
        # No podemos confiar en BM25 solo para búsquedas muy específicas
        max_semantic_score = max((r.get("semantic_score", 0.0) for r in local_results), default=0.0)
        
        if max_semantic_score < self.min_semantic_score:
            # Score semántico demasiado bajo - no es relevante semánticamente
            return False, f"Score semántico muy bajo ({max_semantic_score:.3f} < {self.min_semantic_score}). Los resultados no son relevantes para esta búsqueda."
        
        # Si tenemos al menos semantic_score > 0, consideramos suficiente
        if len(local_results) >= self.min_results:
            return True, f"✅ {len(local_results)} resultados locales encontrados"
        
        return False, f"Solo {len(local_results)} resultados locales (mínimo: {self.min_results})"
