"""
Módulo de Retroalimentación - Calcula score de popularidad
Gestiona las métricas de interacción del usuario (clicks, búsquedas, shares)
y normaliza a escala 0-1 para usar en ranking
"""

from typing import Dict, Any
import json
from pathlib import Path


class PopularityCalculator:
    """
    Calcula el score de popularidad de un documento.
    Lee popularity_metrics.json y normaliza a 0-1.
    
    FÓRMULA:
    popularity_score = (clicks_norm * 0.50) + (search_freq_norm * 0.30) + (shares_norm * 0.20)
    
    Donde cada valor normalizado = valor_actual / valor_máximo_en_archivo
    """
    
    def __init__(self, metrics_path: str = "data/ranking/popularity_metrics.json"):
        """
        Inicializa el calculador de popularidad.
        
        Args:
            metrics_path: Ruta al archivo de métricas de popularidad
        """
        self.metrics_path = Path(metrics_path)
        self.metrics = self._load_metrics()
        self._cache_max_values = None
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Carga popularity_metrics.json, excluye metadatos"""
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Excluir metadatos (empiezan con _), devolver solo documentos
                    return {k: v for k, v in data.items() if not k.startswith('_')}
            except Exception as e:
                print(f"⚠️ Error cargando métricas: {e}")
                return {}
        return {}
    
    def _get_max_values(self) -> Dict[str, float]:
        """
        Calcula los valores máximos de cada métrica en el archivo.
        Se usa para normalización (evitar división por cero).
        """
        if self._cache_max_values:
            return self._cache_max_values
        
        max_clicks = 1.0
        max_search_frequency = 1.0
        max_shares = 1.0
        
        # Recorrer todos los documentos para encontrar máximos
        for doc_id, metrics in self.metrics.items():
            max_clicks = max(max_clicks, metrics.get("clicks", 0))
            max_search_frequency = max(max_search_frequency, metrics.get("search_frequency", 0))
            max_shares = max(max_shares, metrics.get("shares", 0))
        
        self._cache_max_values = {
            "clicks": max_clicks,
            "search_frequency": max_search_frequency,
            "shares": max_shares
        }
        
        return self._cache_max_values
    
    def calculate(self, doc_id: str) -> float:
        """
        Calcula el score de popularidad para un documento (0-1).
        
        PROCESO:
        1. Lee clicks, search_frequency, shares del doc_id
        2. Normaliza cada uno dividiendo por el máximo en el archivo
        3. Combina ponderadamente:
           - clicks: 50%
           - search_frequency: 30%
           - shares: 20%
        
        EJEMPLO:
        Si doc_id tiene: clicks=100, searches=50, shares=10
        Y los máximos en archivo son: max_clicks=100, max_searches=50, max_shares=10
        
        Entonces:
        popularity = (100/100 * 0.50) + (50/50 * 0.30) + (10/10 * 0.20)
                   = (1.0 * 0.50) + (1.0 * 0.30) + (1.0 * 0.20)
                   = 0.50 + 0.30 + 0.20 = 1.0 (Máxima popularidad)
        
        Args:
            doc_id: ID del documento
        
        Returns:
            Score de popularidad entre 0 y 1
        """
        
        # Si el documento no existe en métricas, retorna 0
        if doc_id not in self.metrics:
            return 0.0
        
        doc_metrics = self.metrics[doc_id]
        max_values = self._get_max_values()
        
        # Extraer métricas (con defaults en caso de que falten)
        clicks = doc_metrics.get("clicks", 0)
        search_frequency = doc_metrics.get("search_frequency", 0)
        shares = doc_metrics.get("shares", 0)
        
        # Normalizar cada métrica a 0-1 (valor / máximo)
        clicks_normalized = clicks / max_values["clicks"] if max_values["clicks"] > 0 else 0.0
        search_normalized = search_frequency / max_values["search_frequency"] if max_values["search_frequency"] > 0 else 0.0
        shares_normalized = shares / max_values["shares"] if max_values["shares"] > 0 else 0.0
        
        # Combinar con pesos: clicks (50%) > búsquedas (30%) > shares (20%)
        popularity_score = (
            clicks_normalized * 0.50 +
            search_normalized * 0.30 +
            shares_normalized * 0.20
        )
        
        # Asegurar que está en rango [0, 1]
        return min(1.0, max(0.0, popularity_score))
    
    def get_document_metrics(self, doc_id: str) -> Dict[str, Any]:
        """
        Retorna las métricas raw de un documento (sin normalizar).
        
        Args:
            doc_id: ID del documento
        
        Returns:
            Dict con clicks, search_frequency, shares, last_updated
        """
        return self.metrics.get(doc_id, {
            "clicks": 0,
            "search_frequency": 0,
            "shares": 0,
            "last_updated": None
        })
    
    def add_click(self, doc_id: str):
        """
        Suma +1 a clicks del documento en popularity_metrics.json.
        Se llama cuando usuario CLICKEA un resultado.
        
        Args:
            doc_id: ID del documento
        
        EJEMPLO (desde API/Frontend):
            pop_calc = PopularityCalculator()
            pop_calc.add_click(doc_id="1413f986-6342...")
            → popularity_metrics.json[doc_id].clicks += 1
        """
        self.metrics = self._load_metrics()
        
        if doc_id not in self.metrics:
            self.metrics[doc_id] = {
                "clicks": 0,
                "search_frequency": 0,
                "shares": 0,
                "last_updated": None
            }
        
        self.metrics[doc_id]["clicks"] += 1
        self._save_metrics()
        self._cache_max_values = None
    
    def add_share(self, doc_id: str):
        """
        Suma +1 a shares del documento en popularity_metrics.json.
        Se llama cuando usuario COMPARTE un resultado.
        
        Args:
            doc_id: ID del documento
        
        EJEMPLO (desde API/Frontend):
            pop_calc = PopularityCalculator()
            pop_calc.add_share(doc_id="1413f986-6342...")
            → popularity_metrics.json[doc_id].shares += 1
        """
        self.metrics = self._load_metrics()
        
        if doc_id not in self.metrics:
            self.metrics[doc_id] = {
                "clicks": 0,
                "search_frequency": 0,
                "shares": 0,
                "last_updated": None
            }
        
        self.metrics[doc_id]["shares"] += 1
        self._save_metrics()
        self._cache_max_values = None
    
    def add_search(self, doc_id: str):
        """
        Suma +1 a search_frequency del documento en popularity_metrics.json.
        Se llama cuando un documento APARECE en una búsqueda.
        
        Args:
            doc_id: ID del documento
        
        EJEMPLO (desde API/Frontend):
            pop_calc = PopularityCalculator()
            pop_calc.add_search(doc_id="1413f986-6342...")
            → popularity_metrics.json[doc_id].search_frequency += 1
        """
        self.metrics = self._load_metrics()
        
        if doc_id not in self.metrics:
            self.metrics[doc_id] = {
                "clicks": 0,
                "search_frequency": 0,
                "shares": 0,
                "last_updated": None
            }
        
        self.metrics[doc_id]["search_frequency"] += 1
        self._save_metrics()
        self._cache_max_values = None
    
    def update_metrics(self, doc_id: str, clicks: int = 0, searches: int = 0, shares: int = 0):
        """
        Actualiza las métricas de un documento en popularity_metrics.json.
        
        DEPRECATED: Usar en su lugar add_click(), add_share(), add_search()
        
        Se llamaría desde el FRONTEND cuando el usuario interactúa con un resultado.
        
        Args:
            doc_id: ID del documento
            clicks: Incrementar clicks en X (ej: +1)
            searches: Incrementar search_frequency en X (ej: +1)
            shares: Incrementar shares en X (ej: +1)
        
        EJEMPLO DE USO (desde frontend):
        cuando usuario clickea resultado #2 en búsqueda de "covid":
            pop_calc = PopularityCalculator()
            pop_calc.update_metrics(doc_id="doc_123", clicks=1, searches=1)
        """
        self.metrics = self._load_metrics()  # Recargar para evitar overwrites concurrentes
        
        if doc_id not in self.metrics:
            # Crear entrada si no existe
            self.metrics[doc_id] = {
                "clicks": 0,
                "search_frequency": 0,
                "shares": 0,
                "last_updated": None
            }
        
        # Actualizar valores
        self.metrics[doc_id]["clicks"] += clicks
        self.metrics[doc_id]["search_frequency"] += searches
        self.metrics[doc_id]["shares"] += shares
        
        # Guardar en archivo
        self._save_metrics()
        
        # Invalidar cache de máximos (próxima llamada recalcula)
        self._cache_max_values = None
    
    def _save_metrics(self):
        """Guarda los cambios en popularity_metrics.json preservando metadatos"""
        try:
            # Leer el archivo completo para preservar _metadata y _example
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            
            # Actualizar solo los documentos (no los metadatos)
            for doc_id, metrics in self.metrics.items():
                full_data[doc_id] = metrics
            
            # Guardar
            with open(self.metrics_path, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")
