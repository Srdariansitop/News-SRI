"""
WebDocumentManager - Sistema consolidado de normalización, almacenamiento e indexación
de documentos web. Combina en un solo módulo toda la funcionalidad de:
- WebDocumentNormalizer
- PersistentWebStorage
- IncrementalIndexBuilder + Reindexación
"""

import json
import uuid
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class NormalizedDocument:
    """Documento normalizado a formato estándar NewsIR"""
    id: str
    source: str
    category: str
    url: str
    title: str
    author: str
    date: str
    content: str
    crawled_at: str


class WebDocumentManager:
    """
    Sistema consolidado para:
    1. Normalizar documentos web a formato estándar
    2. Almacenarlos persistentemente en data/raw/
    3. Mantener índice de URLs para evitar duplicados
    4. Detectar documentos nuevos para reindexación
    5. Ejecutar indexación incremental
    
    Todo en un solo módulo optimizado.
    """
    
    # Categorías y palabras clave
    CATEGORIES = {
        "Technology": ["ai", "algorithm", "software", "hardware", "code", "python", "javascript", "tech", "computer", "app", "data", "machine learning", "neural"],
        "Science": ["research", "study", "experiment", "climate", "physics", "chemistry", "biology", "arxiv", "quantum", "particle"],
        "Politics": ["government", "election", "parliament", "congress", "senate", "law", "policy", "political", "president", "minister"],
        "Business": ["market", "economy", "stock", "finance", "company", "trade", "business", "startup", "investor", "revenue"],
        "Health": ["health", "medical", "disease", "doctor", "hospital", "virus", "vaccine", "patient", "treatment", "cure"],
        "Sports": ["sport", "game", "team", "player", "match", "football", "basketball", "league", "championship"],
        "Entertainment": ["movie", "music", "actor", "celebrity", "film", "show", "series", "entertainment"],
    }
    
    def __init__(self, raw_data_path: str = "data/raw", index_path: str = "data/index"):
        """
        Inicializar el gestor de documentos web.
        
        Args:
            raw_data_path: Ruta donde almacenar documentos
            index_path: Ruta de índices de búsqueda
        """
        self.raw_data_path = Path(raw_data_path)
        self.index_path = Path(index_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Archivos de control
        self.url_index_path = self.raw_data_path / ".web_url_index.json"
        self.indexed_docs_file = self.index_path / ".indexed_documents.json"
        self.reindex_marker = self.raw_data_path / ".reindex_needed"
        
        # Cargar índices existentes
        self.url_index: Dict[str, str] = self._load_json(self.url_index_path, {})
        self.indexed_doc_ids: Set[str] = set(self._load_json(self.indexed_docs_file, {}).get("indexed_ids", []))
    
    # ==================== NORMALIZACIÓN ====================
    
    def normalize_search_result(self, search_result: Dict[str, Any], source: str = "web") -> NormalizedDocument:
        """Normaliza resultado de búsqueda web a documento estándar."""
        doc_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        title = search_result.get("title", "Sin título").strip()
        url = search_result.get("url", "").strip()
        content = search_result.get("snippet", "").strip()
        
        if "full_content" in search_result and len(content) < 100:
            content = search_result.get("full_content", content)
        
        content = self._truncate_content(content)
        category = self._detect_category(title, content)
        author = search_result.get("author", "Web Source").strip() or "Web Source"
        date_str = search_result.get("date", now) or now
        
        return NormalizedDocument(
            id=doc_id,
            source=source or "web",
            category=category,
            url=url,
            title=title,
            author=author,
            date=date_str,
            content=content,
            crawled_at=now
        )
    
    def _detect_category(self, title: str, content: str = "") -> str:
        """Detecta categoría por palabras clave."""
        combined = f"{title} {content}".lower()
        for category, keywords in self.CATEGORIES.items():
            if any(keyword in combined for keyword in keywords):
                return category
        return "General"
    
    def _truncate_content(self, content: str, max_length: int = 5000) -> str:
        """Trunca contenido a máximo razonable."""
        if not content:
            return ""
        content = content.strip()
        if len(content) > max_length:
            truncated = content[:max_length]
            last_period = truncated.rfind(".")
            if last_period > max_length // 2:
                truncated = truncated[:last_period + 1]
            else:
                truncated = truncated + "..."
            return truncated
        return content
    
    # ==================== ALMACENAMIENTO ====================
    
    def save_web_result(self, search_result: Dict, source: str = "web") -> Optional[str]:
        """Guarda resultado de búsqueda web si no es duplicado."""
        url = search_result.get("url", "").strip()
        
        if not url:
            print("⚠️ Resultado sin URL, ignorado")
            return None
        
        # Verificar duplicado
        if url in self.url_index:
            print(f"⏭️ Documento web ya almacenado (URL duplicada)")
            return self.url_index[url]
        
        # Normalizar
        normalized = self.normalize_search_result(search_result, source)
        doc_dict = self._normalize_to_dict(normalized)
        filepath = self.raw_data_path / f"{normalized.id}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
            
            # Actualizar índice
            self.url_index[url] = normalized.id
            self._save_json(self.url_index_path, self.url_index)
            
            print(f"✅ Documento web guardado: {normalized.id[:8]}...")
            print(f"   Título: {normalized.title[:60]}")
            return normalized.id
        
        except Exception as e:
            print(f"❌ Error guardando documento: {e}")
            return None
    
    def save_multiple_web_results(self, search_results: List[Dict], source: str = "web", auto_index: bool = False) -> Dict[str, str]:
        """Guarda múltiples resultados web."""
        saved_docs = {}
        duplicates_count = 0
        
        print(f"\n📥 Procesando {len(search_results)} resultados web...\n")
        
        for i, result in enumerate(search_results, 1):
            doc_id = self.save_web_result(result, source)
            if doc_id:
                saved_docs[result.get("url")] = doc_id
            else:
                duplicates_count += 1
            
            if i % 5 == 0:
                print(f"   {i}/{len(search_results)} completados")
        
        # Mensaje más claro sobre qué se guardó y qué fueron duplicados
        if saved_docs:
            print(f"\n✅ {len(saved_docs)} documentos NUEVOS guardados")
            if duplicates_count > 0:
                print(f"⏭️ {duplicates_count} documentos eran duplicados (URL ya existente)")
        else:
            print(f"\n⏭️ Ningún documento nuevo (todos {duplicates_count} eran duplicados)")
        
        if auto_index and saved_docs:
            self._mark_reindex_needed()
        
        return saved_docs
    
    # ==================== INDEXACIÓN INCREMENTAL ====================
    
    def get_new_documents(self) -> List[str]:
        """Obtiene documentos no indexados aún."""
        if not self.raw_data_path.exists():
            return []
        
        new_docs = []
        for filepath in self.raw_data_path.glob("*.json"):
            if filepath.name.startswith("."):
                continue
            doc_id = filepath.stem
            if doc_id not in self.indexed_doc_ids:
                new_docs.append(str(filepath))
        
        return new_docs
    
    def reindex_web_documents(self, incremental_builder) -> Tuple[int, Dict[str, Any]]:
        """
        Ejecuta reindexación incremental con documentos web nuevos.
        
        Args:
            incremental_builder: Instancia de IncrementalIndexBuilder
            
        Returns:
            (cantidad_indexados, estadísticas)
        """
        print("\n" + "="*70)
        print("🔄 REINDEXACIÓN CON DOCUMENTOS WEB")
        print("="*70)
        
        new_docs = self.get_new_documents()
        
        if not new_docs:
            print("\n✅ No hay documentos nuevos para indexar")
            stats = self.get_statistics()
            print("\n📊 Estadísticas:")
            print(f"   - Documentos totales: {stats['total_documents']}")
            print(f"   - Documentos indexados: {stats['indexed_documents']}")
            print(f"   - Documentos web: {stats['web_documents']}")
            return 0, stats
        
        print(f"\n📂 Detectados {len(new_docs)} documentos nuevos")
        
        # Usar incremental builder
        updated_index, updated_metadata, newly_indexed = incremental_builder.index_new_documents()
        
        # Actualizar tracking
        self.indexed_doc_ids.update(newly_indexed)
        self._save_indexed_docs_list(self.indexed_doc_ids)
        
        # Limpiar marcador
        self._clear_reindex_marker()
        
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("✅ REINDEXACIÓN COMPLETADA")
        print("="*70)
        print(f"\n📊 Estadísticas finales:")
        print(f"   - Documentos totales: {stats['total_documents']}")
        print(f"   - Documentos indexados: {stats['indexed_documents']}")
        print(f"   - Documentos web: {stats['web_documents']}")
        print(f"   - Documentos indexados en esta sesión: {len(newly_indexed)}")
        
        return len(newly_indexed), stats
    
    # ==================== ESTADÍSTICAS Y CONSULTAS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de almacenamiento."""
        web_count = len(self.url_index)
        all_docs = len(list(self.raw_data_path.glob("*.json")))
        bbc_docs = all_docs - web_count
        
        return {
            "total_documents": all_docs,
            "bbc_documents": bbc_docs,
            "web_documents": web_count,
            "storage_path": str(self.raw_data_path),
            "url_index_size": len(self.url_index),
            "indexed_documents": len(self.indexed_doc_ids)
        }
    
    def get_stored_urls(self) -> Set[str]:
        """Retorna URLs almacenadas."""
        return set(self.url_index.keys())
    
    def document_exists(self, url: str) -> bool:
        """Verifica si URL ya existe."""
        return url in self.url_index
    
    def reindex_needed_exists(self) -> bool:
        """Verifica si hay reindexación pendiente."""
        return self.reindex_marker.exists()
    
    # ==================== UTILIDADES PRIVADAS ====================
    
    def _normalize_to_dict(self, normalized_doc: NormalizedDocument) -> Dict[str, Any]:
        """Convierte documento normalizado a diccionario JSON."""
        return {
            "id": normalized_doc.id,
            "source": normalized_doc.source,
            "category": normalized_doc.category,
            "url": normalized_doc.url,
            "title": normalized_doc.title,
            "author": normalized_doc.author,
            "date": normalized_doc.date,
            "content": normalized_doc.content,
            "crawled_at": normalized_doc.crawled_at
        }
    
    def _load_json(self, filepath: Path, default: Any) -> Any:
        """Carga JSON de forma segura."""
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error cargando {filepath.name}: {e}")
                return default
        return default
    
    def _save_json(self, filepath: Path, data: Any) -> bool:
        """Guarda JSON de forma segura."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error guardando {filepath.name}: {e}")
            return False
    
    def _save_indexed_docs_list(self, doc_ids: Set[str]):
        """Guarda lista de documentos indexados."""
        data = {
            "indexed_ids": list(doc_ids),
            "last_update": datetime.now(timezone.utc).isoformat()
        }
        self._save_json(self.indexed_docs_file, data)
    
    def _mark_reindex_needed(self):
        """Marca que hay reindexación pendiente."""
        try:
            with open(self.reindex_marker, 'w') as f:
                f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
        except Exception as e:
            print(f"⚠️ Error marcando reindexación: {e}")
    
    def _clear_reindex_marker(self):
        """Limpia el marcador de reindexación."""
        try:
            if self.reindex_marker.exists():
                self.reindex_marker.unlink()
        except Exception as e:
            print(f"⚠️ Error limpiando marcador: {e}")
