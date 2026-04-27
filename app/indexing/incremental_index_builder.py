"""
Módulo para manejo incremental de indexación.
Permite reindexar solo documentos nuevos sin procesar todo de nuevo.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone

from app.core.preprocessor import Preprocessor
from app.indexing.inverted_index import InvertedIndex


class IncrementalIndexBuilder:
    """
    Constructor de índices incremental.
    Reindexa solo documentos nuevos en lugar de reconstruir todo de cero.
    """
    
    def __init__(self, data_path: str = "data/raw", index_path: str = "data/index"):
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = Preprocessor()
        self.index = InvertedIndex()
        self.documents_metadata: Dict[str, dict] = {}
        
        # Archivo de tracking de qué documentos fueron indexados
        self.indexed_docs_file = self.index_path / ".indexed_documents.json"
        self.indexed_doc_ids = self._load_indexed_docs_list()
    
    def _load_indexed_docs_list(self) -> set:
        """Carga la lista de documentos ya indexados."""
        if self.indexed_docs_file.exists():
            try:
                with open(self.indexed_docs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get("indexed_ids", []))
            except Exception as e:
                print(f"⚠️ Error cargando lista de documentos indexados: {e}")
                return set()
        return set()
    
    def _save_indexed_docs_list(self, doc_ids: set):
        """Guarda la lista actualizada de documentos indexados."""
        try:
            with open(self.indexed_docs_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "indexed_ids": list(doc_ids),
                    "last_update": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"❌ Error guardando lista de documentos indexados: {e}")
    
    def get_new_documents(self) -> List[str]:
        """
        Obtiene lista de documentos que aún no han sido indexados.
        
        Returns:
            Lista de rutas a archivos JSON nuevos
        """
        if not self.data_path.exists():
            return []
        
        new_docs = []
        for filepath in self.data_path.glob("*.json"):
            # Ignorar archivos de sistema
            if filepath.name.startswith("."):
                continue
            
            # Ignorar archivos de índice
            if filepath.name.startswith("inverted_") or filepath.name.startswith("documents_"):
                continue
            
            doc_id = filepath.stem  # nombre sin extensión
            
            if doc_id not in self.indexed_doc_ids:
                new_docs.append(str(filepath))
        
        return new_docs
    
    def load_document(self, filepath: str) -> Optional[dict]:
        """Carga un documento JSON."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error cargando {filepath}: {e}")
            return None
    
    def extract_indexable_text(self, doc: dict) -> str:
        """Extrae título + contenido para indexación."""
        title = doc.get("title", "")
        content = doc.get("content", "")
        return f"{title} {content}"
    
    def extract_metadata(self, doc: dict) -> dict:
        """Extrae metadatos del documento."""
        return {
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "category": doc.get("category", ""),
            "url": doc.get("url", ""),
            "author": doc.get("author", ""),
            "date": doc.get("date", ""),
        }
    
    def index_new_documents(self, existing_index: Optional[InvertedIndex] = None) -> tuple:
        """
        Indexa solo los documentos nuevos.
        Mantiene el índice existente.
        
        Args:
            existing_index: Índice invertido existente (si existe)
            
        Returns:
            (updated_index, updated_metadata, new_doc_ids_indexed)
        """
        new_docs = self.get_new_documents()
        
        if not new_docs:
            print("✅ No hay documentos nuevos para indexar")
            return existing_index or self.index, self.documents_metadata, []
        
        print(f"\n📑 Indexando {len(new_docs)} documentos nuevos...\n")
        
        # Usar índice existente o crear uno nuevo
        if existing_index:
            self.index = existing_index
        
        new_indexed_ids = []
        
        for i, filepath in enumerate(new_docs, 1):
            doc = self.load_document(filepath)
            if not doc:
                continue
            
            doc_id = doc.get("id")
            if not doc_id:
                print(f"⚠️ Documento sin ID: {filepath}, ignorado")
                continue
            
            # Extraer y procesar texto
            text = self.extract_indexable_text(doc)
            tokens = self.preprocessor.process(text)
            
            # Construir mapa término → posiciones
            term_positions = {}
            for token in tokens:
                term = token.term
                if term not in term_positions:
                    term_positions[term] = []
                term_positions[term].append(token.position)
            
            # Agregar documento al índice
            self.index.add_document(doc_id, term_positions)
            
            # Guardar metadatos
            self.documents_metadata[doc_id] = self.extract_metadata(doc)
            
            new_indexed_ids.append(doc_id)
            
            print(f"   [{i}/{len(new_docs)}] Indexado: {doc_id[:8]}... | {doc.get('title', 'Sin título')[:50]}")
        
        # Actualizar lista de documentos indexados
        self.indexed_doc_ids.update(new_indexed_ids)
        self._save_indexed_docs_list(self.indexed_doc_ids)
        
        print(f"\n✅ {len(new_indexed_ids)} documentos indexados exitosamente")
        
        return self.index, self.documents_metadata, new_indexed_ids
    
    def get_indexed_documents_count(self) -> int:
        """Retorna cantidad de documentos indexados."""
        return len(self.indexed_doc_ids)
    
    def get_indexing_statistics(self) -> Dict:
        """Retorna estadísticas de indexación."""
        return {
            "indexed_documents": len(self.indexed_doc_ids),
            "index_terms": len(self.index.index),
            "total_postings": sum(len(entry.postings) for entry in self.index.index.values()),
            "index_path": str(self.index_path),
            "tracked_doc_ids": len(self.indexed_doc_ids)
        }
