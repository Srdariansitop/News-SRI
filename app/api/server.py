from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

from app.indexing.index_builder import IndexBuilder
from app.retrieval.bm25 import BM25
from app.vector.embeddings import EmbeddingGenerator
from app.vector.vector_store import VectorStore
from app.retrieval.hybrid import HybridSearcher
from app.RAG.rag import RAGSystem
from app.feedback.popularity_calculator import PopularityCalculator
from app.main import load_raw_documents

app = FastAPI(title="News-SRI API")


class SearchRequest(BaseModel):
    query: str
    enable_web_search: Optional[bool] = True


class ClickShareRequest(BaseModel):
    doc_id: str


def ensure_popularity_metrics():
    """Ensure data/ranking/popularity_metrics.json exists with a minimal template."""
    ranking_path = os.path.join("data", "ranking")
    metrics_file = os.path.join(ranking_path, "popularity_metrics.json")

    if not os.path.exists(ranking_path):
        os.makedirs(ranking_path, exist_ok=True)

    if not os.path.exists(metrics_file):
        default_metrics = {
            "_metadata": {
                "description": "Almacena métricas de popularidad de documentos",
                "created_at": "",
            }
        }
        try:
            with open(metrics_file, "w", encoding="utf-8") as f:
                import json
                from datetime import datetime

                default_metrics["_metadata"]["created_at"] = (
                    datetime.utcnow().isoformat()
                )
                json.dump(default_metrics, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


@app.post("/search")
def search(req: SearchRequest):
    """Execute the RAG complete flow (hybrid search + web + IA) and return results."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # 1. Load raw documents map
    documents = load_raw_documents()
    doc_map = {doc["id"]: doc for doc in documents}

    # 2. Load BM25 index
    builder = IndexBuilder()
    try:
        builder.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading index: {e}")

    # 3. Load embedder and vector store
    try:
        embedder = EmbeddingGenerator()
        vector_store = VectorStore(embedder.get_dimension())
        vector_store.load("data/vector_db")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading embeddings/vector store: {e}"
        )

    # 4. Build hybrid searcher
    try:
        from app.retrieval.bm25 import BM25 as BM25Class

        bm25 = BM25Class(builder.index)

        hybrid_searcher = HybridSearcher(
            bm25=bm25,
            vector_store=vector_store,
            embedder=embedder,
            doc_metadata_map=doc_map,
            enable_web_search=req.enable_web_search,
            save_web_results=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error initializing hybrid searcher: {e}"
        )

    # 5. Run RAG
    rag = RAGSystem(hybrid_searcher=hybrid_searcher, raw_data_path="data/raw")

    try:
        result = rag.answer(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing RAG: {e}")

    # 6. Ensure popularity file and update search_frequency for returned documents
    ensure_popularity_metrics()
    pop = PopularityCalculator()
    # Update search_frequency for each returned doc if it looks like a real id
    for doc in result.get("documents", []):
        doc_id = doc.get("doc_id")
        if doc_id and not str(doc_id).startswith("web_"):
            try:
                pop.add_search(doc_id)
            except Exception:
                # ignore update failures, but continue
                pass

    # 7. Build a clean response with required fields (10 results)
    documents_out = []
    for doc in result.get("documents", [])[:10]:
        documents_out.append(
            {
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "source": doc.get("source"),
                "snippet": doc.get("snippet", ""),
                "date": doc.get(
                    "date", doc_map.get(doc.get("doc_id", ""), {}).get("date", "")
                ),
                "url": doc.get(
                    "url", doc_map.get(doc.get("doc_id", ""), {}).get("url", "")
                ),
                "score": doc.get("score", 0.0),
            }
        )

    return {
        "summary": result.get("summary"),
        "documents": documents_out,
    }


@app.post("/click")
def click(req: ClickShareRequest):
    ensure_popularity_metrics()
    pop = PopularityCalculator()
    try:
        pop.add_click(req.doc_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating clicks: {e}")


@app.post("/share")
def share(req: ClickShareRequest):
    ensure_popularity_metrics()
    pop = PopularityCalculator()
    try:
        pop.add_share(req.doc_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating shares: {e}")
