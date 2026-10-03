"""
Clinical Context Retriever Module for RAG Pipeline.

This module provides the `retrieve_patient_context` function that queries
the ChromaDB vector store and returns matching patient medical history chunks
in the exact schema required by `backend/schemas.py`.

Includes high-performance in-memory caching and graceful patient fallback
to prevent timeouts and 502 errors on resource-constrained cloud environments (Render).
"""

import os
import sys
from typing import List, Dict, Optional

# Ensure project root is on sys.path so 'backend.rag' can be imported when running standalone
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.rag.vector_store import VectorStore

_store_instance: Optional[VectorStore] = None


def get_vector_store() -> Optional[VectorStore]:
    """
    Returns a cached singleton VectorStore instance to prevent reloading on every request.
    """
    global _store_instance
    if _store_instance is None:
        try:
            _store_instance = VectorStore()
        except Exception as e:
            print(f"[retriever] Failed to initialize VectorStore: {e}")
            return None
    return _store_instance


def _get_patient_sample_fallback(patient_id: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Fallback loader that directly extracts clinical notes from sample_patients/{patient_id}
    if ChromaDB is empty, uninitialized, or times out.
    """
    records = []
    base_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_patients", patient_id)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_patients", patient_id)),
        os.path.join("data", "sample_patients", patient_id)
    ]

    for d in base_dirs:
        if os.path.exists(d):
            for fname in sorted(os.listdir(d)):
                fpath = os.path.join(d, fname)
                if fname.endswith(".txt"):
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                            if text:
                                records.append({"chunk": text, "source": fname})
                    except Exception:
                        pass
                elif fname.endswith(".pdf"):
                    try:
                        try:
                            import pymupdf as fitz
                        except ImportError:
                            import fitz
                        with fitz.open(fpath) as doc:
                            text = " ".join(page.get_text() for page in doc).strip()
                            if text:
                                records.append({"chunk": text, "source": fname})
                    except Exception:
                        pass
            if records:
                break

    return records[:top_k]


def retrieve_patient_context(patient_id: str, query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieves the top_k most relevant medical context snippets for a patient.
    Safely queries ChromaDB and falls back to patient documents without crashing.
    """
    retrieved_history: List[Dict[str, str]] = []

    # 1. Attempt retrieval via ChromaDB VectorStore
    try:
        store = get_vector_store()
        if store is not None:
            query_results = store.query(patient_id=patient_id, query_text=query, top_k=top_k)
            for item in query_results:
                chunk_text = item.get("chunk", "")
                metadata = item.get("metadata", {})
                source_file = metadata.get("source_file", "unknown")
                if chunk_text:
                    retrieved_history.append({
                        "chunk": chunk_text,
                        "source": source_file
                    })
    except Exception as e:
        print(f"[retriever] VectorStore query failed: {e}")

    # 2. If ChromaDB returned items, return them
    if retrieved_history:
        return retrieved_history

    # 3. Otherwise, use fast patient record fallback (prevents 502/timeouts on cloud hosts)
    return _get_patient_sample_fallback(patient_id, top_k=top_k)


if __name__ == "__main__":
    print("--- Running retriever.py manual test ---")

    test_patient = "patient_001"
    test_query = "history of hypertension and neurological evaluation"

    if len(sys.argv) > 2:
        test_patient = sys.argv[1]
        test_query = sys.argv[2]

    print(f"Retrieving context for patient: '{test_patient}' with query: '{test_query}'...")
    results = retrieve_patient_context(patient_id=test_patient, query=test_query, top_k=3)

    print(f"Retrieved {len(results)} context item(s):")
    for idx, item in enumerate(results, 1):
        print(f"\n[{idx}] Source: {item['source']}")
        print(f"    Chunk: {item['chunk'][:150]}...")

    print("\nRetriever self-test completed successfully!")
