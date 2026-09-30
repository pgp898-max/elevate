"""
Clinical Context Retriever Module for RAG Pipeline.

This module provides the `retrieve_patient_context` function that queries
the ChromaDB vector store and returns matching patient medical history chunks
in the exact schema required by `backend/schemas.py`.
"""

import os
import sys
from typing import List, Dict

# Ensure project root is on sys.path so 'backend.rag' can be imported when running standalone
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.rag.vector_store import VectorStore


def retrieve_patient_context(patient_id: str, query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieves the top_k most relevant medical context snippets for a patient.

    Args:
        patient_id (str): Unique patient ID (e.g. 'patient_001').
        query (str): Clinical question or query topic (e.g. 'hypertension history or brain tumor symptoms').
        top_k (int): Maximum number of snippets to retrieve (default: 3).

    Returns:
        list[dict]: Results matching the backend/schemas.py RetrievedHistory schema:
                    [
                      {"chunk": "Patient has history of hypertension...", "source": "discharge_2023.pdf"}
                    ]
    """
    # 1. Initialize VectorStore instance connected to persistent ChromaDB
    store = VectorStore()

    # 2. Query patient-partitioned records
    query_results = store.query(patient_id=patient_id, query_text=query, top_k=top_k)

    # 3. Format into the exact contract shape: [{"chunk": ..., "source": ...}]
    retrieved_history: List[Dict[str, str]] = []
    for item in query_results:
        chunk_text = item.get("chunk", "")
        metadata = item.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")

        retrieved_history.append({
            "chunk": chunk_text,
            "source": source_file
        })

    return retrieved_history


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
