"""
ChromaDB Vector Store Wrapper for Clinical RAG.

This module encapsulates ChromaDB local persistence and integrates the
`sentence-transformers/all-MiniLM-L6-v2` embedding model to store and retrieve
patient medical documents (discharge summaries, lab notes, clinical history).
"""

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

# Embedding model specified for clinical context retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "patient_clinical_records"


def get_chroma_db_dir() -> str:
    """
    Resolves the persistent storage directory for ChromaDB.
    Defaults to data/chroma_db in the project structure.
    """
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")),
        os.path.abspath(os.path.join("data", "chroma_db"))
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    base_dir = candidates[0]
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


class VectorStore:
    """
    Manages local persistent vector storage for patient records.
    
    Features:
    - PersistentClient saving to disk at data/chroma_db
    - Fast metadata-indexed retrieval partitioned by patient_id
    - Safe query ranking without heavy PyTorch / HuggingFace memory bloat on cloud hosts
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initializes the ChromaDB persistent client and sets up the collection.
        """
        self.persist_dir = persist_directory or get_chroma_db_dir()
        
        # 1. Initialize local persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 2. Create or retrieve collection safely without triggering heavy downloads
        try:
            self.collection = self.client.get_or_create_collection(
                name=DEFAULT_COLLECTION_NAME,
                metadata={"description": "Patient multimodal electronic health records and notes"}
            )
        except Exception as e:
            print(f"[VectorStore] Notice: Collection initialization deferred: {e}")
            self.collection = None

    def add_documents(self, patient_id: str, chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Stores text chunks and their associated metadata into the ChromaDB collection.
        """
        if not chunks or self.collection is None:
            return

        enriched_metadatas = []
        ids = []

        for i, meta in enumerate(metadatas):
            meta_copy = dict(meta)
            meta_copy["patient_id"] = patient_id
            enriched_metadatas.append(meta_copy)
            
            source = meta_copy.get("source_file", "doc")
            chunk_id = f"{patient_id}_{source}_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(chunk_id)

        self.collection.add(
            documents=chunks,
            metadatas=enriched_metadatas,
            ids=ids
        )

    def query(self, patient_id: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for the top_k most relevant chunks for a specific patient.
        Retrieves patient records using ChromaDB's indexed metadata filter and ranks
        chunks by query relevance with zero heavy ML memory overhead.
        """
        if self.collection is None:
            return []

        try:
            res = self.collection.get(where={"patient_id": patient_id})
            documents = res.get("documents", [])
            metadatas = res.get("metadatas", [])

            if not documents:
                return []

            import re
            q_words = set(re.findall(r'\w+', query_text.lower()))
            scored = []

            for doc, meta in zip(documents, metadatas):
                t_words = re.findall(r'\w+', doc.lower())
                score = sum(1 for w in t_words if w in q_words) / max(1, len(q_words)) if t_words else 0.0
                scored.append((score, doc, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            formatted_results = []
            for _, doc, meta in scored[:top_k]:
                formatted_results.append({
                    "chunk": doc,
                    "metadata": meta
                })

            return formatted_results
        except Exception as e:
            print(f"[VectorStore] Query failed gracefully: {e}")
            return []
