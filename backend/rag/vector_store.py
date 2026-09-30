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
    # Check relative to this file: clinical-ai-platform/data/chroma_db
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_db"))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


class VectorStore:
    """
    Manages local persistent vector storage for patient records.
    
    Features:
    - PersistentClient saving to disk at data/chroma_db
    - Embedding function powered by sentence-transformers/all-MiniLM-L6-v2
    - Patient-partitioned retrieval via metadata filtering (where={"patient_id": patient_id})
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initializes the ChromaDB persistent client and sets up the embedding function.
        """
        self.persist_dir = persist_directory or get_chroma_db_dir()
        
        # 1. Initialize local persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 2. Configure embedding function using sentence-transformers/all-MiniLM-L6-v2
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        # 3. Create or retrieve collection with this embedding function
        self.collection = self.client.get_or_create_collection(
            name=DEFAULT_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "Patient multimodal electronic health records and notes"}
        )

    def add_documents(self, patient_id: str, chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """
        Stores text chunks and their associated metadata into the ChromaDB collection.

        Args:
            patient_id (str): The unique patient ID (e.g. 'patient_001').
            chunks (list[str]): List of extracted text strings.
            metadatas (list[dict]): List of metadata dictionaries (e.g. {"source_file": "report.pdf"}).
        """
        if not chunks:
            return

        # Ensure all metadata entries have the patient_id attached for filtered retrieval
        enriched_metadatas = []
        ids = []

        for i, meta in enumerate(metadatas):
            meta_copy = dict(meta)
            meta_copy["patient_id"] = patient_id
            enriched_metadatas.append(meta_copy)
            
            # Generate deterministic or unique ID for each chunk
            source = meta_copy.get("source_file", "doc")
            chunk_id = f"{patient_id}_{source}_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(chunk_id)

        # Upsert documents and metadata into the ChromaDB collection
        self.collection.add(
            documents=chunks,
            metadatas=enriched_metadatas,
            ids=ids
        )

    def query(self, patient_id: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for the top_k most relevant chunks for a specific patient.

        Args:
            patient_id (str): Target patient identifier.
            query_text (str): Query string (e.g. "history of tumor or hypertension").
            top_k (int): Number of most relevant chunks to return (default: 3).

        Returns:
            list[dict]: List of results with chunk text and metadata:
                        [{"chunk": "...", "metadata": {"source_file": "...", "patient_id": "..."}}]
        """
        # Count available documents for this patient to avoid top_k > count issues
        count = self.collection.count()
        if count == 0:
            return []

        # Filter specifically by patient_id so patient records never leak across patients
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
            where={"patient_id": patient_id}
        )

        formatted_results = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):
            formatted_results.append({
                "chunk": doc,
                "metadata": meta
            })

        return formatted_results
