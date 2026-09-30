"""
Clinical Document Ingestion Module for RAG Pipeline.

This module processes patient medical documents (PDFs via PyMuPDF/fitz and plain text),
splits content into token-sized overlapping chunks, and registers them into the
persistent ChromaDB vector store.
"""

import os
import sys
from typing import List, Dict, Any

# Ensure project root is on sys.path so 'backend.rag' can be imported when running standalone
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pymupdf as fitz  # Modern PyMuPDF import
except ImportError:
    import fitz  # Fallback for older versions

from backend.rag.vector_store import VectorStore


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text from all pages of a PDF document using PyMuPDF (fitz).
    """
    full_text = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def extract_text_from_txt(txt_path: str) -> str:
    """
    Reads plain text from a .txt file, attempting UTF-8 first with latin-1 fallback.
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="latin-1") as f:
            return f.read()


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of approximately ~500 tokens with ~50 token overlap.
    Using word-based segmentation provides clean linguistic boundaries while
    closely matching subword token counts.
    """
    words = text.split()
    if not words:
        return []

    # If document is shorter than one chunk size, return it as a single chunk
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(chunk_str)
        # Stop once the end of the text is reached
        if i + chunk_size >= len(words):
            break

    return chunks


def ingest_patient_documents(patient_id: str, folder_path: str) -> int:
    """
    Ingests all clinical documents for a patient from a target directory:
    1. Reads every PDF and .txt file in folder_path.
    2. Splits each document's text into ~500 token chunks with 50 token overlap.
    3. Stores each chunk via VectorStore.add_documents() with {patient_id, source_file}.

    Args:
        patient_id (str): Unique patient ID (e.g. 'patient_001').
        folder_path (str): Path to directory containing patient clinical files.

    Returns:
        int: Total number of chunks successfully ingested into VectorStore.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Patient document directory not found: {folder_path}")

    # Initialize the persistent vector store
    store = VectorStore()

    total_chunks = 0
    all_chunks: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    # List all files in the patient's directory
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # Skip subdirectories
        if os.path.isdir(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        extracted_text = ""

        # Step 1: Read supported file formats
        if ext == ".pdf":
            print(f"[ingest] Reading PDF: {filename}")
            extracted_text = extract_text_from_pdf(file_path)
        elif ext == ".txt":
            print(f"[ingest] Reading Text file: {filename}")
            extracted_text = extract_text_from_txt(file_path)
        else:
            # Skip unsupported file formats (images, audio, etc.)
            continue

        if not extracted_text.strip():
            print(f"[ingest] Warning: No text content found in {filename}")
            continue

        # Step 2: Split text into ~500 token chunks with 50 token overlap
        doc_chunks = split_text_into_chunks(extracted_text, chunk_size=500, overlap=50)
        print(f"[ingest] Generated {len(doc_chunks)} chunks for {filename}")

        # Step 3: Prepare metadata with patient_id and source_file
        for chunk in doc_chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "patient_id": patient_id,
                "source_file": filename
            })

    # Add all accumulated chunks to ChromaDB
    if all_chunks:
        store.add_documents(
            patient_id=patient_id,
            chunks=all_chunks,
            metadatas=all_metadatas
        )
        total_chunks = len(all_chunks)

    print(f"[ingest] Ingestion complete for '{patient_id}'. Total chunks stored: {total_chunks}")
    return total_chunks


if __name__ == "__main__":
    print("--- Running ingest.py manual test ---")

    # If folder passed in CLI arguments, use it; otherwise prepare a sample test folder
    if len(sys.argv) > 2:
        test_patient_id = sys.argv[1]
        test_folder = sys.argv[2]
    else:
        test_patient_id = "patient_001"
        test_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_patients", test_patient_id)
        )
        os.makedirs(test_folder, exist_ok=True)

        # Create a sample discharge summary text file if not present
        sample_txt = os.path.join(test_folder, "discharge_2023.txt")
        if not os.path.exists(sample_txt):
            with open(sample_txt, "w", encoding="utf-8") as f:
                f.write(
                    "CLINICAL DISCHARGE SUMMARY\n"
                    "Patient ID: patient_001\n"
                    "Admit Date: 2023-04-12 | Discharge Date: 2023-04-18\n"
                    "History of Present Illness: Patient has a 5-year documented history of essential hypertension "
                    "managed with Lisinopril 10mg daily. Reported persistent headaches and mild visual disturbances.\n"
                    "Hospital Course: Neurological evaluation noted no acute focal deficits. Brain MRI ordered for further screening.\n"
                    "Discharge Diagnoses: 1. Controlled Hypertension. 2. Chronic Tension Headaches. 3. Rule out intracranial space-occupying lesion.\n"
                    "Medications on Discharge: Lisinopril 10mg daily, Aspirin 81mg daily.\n"
                    "Follow-up: Outpatient neurology clinic in 4 weeks.\n"
                )
            print(f"Created sample clinical text file at: {sample_txt}")

        # Also create a sample PDF using PyMuPDF (fitz) to verify PDF ingestion
        sample_pdf = os.path.join(test_folder, "cardiology_consult.pdf")
        if not os.path.exists(sample_pdf):
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                fitz.Point(50, 72),
                "CARDIOLOGY CONSULTATION REPORT\n"
                "Patient: patient_001\n"
                "Cardiac Examination: Regular rate and rhythm, normal S1 and S2. No murmurs, rubs, or gallops.\n"
                "ECG Findings: Normal sinus rhythm with occasional premature ventricular contractions (PVCs).\n"
                "Recommendations: Continue current antihypertensive regimen. Repeat ECG in 6 months.\n"
            )
            doc.save(sample_pdf)
            doc.close()
            print(f"Created sample PDF document at: {sample_pdf}")

    stored_count = ingest_patient_documents(test_patient_id, test_folder)
    print(f"Ingestion verified successfully! Chunks stored in ChromaDB: {stored_count}")
