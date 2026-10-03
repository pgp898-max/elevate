"""
FastAPI Backend Application
Multimodal AI Clinical Screening & Intelligence Platform

This module exposes the REST API for the platform, including the core
`POST /screen` endpoint that receives patient multimodal data, routes inputs
to specialist AI models via a resource-aware ModelManager, queries clinical RAG
history, and returns structured findings.
"""

import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, Form, UploadFile, status
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

# Import validation schemas
from backend.schemas import (
    ScreeningResponse,
    ModelFinding,
    RetrievedHistory,
    OrchestrationLogEntry
)

# Import orchestrator and RAG modules
import threading
from backend.orchestrator.router import route_files, route_files_with_inputs
from backend.orchestrator.model_manager import ModelManager
from backend.rag.retriever import retrieve_patient_context, get_vector_store

# 1. Initialize the FastAPI application instance
app = FastAPI(
    title="Multimodal Clinical AI Screening Platform",
    description="Unified API for routing patient imaging and clinical records to specialist AI models.",
    version="1.0.0"
)


def custom_openapi():
    """
    Ensures file upload parameters in OpenAPI 3.1 contain format: binary,
    allowing Swagger UI to render native file selection inputs instead of plain string arrays.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    for name, schema in openapi_schema.get("components", {}).get("schemas", {}).items():
        for prop_name, prop in schema.get("properties", {}).items():
            if prop.get("type") == "array" and "items" in prop:
                if prop_name == "files" or "contentMediaType" in prop.get("items", {}):
                    prop["items"]["format"] = "binary"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# 2. Configure Cross-Origin Resource Sharing (CORS)
# Allows the Streamlit frontend (:8501) to communicate with this backend (:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize a shared ModelManager instance (default 400MB RAM budget)
# This manager enforces LRU eviction when cumulative model footprints exceed the budget.
model_manager = ModelManager()


@app.on_event("startup")
def startup_init():
    """Initializes the vector store safely at boot."""
    try:
        get_vector_store()
    except Exception as e:
        print(f"[startup] Vector store initialization deferred: {e}")


@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """
    Root / Healthcheck endpoint.
    """
    return {
        "status": "healthy",
        "service": "Multimodal Clinical AI Screening API",
        "docs_url": "/docs"
    }


@app.post(
    "/screen",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen patient data across multimodal AI specialist models"
)
async def screen_patient(
    patient_id: str = Form(..., description="Unique patient identifier, e.g. 'patient_001'"),
    files: List[UploadFile] = File(
        default=[],
        description="One or more uploaded medical files (MRI/CT scans, chest X-rays, ECGs, clinical PDFs)"
    )
):
    """
    Primary clinical screening endpoint.

    Execution Pipeline:
    1. Saves uploaded files to a temporary workspace on disk.
    2. Calls router.route_files() to determine which specialist AI models should process the inputs.
    3. Runs each selected model via the shared ModelManager (enforcing RAM budget and LRU eviction).
    4. Queries patient-partitioned clinical records via RAG (retrieve_patient_context).
    5. Returns a structured JSON payload conforming to the exact platform API contract.
    """
    saved_file_paths: List[str] = []
    temp_dir = tempfile.mkdtemp(prefix=f"screening_{patient_id}_")

    try:
        # Step 1: Save uploaded files to the temporary directory
        if files:
            for upload in files:
                if not upload.filename:
                    continue
                file_dest = os.path.join(temp_dir, upload.filename)
                with open(file_dest, "wb") as buffer:
                    content = await upload.read()
                    buffer.write(content)
                saved_file_paths.append(file_dest)

        print(f"[/screen] Patient '{patient_id}': Saved {len(saved_file_paths)} input file(s) for screening.")

        # Step 2: Route files to appropriate specialist models
        # route_files returns model names; route_files_with_inputs associates each model with its file
        selected_models = route_files(saved_file_paths)
        model_assignments = route_files_with_inputs(saved_file_paths)
        print(f"[/screen] Router assigned models: {selected_models}")

        # Step 3: Execute models using the shared ModelManager
        model_findings: List[dict] = []
        for model_name, input_path in model_assignments:
            try:
                print(f"[/screen] Running '{model_name}' on '{os.path.basename(input_path)}'...")
                prediction = model_manager.run_model(model_name=model_name, input_path=input_path)
                model_findings.append({
                    "model": model_name,
                    "label": prediction["label"],
                    "confidence": prediction["confidence"]
                })
            except Exception as e:
                print(f"[/screen] Error executing model '{model_name}': {e}")
                # Ensure a failure in one model doesn't crash the entire screening report
                model_findings.append({
                    "model": model_name,
                    "label": f"Screening Error: {str(e)}",
                    "confidence": 0.0
                })

        # Step 4: Retrieve patient context via RAG
        # Formulate query focused on clinical history and the detected findings
        if model_findings:
            labels_text = ", ".join([f["label"] for f in model_findings if f["confidence"] > 0])
            rag_query = f"history of {labels_text} medical assessment discharge summary"
        else:
            rag_query = "patient medical history discharge summary clinical diagnosis"

        print(f"[/screen] Querying RAG context for '{patient_id}' with topic: '{rag_query[:60]}...'")
        try:
            retrieved_history = retrieve_patient_context(
                patient_id=patient_id,
                query=rag_query,
                top_k=3
            )
        except Exception as e:
            print(f"[/screen] Error retrieving patient context: {e}")
            retrieved_history = []

        # Step 5: Assemble response matching exact schema in backend/schemas.py
        response_payload = {
            "patient_id": patient_id,
            "model_findings": model_findings,
            "retrieved_history": retrieved_history,
            "orchestration_log": model_manager.get_log()
        }

        return response_payload

    finally:
        # Cleanup temporary uploaded files from disk to prevent storage bloat
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Clinical AI API server on port {port}...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
