"""
Pydantic Schemas for the Clinical AI Screening Platform.

These schemas define and validate the data structures used by the API endpoints,
ensuring strict adherence to the project API contract.
"""

from typing import List
from pydantic import BaseModel, Field


class ModelFinding(BaseModel):
    """
    Represents an individual specialist AI diagnostic finding.
    Example:
        {"model": "brain_mri", "label": "Tumor detected", "confidence": 0.87}
    """
    model: str = Field(..., description="Name of the specialist model (e.g. brain_mri, chest_xray)")
    label: str = Field(..., description="Predicted medical condition or finding label")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")


class RetrievedHistory(BaseModel):
    """
    Represents an extracted context chunk from the patient's medical records (RAG).
    Example:
        {"chunk": "Patient has history of hypertension...", "source": "discharge_2023.pdf"}
    """
    chunk: str = Field(..., description="Extracted text snippet from patient record")
    source: str = Field(..., description="Source document filename or identifier")


class OrchestrationLogEntry(BaseModel):
    """
    Represents a resource management event logged during model orchestration.
    Example:
        {"event": "loaded", "model": "brain_mri", "ram_mb": 1200}
    """
    event: str = Field(..., description="Event type: 'loaded', 'unloaded', 'swapped', etc.")
    model: str = Field(..., description="Target model involved in this resource event")
    ram_mb: int = Field(..., description="Estimated RAM consumed or released in Megabytes")


class ScreeningResponse(BaseModel):
    """
    The exact top-level JSON response contract returned by the /screen endpoint.
    
    Shape:
    {
      "patient_id": "patient_001",
      "model_findings": [...],
      "retrieved_history": [...],
      "orchestration_log": [...]
    }
    """
    patient_id: str = Field(..., description="Identifier for the screened patient")
    model_findings: List[ModelFinding] = Field(
        default_factory=list,
        description="List of findings returned by diagnostic specialist models"
    )
    retrieved_history: List[RetrievedHistory] = Field(
        default_factory=list,
        description="Relevant background history retrieved from medical records via RAG"
    )
    orchestration_log: List[OrchestrationLogEntry] = Field(
        default_factory=list,
        description="Audit trace of model dynamic loading/unloading and RAM consumption"
    )
