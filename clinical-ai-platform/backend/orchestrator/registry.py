"""
Model Registry for Multimodal AI Specialist Models.

This registry catalogs all 6 specialist AI diagnostic models, their Python import
paths, expected input modalities, accepted file extensions, and estimated RAM footprints.
This enables the ModelManager and Router to dynamically load, route, and evict models
to remain strictly within a configured RAM budget.
"""

from typing import Dict, Any

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "brain_mri": {
        "module_path": "backend.models.brain_mri",
        "function_name": "predict",
        "input_type": "image",
        "accepted_extensions": [".jpg", ".png", ".jpeg", ".dcm"],
        "ram_cost_mb": 1200,
        "description": "EfficientNetB4 model for brain MRI tumor classification (glioma, meningioma, pituitary, no tumor)."
    },
    "chest_xray": {
        "module_path": "backend.models.chest_xray",
        "function_name": "predict",
        "input_type": "image",
        "accepted_extensions": [".jpg", ".png", ".jpeg"],
        "ram_cost_mb": 900,
        "description": "Vision Transformer (ViT) for chest X-ray pneumonia and normality screening."
    },
    "skin_lesion": {
        "module_path": "backend.models.skin_lesion",
        "function_name": "predict",
        "input_type": "image",
        "accepted_extensions": [".jpg", ".png", ".jpeg"],
        "ram_cost_mb": 700,
        "description": "Dermatology classifier for malignant vs benign skin lesion identification."
    },
    "ecg_arrhythmia": {
        "module_path": "backend.models.ecg_arrhythmia",
        "function_name": "predict",
        "input_type": "signal",
        "accepted_extensions": [".csv", ".dat", ".mat", ".npy"],
        "ram_cost_mb": 350,
        "description": "Cardiology signal processor for arrhythmia and rhythm anomaly detection."
    },
    "retinopathy": {
        "module_path": "backend.models.retinopathy",
        "function_name": "predict",
        "input_type": "image",
        "accepted_extensions": [".jpg", ".png", ".jpeg"],
        "ram_cost_mb": 800,
        "description": "Fundus image classifier for diabetic retinopathy grade estimation."
    },
    "lab_report_ner": {
        "module_path": "backend.models.lab_report_ner",
        "function_name": "predict",
        "input_type": "document",
        "accepted_extensions": [".pdf", ".txt", ".json"],
        "ram_cost_mb": 600,
        "description": "Clinical BioBERT NLP model for extracting abnormal lab biomarkers and entities."
    }
}
