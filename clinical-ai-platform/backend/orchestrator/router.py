"""
File Router for Multimodal AI Specialist Models.

This module inspects uploaded patient files, evaluates their file extensions
and modality signatures against the MODEL_REGISTRY, and routes each file to
the appropriate diagnostic model(s).
"""

import os
import sys
from typing import List, Tuple, Dict

# Ensure project root is available on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.orchestrator.registry import MODEL_REGISTRY


# Keyword hints to resolve modality when multiple models share identical extensions (e.g. .png, .jpg)
KEYWORD_HINTS: Dict[str, List[str]] = {
    "brain_mri": ["mri", "brain", "neuro", "cerebr", "head", "tumor"],
    "chest_xray": ["xray", "cxr", "chest", "lung", "pneumonia", "thorax"],
    "skin_lesion": ["skin", "lesion", "derm", "melanoma", "mole"],
    "ecg_arrhythmia": ["ecg", "ekg", "cardio", "arrhythmia", "lead", "pulse"],
    "retinopathy": ["retina", "eye", "fundus", "macula", "diabetic"],
    "lab_report_ner": ["lab", "blood", "report", "test", "cbc", "biomarker", "discharge"]
}


def _match_model_for_file(file_path: str) -> List[str]:
    """
    Identifies which model(s) should process a specific file based on:
    1. File extension compatibility with MODEL_REGISTRY
    2. Modality keyword hints in the filename (if ambiguous)
    """
    base_name = os.path.basename(file_path).lower()
    ext = os.path.splitext(file_path)[1].lower()

    # Find all models that accept this file extension
    candidate_models = [
        model_name
        for model_name, config in MODEL_REGISTRY.items()
        if ext in config.get("accepted_extensions", [])
    ]

    if not candidate_models:
        return []

    # If only one model accepts this extension (e.g. .csv -> ecg_arrhythmia), choose it directly
    if len(candidate_models) == 1:
        return candidate_models

    # If multiple models accept the extension (e.g., images .jpg/.png), check keyword hints in filename
    matched_by_keyword = []
    for model_name in candidate_models:
        hints = KEYWORD_HINTS.get(model_name, [])
        if any(hint in base_name for hint in hints):
            matched_by_keyword.append(model_name)

    if matched_by_keyword:
        return matched_by_keyword

    # Default fallback for generic image filenames without keywords:
    # Route to the primary active imaging specialists (brain_mri or chest_xray)
    if "brain_mri" in candidate_models and "chest_xray" in candidate_models:
        return ["brain_mri"]

    return [candidate_models[0]]


def route_files(file_paths: List[str]) -> List[str]:
    """
    Evaluates a list of input file paths and determines which specialist AI
    models should be executed.

    1. Checks each file's extension against MODEL_REGISTRY's accepted_extensions.
    2. Returns a list of unique model names that should run.
    3. Silently skips files that don't match any registered model without crashing.

    Args:
        file_paths (list[str]): List of absolute or relative file paths.

    Returns:
        list[str]: Unique list of model names to run (e.g. ["brain_mri", "chest_xray"]).
    """
    selected_models: List[str] = []

    for path in file_paths:
        matched = _match_model_for_file(path)
        for model_name in matched:
            if model_name not in selected_models:
                selected_models.append(model_name)

    return selected_models


def route_files_with_inputs(file_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Associates each selected model with its corresponding input file path.
    Useful for orchestrators that need to know which file to supply to which model.

    Returns:
        list of (model_name, file_path) pairs.
    """
    assignments: List[Tuple[str, str]] = []
    seen_models = set()

    for path in file_paths:
        matched = _match_model_for_file(path)
        for model_name in matched:
            if model_name not in seen_models:
                seen_models.add(model_name)
                assignments.append((model_name, path))

    return assignments
