"""
Resource-Aware Model Manager with LRU Eviction.

This module enforces a strict RAM budget across multiple heavyweight clinical AI models.
When a model is requested:
1. It checks if loading the model fits within the RAM budget.
2. If memory is insufficient, it evicts least-recently-used (LRU) models to reclaim RAM.
3. It dynamically imports and runs the target model's `predict()` entry point.
4. It logs all load/unload events into an audit trail matching the API contract.
"""

import os
import sys
import time
import importlib
from typing import Dict, List, Any

# Ensure project root is available on sys.path for dynamic module importing
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.orchestrator.registry import MODEL_REGISTRY


class ModelManager:
    """
    Orchestrates dynamic model lifecycle and simulated RAM allocation.
    """

    def __init__(self, ram_budget_mb: int = 400):
        """
        Initializes the ModelManager with a maximum RAM budget in Megabytes.

        Args:
            ram_budget_mb (int): Simulated memory cap in MB (default: 400MB).
        """
        self.ram_budget_mb = ram_budget_mb

        # Stores currently loaded models mapped to their last-used epoch timestamp
        # Format: {model_name: timestamp}
        self.loaded_models: Dict[str, float] = {}

        # Audit log of resource management events
        # Format: [{"event": "loaded"/"unloaded", "model": <str>, "ram_mb": <int>}]
        self.log: List[Dict[str, Any]] = []

    def get_current_ram_usage(self) -> int:
        """
        Computes the total RAM consumed by all currently loaded models.
        """
        return sum(MODEL_REGISTRY[m]["ram_cost_mb"] for m in self.loaded_models if m in MODEL_REGISTRY)

    def load_model(self, model_name: str) -> None:
        """
        Ensures a model is loaded in memory within the RAM budget.
        If loading exceeds the budget, evicts the least-recently-used (LRU) model(s) first.

        Args:
            model_name (str): Identifier key present in MODEL_REGISTRY.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Not found in MODEL_REGISTRY.")

        # If already loaded, just update its last-used timestamp
        if model_name in self.loaded_models:
            self.loaded_models[model_name] = time.time()
            return

        needed_ram = MODEL_REGISTRY[model_name]["ram_cost_mb"]
        current_ram = self.get_current_ram_usage()

        # Evict least-recently-used models until there is enough headroom for the new model
        while (current_ram + needed_ram > self.ram_budget_mb) and self.loaded_models:
            # Find the model with the oldest timestamp (minimum value)
            lru_model = min(self.loaded_models, key=lambda k: self.loaded_models[k])
            evicted_ram = MODEL_REGISTRY[lru_model]["ram_cost_mb"]

            # Unload the LRU model
            del self.loaded_models[lru_model]
            current_ram -= evicted_ram

            # Record and display eviction event
            self.log.append({
                "event": "unloaded",
                "model": lru_model,
                "ram_mb": evicted_ram
            })
            print(f"[ModelManager] Unloaded '{lru_model}' to free {evicted_ram}MB (RAM budget: {self.ram_budget_mb}MB)")

        # Mark new model as loaded
        self.loaded_models[model_name] = time.time()
        self.log.append({
            "event": "loaded",
            "model": model_name,
            "ram_mb": needed_ram
        })
        print(f"[ModelManager] Loaded '{model_name}' ({needed_ram}MB) (RAM budget: {self.ram_budget_mb}MB)")

    def run_model(self, model_name: str, input_path: str) -> Dict[str, Any]:
        """
        Executes inference for a specific model:
        1. Ensures the model is loaded via `load_model()` (triggering LRU eviction if needed).
        2. Dynamically imports its module and entry-point function.
        3. Invokes inference and returns the predicted label and confidence.
        4. Updates the model's last-used timestamp.

        Args:
            model_name (str): Target model name.
            input_path (str): File path to the input data (image, signal, or document).

        Returns:
            dict: {"label": <str>, "confidence": <float>}
        """
        # Step 1: Resource management check / load
        self.load_model(model_name)

        # Step 2: Fetch registry entry
        entry = MODEL_REGISTRY[model_name]
        module_path = entry["module_path"]
        function_name = entry["function_name"]

        # Step 3: Dynamically import the module and execute the function
        module = importlib.import_module(module_path)
        predict_fn = getattr(module, function_name)
        result = predict_fn(input_path)

        # Step 4: Refresh last-used timestamp
        self.loaded_models[model_name] = time.time()

        return result

    def get_log(self) -> List[Dict[str, Any]]:
        """
        Returns the sequential audit log of all model load and unload events.
        """
        return list(self.log)
