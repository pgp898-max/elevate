"""
End-to-End Orchestrator and API Verification Script.
"""

import os
import sys
import tempfile
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.orchestrator.registry import MODEL_REGISTRY
from backend.orchestrator.router import route_files, route_files_with_inputs
from backend.orchestrator.model_manager import ModelManager


def test_registry_and_router():
    print("\n--- 1. Testing Registry & Router ---")
    assert len(MODEL_REGISTRY) == 6, f"Expected 6 models in registry, found {len(MODEL_REGISTRY)}"
    for name, config in MODEL_REGISTRY.items():
        assert "ram_cost_mb" in config
        assert "module_path" in config
        assert "function_name" in config
        print(f"  [OK] Registry entry for '{name}': {config['ram_cost_mb']}MB, extensions: {config['accepted_extensions']}")

    test_paths = [
        "c:/data/patient_mri_scan.png",
        "c:/data/chest_xray_normal.jpg",
        "c:/data/ecg_lead_ii.csv",
        "c:/data/unrelated_sound.mp3"  # Should be skipped without crashing
    ]
    routed = route_files(test_paths)
    print(f"  Routed models for {test_paths}: {routed}")
    assert "brain_mri" in routed
    assert "chest_xray" in routed
    assert "ecg_arrhythmia" in routed
    assert len(routed) == 3


def test_model_manager_lru():
    print("\n--- 2. Testing ModelManager LRU Eviction ---")
    # Budget of 2000MB:
    # brain_mri (1200MB) + chest_xray (900MB) = 2100MB > 2000MB -> Must evict brain_mri!
    mm = ModelManager(ram_budget_mb=2000)
    mm.load_model("brain_mri")
    assert "brain_mri" in mm.loaded_models
    assert mm.get_current_ram_usage() == 1200

    mm.load_model("chest_xray")
    assert "chest_xray" in mm.loaded_models
    assert "brain_mri" not in mm.loaded_models  # brain_mri was evicted!
    assert mm.get_current_ram_usage() == 900

    log = mm.get_log()
    print("  ModelManager event log:")
    for entry in log:
        print(f"    {entry}")
    assert any(e["event"] == "unloaded" and e["model"] == "brain_mri" for e in log)
    print("  [OK] LRU eviction verified successfully!")


def test_api_screen_endpoint():
    print("\n--- 3. Testing POST /screen Endpoint ---")
    client = TestClient(app)

    temp_dir = tempfile.gettempdir()
    mri_file = os.path.join(temp_dir, "test_patient_mri.png")
    Image.new("RGB", (380, 380), color=(60, 60, 60)).save(mri_file)

    cxr_file = os.path.join(temp_dir, "test_chest_xray.png")
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(cxr_file)

    with open(mri_file, "rb") as f1, open(cxr_file, "rb") as f2:
        response = client.post(
            "/screen",
            data={"patient_id": "patient_001"},
            files=[
                ("files", ("test_patient_mri.png", f1, "image/png")),
                ("files", ("test_chest_xray.png", f2, "image/png"))
            ]
        )

    print(f"  Response status code: {response.status_code}")
    assert response.status_code == 200
    data = response.json()

    print("  Response JSON keys:", list(data.keys()))
    assert data["patient_id"] == "patient_001"
    assert "model_findings" in data
    assert "retrieved_history" in data
    assert "orchestration_log" in data

    print(f"  Model Findings count: {len(data['model_findings'])}")
    for finding in data["model_findings"]:
        print(f"    - Model: {finding['model']} | Label: {finding['label']} | Confidence: {finding['confidence']}")

    print(f"  Retrieved History count: {len(data['retrieved_history'])}")
    for history in data["retrieved_history"]:
        print(f"    - Source: {history['source']} | Preview: {history['chunk'][:80]}...")

    print(f"  Orchestration Log events count: {len(data['orchestration_log'])}")
    for log_item in data["orchestration_log"]:
        print(f"    - Event: {log_item['event']} | Model: {log_item['model']} | RAM: {log_item['ram_mb']}MB")

    print("\n--- ALL VERIFICATIONS PASSED SUCCESSFULLY! ---")


if __name__ == "__main__":
    test_registry_and_router()
    test_model_manager_lru()
    test_api_screen_endpoint()
