"""
Chest X-Ray Classification Model
Model: NeuronZero/CXR-Classifier (Hugging Face)
Task: Classifies chest X-ray images (e.g., NORMAL vs PNEUMONIA)

This module provides a standalone `predict(image_path)` function.
It uses Hugging Face transformers with lazy loading so the model is
only loaded into RAM when actually invoked.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Hugging Face model identifier
MODEL_ID = "NeuronZero/CXR-Classifier"

# Module-level variables to hold the loaded processor and model in memory (singleton pattern)
_processor: Optional[AutoImageProcessor] = None
_model: Optional[AutoModelForImageClassification] = None


def load_model():
    """
    Loads the processor and model into memory if not already loaded.
    This 'lazy loading' pattern keeps RAM usage low until inference is needed.
    """
    global _processor, _model
    if _processor is None:
        # Load the feature extractor / image preprocessor suited for this ViT model
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    if _model is None:
        # Load the pre-trained weights for image classification
        _model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        # Set to evaluation mode (turns off dropout, batchnorm training updates)
        _model.eval()
    return _processor, _model


def predict(image_path: str) -> Dict[str, Any]:
    """
    Runs classification inference on a single chest X-ray image.

    Args:
        image_path (str): File path to the chest X-ray image (PNG, JPG, etc.).

    Returns:
        dict: Exact dictionary matching API contract:
              {"label": <str>, "confidence": <float>}
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Step 1: Ensure processor and model are loaded
    processor, model = load_model()

    # Step 2: Load the image using PIL and ensure standard 3-channel RGB format
    image = Image.open(image_path).convert("RGB")

    # Step 3: Preprocess image (resizes, normalizes, converts to PyTorch tensor)
    inputs = processor(images=image, return_tensors="pt")

    # Step 4: Perform inference without computing gradients (saves memory and CPU/GPU cycles)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Step 5: Convert raw logits to probabilities via softmax
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Step 6: Find the index with the highest probability
        top_prob, top_idx = torch.max(probabilities, dim=-1)
        idx_val = top_idx.item()

        # Step 7: Map the index back to human-readable label from model config
        # e.g., {0: "NORMAL", 1: "PNEUMONIA"}
        label = model.config.id2label.get(idx_val, model.config.id2label.get(str(idx_val), f"CLASS_{idx_val}"))
        confidence = float(top_prob.item())

    # Return result conforming to platform requirements
    return {
        "label": label,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    import sys
    import tempfile

    print("--- Running chest_xray.py self-test ---")

    # If an image path was passed as a CLI argument, test with it
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # Otherwise, create a synthetic test image to verify the inference pipeline end-to-end
        temp_dir = tempfile.gettempdir()
        test_path = os.path.join(temp_dir, "test_cxr_sample.png")
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy_img.save(test_path)
        print(f"Created synthetic test image at: {test_path}")

    result = predict(test_path)
    print(f"Prediction result: {result}")
    print("Self-test completed successfully!")
