# Clinical AI Platform

A multimodal, AI-powered clinical screening and intelligence platform that coordinates specialized medical AI models with patient history and RAG-based evidence retrieval.

## Key Features

* 🧠 Multimodal analysis: MRI, X-ray, ECG, medical images, and clinical documents
* 🤖 6+ specialized AI models across different medical domains
* 🔀 Intelligent model orchestration and routing
* 💾 Resource-aware model loading/unloading for limited RAM
* 📚 Patient-centric RAG for medical history and reports
* 📋 Unified, explainable clinical screening report
* 🔍 Traceability of model findings and retrieved evidence

## Architecture

```text
Patient Data
     ↓
Model Orchestrator
     ↓
Specialized AI Models
     ↓
Patient RAG + Medical History
     ↓
Unified Clinical Screening Report
```

## Tech Stack

**Backend:** FastAPI · Python
**Frontend:** Streamlit
**RAG:** ChromaDB · Sentence Transformers
**AI/ML:** Specialized multimodal models

> **Disclaimer:** This platform is designed for clinical screening and decision support, not autonomous medical diagnosis.
