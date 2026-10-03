# Multimodal Clinical AI Screening & Intelligence Platform

An orchestrated multimodal clinical screening platform that routes patient diagnostic imaging and clinical documentation to specialist AI models within a dynamic RAM budget, coupled with patient-partitioned clinical record retrieval (RAG) and transparent audit logging.

---

## Architecture Overview

- **Backend (FastAPI):**
  - Core screening endpoint: `POST /screen`
  - Model Manager with dynamic LRU eviction enforcing a 2GB RAM ceiling across specialist models (Brain MRI EfficientNet, Chest X-Ray Vision Transformer, etc.).
  - Clinical RAG pipeline using ChromaDB and `sentence-transformers/all-MiniLM-L6-v2`.
- **Frontend (Streamlit):**
  - Modern clinical decision support interface at `frontend/app.py`.
  - Visualizes diagnostic model findings, retrieved patient history, and real-time orchestration traces.

---

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend API

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start Streamlit Frontend

```bash
python -m streamlit run frontend/app.py
```

---

## Deployment

### 1. Backend Deployment (Render)

- **Platform:** [Render](https://render.com)
- **Configuration:** Render automatically detects and applies [`render.yaml`](render.yaml) located in the repository root.
- **Steps:**
  1. In the Render dashboard, click **New** > **Blueprint**.
  2. Connect your Git repository.
  3. Render will automatically configure a Python Web Service with:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
  4. Deploy the service and copy the live URL (e.g., `https://clinical-ai-backend.onrender.com`).

### 2. Frontend Deployment (Streamlit Community Cloud)

- **Platform:** [Streamlit Community Cloud](https://share.streamlit.io)
- **Main file path:** `frontend/app.py`
- **Steps:**
  1. Log in to Streamlit Community Cloud and click **New app**.
  2. Select your repository and branch.
  3. Set **Main file path** to `frontend/app.py`.
  4. Under **Advanced settings** > **Secrets / Environment Variables**, configure:
     ```bash
     BACKEND_URL="https://<YOUR-RENDER-BACKEND-URL>/screen"
     ```
     _(Example: `https://clinical-ai-backend.onrender.com/screen`)_
  5. # Click **Deploy**. The frontend will now call your production backend on Render while falling back to `http://localhost:8000/screen` during local development.
     A full-stack RAG (Retrieval-Augmented Generation) application featuring a FastAPI backend with ChromaDB vector search and a Streamlit interactive frontend.

## Architecture Overview

- **Backend**: FastAPI serving REST endpoints and managing retrieval workflows.
- **Vector Database**: ChromaDB for document storage and similarity search.
- **Frontend**: Streamlit web dashboard communicating with the backend via HTTP requests.

## Prerequisites

- **Python 3.10 to 3.12** installed on your system.
- _(Windows only)_ [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (required if building native dependencies for ChromaDB).

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/elevate-main.git
cd elevate-main
5bb8377aa4a0f5d82a2262ee153117df54b4ec7d
```
