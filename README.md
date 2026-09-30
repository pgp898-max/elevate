A full-stack RAG (Retrieval-Augmented Generation) application featuring a FastAPI backend with ChromaDB vector search and a Streamlit interactive frontend.

## Architecture Overview

- **Backend**: FastAPI serving REST endpoints and managing retrieval workflows.
- **Vector Database**: ChromaDB for document storage and similarity search.
- **Frontend**: Streamlit web dashboard communicating with the backend via HTTP requests.

## Prerequisites

- **Python 3.10 to 3.12** installed on your system.
- *(Windows only)* [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (required if building native dependencies for ChromaDB).

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/elevate-main.git
cd elevate-main
