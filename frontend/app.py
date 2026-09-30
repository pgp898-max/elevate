"""
Multimodal Clinical AI Screening Platform — Streamlit Frontend
Place this file at: frontend/app.py in your project.
Run with: python -m streamlit run frontend/app.py
(Make sure your FastAPI backend is already running on http://localhost:8000)
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Clinical AI Screening Platform",
    layout="wide",
)

st.title("🩺 Multimodal Clinical AI Screening Platform")
st.caption("Orchestrated multi-model screening with patient-centric retrieval and traceable evidence.")

BACKEND_URL = "http://localhost:8000/screen"

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Input")
    patient_id = st.text_input("Patient ID", value="patient_001")

    uploaded_files = st.file_uploader(
        "Upload patient files (images, PDFs, or notes)",
        type=["jpg", "jpeg", "png", "pdf", "txt"],
        accept_multiple_files=True,
    )

    run_clicked = st.button("Run Screening", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if run_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one file before running screening.")
    else:
        with st.spinner("Running orchestration..."):
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploaded_files
                ]
                data_payload = {"patient_id": patient_id}

                response = requests.post(
                    BACKEND_URL,
                    data=data_payload,
                    files=files_payload,
                    timeout=120,
                )

                if response.status_code != 200:
                    st.error(
                        f"Backend returned an error (status {response.status_code}). "
                        f"Details: {response.text}"
                    )
                else:
                    result = response.json()
                    st.session_state["last_result"] = result

            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the backend at http://localhost:8000. "
                    "Make sure your FastAPI server is running (`python -m uvicorn backend.main:app --reload`)."
                )
            except Exception as e:
                st.error(f"Something went wrong while calling the backend: {e}")

# ---------------------------------------------------------------------------
# Render results (persisted in session_state so they survive reruns)
# ---------------------------------------------------------------------------
result = st.session_state.get("last_result")

if result:
    model_findings = result.get("model_findings", [])
    retrieved_history = result.get("retrieved_history", [])
    orchestration_log = result.get("orchestration_log", [])

    tab1, tab2, tab3 = st.tabs(
        ["🔬 AI Model Findings", "📋 Retrieved Patient History", "⚙️ Orchestration Trace"]
    )

    # ---- Section 1: Model Findings ----
    with tab1:
        if not model_findings:
            st.info("No model findings returned.")
        for finding in model_findings:
            model_name = finding.get("model", "unknown_model")
            label = finding.get("label", "N/A")
            confidence = float(finding.get("confidence", 0))

            col1, col2 = st.columns([2, 3])
            with col1:
                if confidence >= 0.85:
                    st.success(f"**{model_name}** → {label}")
                elif confidence < 0.6:
                    st.warning(f"**{model_name}** → {label}")
                else:
                    st.write(f"**{model_name}** → {label}")
            with col2:
                st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {confidence * 100:.1f}%")

    # ---- Section 2: Retrieved Patient History ----
    with tab2:
        if not retrieved_history:
            st.info("No historical context retrieved for this patient.")
        for item in retrieved_history:
            chunk = item.get("chunk", "")
            source = item.get("source", "unknown source")
            with st.container(border=True):
                st.write(chunk)
                st.caption(f"*Source: {source}*")

    # ---- Section 3: Orchestration Trace ----
    with tab3:
        if not orchestration_log:
            st.info("No orchestration events recorded.")
        else:
            models_run = len({e.get("model") for e in orchestration_log if e.get("event") == "loaded"})
            peak_ram = max((e.get("ram_mb", 0) for e in orchestration_log), default=0)

            m1, m2 = st.columns(2)
            m1.metric("Models Run", models_run)
            m2.metric("Peak Simulated RAM (MB)", peak_ram)

            st.dataframe(
                orchestration_log,
                column_order=["event", "model", "ram_mb"],
                use_container_width=True,
            )

else:
    st.info("Upload patient files in the sidebar and click **Run Screening** to begin.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ This tool is for AI-assisted screening support only, and is not a substitute "
    "for professional medical diagnosis."
)
