"""
Multimodal Clinical AI Screening Platform — Streamlit Frontend

Modernized clinical design system with deep blue/teal palette, responsive cards,
interactive metrics, and traceable orchestration audit logs.
"""

import html
import os
import pandas as pd
import requests
import streamlit as st

# Optional streamlit-extras styling
try:
    from streamlit_extras.metric_cards import style_metric_cards
    HAS_EXTRAS = True
except ImportError:
    HAS_EXTRAS = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Clinical AI Screening Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS: Modern Medical / Clinical Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* -------------------------------------------------------------
   CLINICAL DESIGN SYSTEM: Fonts & Variables
   ------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

:root {
    --primary-teal: #0d9488;
    --primary-dark: #0f766e;
    --navy-dark: #0f172a;
    --navy-header: #082f49;
    --sky-accent: #0284c7;
    --bg-light: #f8fafc;
    --card-border: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #64748b;
}

/* Base App Background & Padding */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* -------------------------------------------------------------
   CONTAINERS & CARDS
   ------------------------------------------------------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
    padding: 0.85rem 1rem !important;
    margin-bottom: 0.8rem;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 8px 16px -2px rgba(15, 23, 42, 0.06);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0 !important;
    padding: 1rem !important;
}

/* Custom Hero Banner */
.clinical-hero-banner {
    background: linear-gradient(135deg, #082f49 0%, #0369a1 50%, #0d9488 100%);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    color: #ffffff;
    box-shadow: 0 12px 28px -6px rgba(3, 105, 161, 0.28);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
    position: relative;
    overflow: hidden;
}

.clinical-hero-banner::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(20, 184, 166, 0.25) 0%, rgba(255,255,255,0) 70%);
    pointer-events: none;
}

.hero-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 1rem;
    position: relative;
    z-index: 1;
}

.hero-branding {
    max-width: 780px;
}

.hero-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 9999px;
    padding: 0.25rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #f0fdfa;
    margin-bottom: 0.6rem;
}

.live-dot {
    width: 7px;
    height: 7px;
    background-color: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
    display: inline-block;
}

.hero-title {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0.2rem 0 0.4rem 0 !important;
    letter-spacing: -0.025em;
    line-height: 1.25 !important;
}

.hero-subtitle {
    font-size: 0.96rem;
    color: rgba(240, 249, 255, 0.9);
    margin: 0;
    line-height: 1.5;
}

.orchestration-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.35);
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.45rem 0.95rem;
    border-radius: 9999px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    white-space: nowrap;
}

/* -------------------------------------------------------------
   SIDEBAR COMPONENTS
   ------------------------------------------------------------- */
.sidebar-header-box {
    margin-bottom: 0.5rem;
}
.sidebar-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.01em;
}
.sidebar-caption {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.2rem;
}
.input-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #f1f5f9;
}
.input-card-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f172a;
}
.modality-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.5rem;
}
.modality-pill {
    background: #f1f5f9;
    color: #475569;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}
.system-status-box {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-size: 0.8rem;
}
.status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px dashed #f1f5f9;
    padding-bottom: 0.3rem;
}
.status-label {
    color: #64748b;
    font-weight: 500;
}
.status-val {
    color: #0f172a;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.status-val-green {
    color: #059669;
    font-weight: 700;
}

/* -------------------------------------------------------------
   PATIENT RESULT BANNER
   ------------------------------------------------------------- */
.patient-result-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}
.patient-banner-left {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.patient-avatar {
    font-size: 1.8rem;
    background: #f0fdfa;
    border: 1px solid #ccfbf1;
    border-radius: 50%;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.patient-tag {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #0d9488;
    text-transform: uppercase;
}
.patient-id-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}
.patient-banner-stats {
    display: flex;
    gap: 1.5rem;
}
.banner-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.stat-number {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 0.2rem;
}

/* -------------------------------------------------------------
   AI MODEL FINDINGS CARDS
   ------------------------------------------------------------- */
.finding-info-block {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}
.model-badge-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    letter-spacing: 0.02em;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.confidence-tier-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    border: 1px solid;
}
.finding-label-heading {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.3;
    letter-spacing: -0.01em;
}
.finding-meta {
    font-size: 0.8rem;
    color: #64748b;
}

/* Circular Progress Meter */
.gauge-container {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 1rem;
}
.gauge-label-text {
    text-align: right;
}
.gauge-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
}
.gauge-score {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
}

/* Styled Streamlit Progress Bar (Gradient: Red to Amber to Emerald) */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%) !important;
    border-radius: 9999px !important;
    height: 8px !important;
}
div[data-testid="stProgress"] {
    margin-top: 0.35rem;
}

/* -------------------------------------------------------------
   RETRIEVED PATIENT HISTORY (QUOTE STYLE)
   ------------------------------------------------------------- */
.history-quote-card {
    border-left: 4px solid #0d9488;
    padding: 0.5rem 0.5rem 0.5rem 1.1rem;
    background: #ffffff;
    border-radius: 0 10px 10px 0;
}
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.quote-tag {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.quote-mark {
    font-size: 2rem;
    line-height: 0.8;
    color: #0d9488;
    font-family: Georgia, serif;
    font-weight: bold;
}
.chunk-index {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.rag-badge {
    background: #f0fdfa;
    color: #0f766e;
    border: 1px solid #ccfbf1;
    border-radius: 9999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.73rem;
    font-weight: 700;
}
.history-chunk-text {
    font-size: 0.92rem;
    line-height: 1.65;
    color: #334155;
    background: #f8fafc;
    padding: 0.9rem 1.1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.history-source-row {
    margin-top: 0.7rem;
    display: flex;
    align-items: center;
}
.source-tag-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 600;
}
.source-filename {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #14532d;
}

/* -------------------------------------------------------------
   LOADING SKELETON ANIMATION
   ------------------------------------------------------------- */
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.skeleton-container-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
}
.skeleton-header-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.clinical-spinner {
    width: 32px;
    height: 32px;
    border: 3.5px solid #ccfbf1;
    border-top: 3.5px solid #0d9488;
    border-radius: 50%;
    animation: spin 0.85s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.skeleton-title-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
}
.skeleton-sub-text {
    font-size: 0.85rem;
    color: #64748b;
}
.skeleton-cards-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}
.skeleton-shimmer-card {
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    border-radius: 10px;
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.shimmer-line {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}
.w-30 { width: 30%; }
.w-40 { width: 40%; }
.w-50 { width: 50%; }
.w-60 { width: 60%; }
.w-70 { width: 70%; }
.w-80 { width: 80%; }

/* -------------------------------------------------------------
   EMPTY STATE PLACEHOLDER
   ------------------------------------------------------------- */
.empty-state-container {
    background: #ffffff;
    border: 1.5px dashed #cbd5e1;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.02);
}
.empty-state-icon-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: #f0fdfa;
    border: 2px solid #ccfbf1;
    font-size: 2.2rem;
    margin-bottom: 1.25rem;
}
.empty-state-heading {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.5rem;
    letter-spacing: -0.015em;
}
.empty-state-subheading {
    font-size: 0.92rem;
    color: #64748b;
    max-width: 580px;
    margin: 0 auto 2rem auto;
    line-height: 1.5;
}
.workflow-steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.25rem;
    max-width: 820px;
    margin: 0 auto;
    text-align: left;
}
.workflow-step-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    position: relative;
    transition: transform 0.2s ease;
}
.workflow-step-card:hover {
    transform: translateY(-2px);
    border-color: #cbd5e1;
}
.step-num {
    font-size: 0.72rem;
    font-weight: 800;
    color: #0d9488;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.step-icon {
    font-size: 1.4rem;
    margin-bottom: 0.4rem;
}
.step-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
}
.step-desc {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.45;
}

/* -------------------------------------------------------------
   DISCLAIMER FOOTER
   ------------------------------------------------------------- */
.clinical-disclaimer-box {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    margin-top: 1.5rem;
}
.disclaimer-icon {
    font-size: 1.4rem;
}
.disclaimer-text {
    font-size: 0.82rem;
    color: #92400e;
    line-height: 1.45;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header: Gradient Banner with App Title, Subtitle, & Tagline
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="clinical-hero-banner">
        <div class="hero-header-row">
            <div class="hero-branding">
                <div class="hero-badge-pill">
                    <span class="live-dot"></span>
                    <span>Clinical Decision Support System</span>
                </div>
                <h1 class="hero-title">🩺 Multimodal Clinical AI Screening Platform</h1>
                <p class="hero-subtitle">
                    Orchestrated multi-model screening with patient-centric retrieval and traceable evidence.
                </p>
            </div>
            <div>
                <span class="orchestration-tag">⚡ Powered by AI Orchestration</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/screen")

# ---------------------------------------------------------------------------
# Sidebar — Inputs & Grouped Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header-box">
            <div class="sidebar-title">🏥 Patient Intake & Screening</div>
            <div class="sidebar-caption">Configure patient context and screening modalities</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Visually group Patient ID and file uploader inside a bordered card-like container
    with st.container(border=True):
        st.markdown(
            """
            <div class="input-card-header">
                <span class="input-card-title">📋 Patient Data & Files</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        patient_id = st.text_input(
            "🆔 Patient ID",
            value="patient_001",
            help="Unique identifier for patient records partition in RAG",
        )

        uploaded_files = st.file_uploader(
            "📂 Upload patient files (images, PDFs, or notes)",
            type=["jpg", "jpeg", "png", "pdf", "txt"],
            accept_multiple_files=True,
            help="Supported: Chest X-Rays, Brain MRIs, ECGs, Lab Reports, or Clinical Discharge Notes",
        )

        st.markdown(
            """
            <div class="modality-pills">
                <span class="modality-pill">🫁 Chest X-Ray</span>
                <span class="modality-pill">🧠 Brain MRI</span>
                <span class="modality-pill">❤️ ECG</span>
                <span class="modality-pill">📄 Clinical Notes</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("🚀 Run Screening", type="primary", use_container_width=True)

    # Sidebar system status card
    with st.container(border=True):
        st.markdown(
            """
            <div class="system-status-box">
                <div class="status-item">
                    <span class="status-label">Backend API</span>
                    <span class="status-val-green">● Connected (:8000)</span>
                </div>
                <div class="status-item">
                    <span class="status-label">RAM Budget</span>
                    <span class="status-val">2000 MB Max</span>
                </div>
                <div class="status-item">
                    <span class="status-label">RAG Engine</span>
                    <span class="status-val">ChromaDB Active</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Main Logic & Orchestration Execution
# ---------------------------------------------------------------------------
if run_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one file before running screening.")
    else:
        # Loading skeleton placeholder during execution
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown(
                """
                <div class="skeleton-container-box">
                    <div class="skeleton-header-row">
                        <div class="clinical-spinner"></div>
                        <div>
                            <div class="skeleton-title-text">Orchestrating Multimodal AI Pipeline...</div>
                            <div class="skeleton-sub-text">Routing inputs to specialist models, managing dynamic RAM budget, and querying clinical RAG history.</div>
                        </div>
                    </div>
                    <div class="skeleton-cards-grid">
                        <div class="skeleton-shimmer-card">
                            <div class="shimmer-line w-30"></div>
                            <div class="shimmer-line w-80"></div>
                            <div class="shimmer-line w-50"></div>
                        </div>
                        <div class="skeleton-shimmer-card">
                            <div class="shimmer-line w-40"></div>
                            <div class="shimmer-line w-70"></div>
                            <div class="shimmer-line w-60"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            finally:
                loading_placeholder.empty()

# ---------------------------------------------------------------------------
# Render Results (persisted in session_state so they survive reruns)
# ---------------------------------------------------------------------------
result = st.session_state.get("last_result")

if result:
    model_findings = result.get("model_findings", [])
    retrieved_history = result.get("retrieved_history", [])
    orchestration_log = result.get("orchestration_log", [])

    # Patient Result Overview Banner
    st.markdown(
        f"""
        <div class="patient-result-banner">
            <div class="patient-banner-left">
                <span class="patient-avatar">👤</span>
                <div>
                    <span class="patient-tag">PATIENT RECORD</span>
                    <div class="patient-id-title">{html.escape(str(result.get('patient_id', patient_id)))}</div>
                </div>
            </div>
            <div class="patient-banner-stats">
                <div class="banner-stat">
                    <span class="stat-number">{len(model_findings)}</span>
                    <span class="stat-label">Model Findings</span>
                </div>
                <div class="banner-stat">
                    <span class="stat-number">{len(retrieved_history)}</span>
                    <span class="stat-label">RAG Records</span>
                </div>
                <div class="banner-stat">
                    <span class="stat-number">{len(orchestration_log)}</span>
                    <span class="stat-label">Trace Events</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["🔬 AI Model Findings", "📋 Retrieved Patient History", "⚙️ Orchestration Trace"]
    )

    # ---- Section 1: Model Findings ----
    with tab1:
        if not model_findings:
            st.info("No model findings returned.")
        else:
            for idx, finding in enumerate(model_findings):
                model_name = finding.get("model", "unknown_model")
                label = finding.get("label", "N/A")
                confidence = float(finding.get("confidence", 0.0))
                pct = min(max(confidence, 0.0), 1.0) * 100.0

                # Modality specific styling
                mod_lower = model_name.lower()
                if "xray" in mod_lower:
                    mod_icon = "🫁"
                    mod_title = "Chest X-Ray Specialist"
                    badge_bg = "#0d9488"
                    badge_fg = "#ffffff"
                elif "mri" in mod_lower:
                    mod_icon = "🧠"
                    mod_title = "Brain MRI Specialist"
                    badge_bg = "#4f46e5"
                    badge_fg = "#ffffff"
                elif "arrhythmia" in mod_lower or "ecg" in mod_lower:
                    mod_icon = "❤️"
                    mod_title = "ECG Specialist"
                    badge_bg = "#e11d48"
                    badge_fg = "#ffffff"
                elif "ner" in mod_lower or "lab" in mod_lower:
                    mod_icon = "🧪"
                    mod_title = "Clinical NLP & NER"
                    badge_bg = "#d97706"
                    badge_fg = "#ffffff"
                else:
                    mod_icon = "🔬"
                    mod_title = model_name.replace("_", " ").title()
                    badge_bg = "#0284c7"
                    badge_fg = "#ffffff"

                # Confidence tier and color
                if confidence >= 0.85:
                    tier_text = "High Diagnostic Confidence"
                    tier_color = "#059669"
                    tier_icon = "●"
                    stroke_color = "#10b981"
                elif confidence < 0.60:
                    tier_text = "Low Confidence / Inconclusive"
                    tier_color = "#dc2626"
                    tier_icon = "●"
                    stroke_color = "#ef4444"
                else:
                    tier_text = "Moderate Diagnostic Confidence"
                    tier_color = "#d97706"
                    tier_icon = "●"
                    stroke_color = "#f59e0b"

                with st.container(border=True):
                    col_info, col_meter = st.columns([3, 2], vertical_alignment="center")

                    with col_info:
                        st.markdown(
                            f"""
                            <div class="finding-info-block">
                                <div class="model-badge-row">
                                    <span class="model-pill" style="background-color: {badge_bg}; color: {badge_fg};">
                                        {mod_icon} {mod_title}
                                    </span>
                                    <span class="confidence-tier-pill" style="color: {tier_color}; border-color: {tier_color}40; background-color: {tier_color}12;">
                                        {tier_icon} {tier_text}
                                    </span>
                                </div>
                                <div class="finding-label-heading">{html.escape(label)}</div>
                                <div class="finding-meta">Target Specialist: <code>{html.escape(model_name)}</code> &bull; Automated multi-modal inference</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col_meter:
                        st.markdown(
                            f"""
                            <div class="gauge-container">
                                <svg viewBox="0 0 36 36" style="width: 76px; height: 76px; transform: rotate(-90deg);">
                                    <path stroke="#e2e8f0" stroke-width="3.5" fill="none"
                                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                    <path stroke="{stroke_color}" stroke-width="3.5" stroke-dasharray="{pct:.1f}, 100" stroke-linecap="round" fill="none"
                                          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                </svg>
                                <div class="gauge-label-text">
                                    <div class="gauge-title">Confidence</div>
                                    <div class="gauge-score" style="color: {tier_color};">{pct:.1f}%</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(
                            min(max(confidence, 0.0), 1.0),
                            text=f"Confidence: {confidence * 100:.1f}%"
                        )

    # ---- Section 2: Retrieved Patient History ----
    with tab2:
        if not retrieved_history:
            st.info("No historical context retrieved for this patient.")
        else:
            for idx, item in enumerate(retrieved_history):
                chunk = item.get("chunk", "")
                source = item.get("source", "unknown source")
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="history-quote-card">
                            <div class="history-header">
                                <div class="quote-tag">
                                    <span class="quote-mark">“</span>
                                    <span class="chunk-index">Clinical Record #{idx + 1}</span>
                                </div>
                                <span class="rag-badge">🔍 RAG Verified Context</span>
                            </div>
                            <div class="history-chunk-text">
                                {html.escape(chunk)}
                            </div>
                            <div class="history-source-row">
                                <span class="source-tag-pill">
                                    <span class="source-icon">📄</span>
                                    <span class="source-label">Source Citation:</span>
                                    <span class="source-filename">{html.escape(source)}</span>
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ---- Section 3: Orchestration Trace ----
    with tab3:
        if not orchestration_log:
            st.info("No orchestration events recorded.")
        else:
            models_run = len({e.get("model") for e in orchestration_log if e.get("event") == "loaded"})
            peak_ram = max((e.get("ram_mb", 0) for e in orchestration_log), default=0)
            total_events = len(orchestration_log)
            loaded_count = sum(1 for e in orchestration_log if e.get("event") == "loaded")
            unloaded_count = sum(1 for e in orchestration_log if e.get("event") == "unloaded")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="Specialist Models Run",
                    value=f"{models_run}",
                    delta=f"{loaded_count} loads" if loaded_count else None,
                    delta_color="normal",
                )
            with m2:
                ram_headroom = 2000 - peak_ram
                st.metric(
                    label="Peak Simulated RAM",
                    value=f"{peak_ram} MB",
                    delta=f"{ram_headroom} MB Headroom" if peak_ram <= 2000 else f"+{abs(ram_headroom)} MB Over Budget",
                    delta_color="normal" if peak_ram <= 2000 else "inverse",
                )
            with m3:
                st.metric(
                    label="Audit Trace Events",
                    value=f"{total_events}",
                    delta=f"{loaded_count} in / {unloaded_count} out",
                    delta_color="off",
                )

            if HAS_EXTRAS:
                try:
                    style_metric_cards(
                        border_left_color="#0d9488",
                        border_radius_px=10,
                        box_shadow=True,
                    )
                except Exception:
                    pass

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            formatted_log = []
            for idx, entry in enumerate(orchestration_log, start=1):
                raw_event = str(entry.get("event", "")).strip().lower()
                if raw_event == "loaded":
                    event_badge = "🟢 LOADED (In Memory)"
                elif raw_event == "unloaded":
                    event_badge = "🔴 UNLOADED (Evicted)"
                else:
                    event_badge = f"⚪ {raw_event.upper()}"

                formatted_log.append({
                    "Step": idx,
                    "event": event_badge,
                    "model": entry.get("model", "unknown"),
                    "ram_mb": int(entry.get("ram_mb", 0)),
                })

            df_log = pd.DataFrame(formatted_log)

            st.dataframe(
                df_log,
                column_order=["Step", "event", "model", "ram_mb"],
                column_config={
                    "Step": st.column_config.NumberColumn("#", width="small", help="Execution sequence"),
                    "event": st.column_config.TextColumn("Lifecycle Event", width="medium", help="Model allocation or eviction event"),
                    "model": st.column_config.TextColumn("Specialist Model", width="medium"),
                    "ram_mb": st.column_config.ProgressColumn(
                        "Simulated RAM",
                        format="%d MB",
                        min_value=0,
                        max_value=2000,
                        help="Allocated memory against 2000 MB limit",
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

else:
    # Improved friendly empty state
    st.markdown(
        """
        <div class="empty-state-container">
            <div class="empty-state-illustration">
                <div class="empty-state-icon-circle">
                    🩺
                </div>
            </div>
            <h3 class="empty-state-heading">Ready for Clinical Screening</h3>
            <p class="empty-state-subheading">
                Enter a Patient ID and upload diagnostic media (Chest X-Ray, Brain MRI, ECG, or clinical notes) in the sidebar to initiate multi-model AI screening.
            </p>
            <div class="workflow-steps-grid">
                <div class="workflow-step-card">
                    <div class="step-num">STEP 01</div>
                    <div class="step-icon">👤</div>
                    <div class="step-title">Patient Context</div>
                    <div class="step-desc">Confirm Patient ID to query patient-partitioned EHR history via RAG.</div>
                </div>
                <div class="workflow-step-card">
                    <div class="step-num">STEP 02</div>
                    <div class="step-icon">📂</div>
                    <div class="step-title">Multimodal Upload</div>
                    <div class="step-desc">Supply medical imaging or discharge records for automated routing.</div>
                </div>
                <div class="workflow-step-card">
                    <div class="step-num">STEP 03</div>
                    <div class="step-icon">⚡</div>
                    <div class="step-title">AI Orchestration</div>
                    <div class="step-desc">Specialist models execute with dynamic LRU memory eviction.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div class="clinical-disclaimer-box">
        <span class="disclaimer-icon">⚠️</span>
        <span class="disclaimer-text">
            <strong>Clinical Decision Support Disclaimer:</strong> This platform is designed solely for investigational and AI-assisted screening support. 
            It does not constitute a definitive medical diagnosis. All findings, confidence scores, and historical retrievals must be independently reviewed and verified by a licensed healthcare professional.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

