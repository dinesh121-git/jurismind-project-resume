"""
JurisMind frontend — Streamlit UI.

This talks to the Flask backend over HTTP; it never loads ML models itself.
That separation (frontend / backend / database) is what makes this a full-stack app.
"""
import base64

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = "http://localhost:5000/api"

st.set_page_config(page_title="JurisMind — Sentiment & Summarization", page_icon="⚖️", layout="wide")


def auth_headers():
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(path, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update(auth_headers())
    return requests.get(f"{API_BASE}{path}", headers=headers, **kwargs)


def api_post(path, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.update(auth_headers())
    return requests.post(f"{API_BASE}{path}", headers=headers, **kwargs)


# ---------------- Login gate ----------------

if "token" not in st.session_state:
    st.title("⚖️ JurisMind")
    st.caption("AI-powered sentiment analysis and summarization for public policy comments")

    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            try:
                resp = requests.post(f"{API_BASE}/login", json={"username": username, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["token"] = data["token"]
                    st.session_state["user"] = data["user"]
                    st.rerun()
                else:
                    st.error(resp.json().get("error", "Login failed"))
            except requests.exceptions.ConnectionError:
                st.error("Could not reach backend. Is Flask running on port 5000?")

        st.caption("Default account for first-time setup: **admin / admin123** — change this in production.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password (min 6 characters)", type="password")
            reg_submitted = st.form_submit_button("Create account", type="primary")
        if reg_submitted:
            try:
                resp = requests.post(f"{API_BASE}/register", json={"username": new_username, "password": new_password})
                if resp.status_code == 200:
                    st.success("Account created — you can now log in from the 'Log in' tab.")
                else:
                    st.error(resp.json().get("error", "Registration failed"))
            except requests.exceptions.ConnectionError:
                st.error("Could not reach backend. Is Flask running on port 5000?")

    st.stop()  # halt here — nothing below renders until logged in


# ---------------- Defaults & auto-load on first visit ----------------

DEFAULT_BERT_PATH = "./legal-bert-sentiment-model-3class"
DEFAULT_T5_PATH = "./t5-model"
DEFAULT_WORD_THRESHOLD = 50

if "auto_load_attempted" not in st.session_state:
    st.session_state["auto_load_attempted"] = True
    with st.spinner("Starting up — loading models automatically..."):
        try:
            resp = api_post("/load-models", json={
                "bert_model_path": DEFAULT_BERT_PATH,
                "t5_model_path": DEFAULT_T5_PATH,
                "enable_t5": True,
            })
            if resp.status_code == 200:
                st.session_state["auto_load_result"] = resp.json()
        except requests.exceptions.ConnectionError:
            st.session_state["auto_load_result"] = None  # backend not up yet — admin panel can retry

# ---------------- Sidebar: account ----------------

st.sidebar.markdown(f"👤 **{st.session_state['user']['username']}**  ({st.session_state['user']['role']})")
if st.sidebar.button("🚪 Log out"):
    try:
        api_post("/logout")
    except requests.exceptions.ConnectionError:
        pass
    st.session_state.clear()
    st.rerun()

# ---------------- Sidebar: model loading (tucked away as "Admin Setup") ----------------

with st.sidebar.expander("⚙️ Admin Setup (model paths)", expanded=False):
    bert_path = st.text_input("LegalBERT model path", value=DEFAULT_BERT_PATH)
    enable_t5 = st.checkbox("Enable T5 summarization", value=True)
    t5_path = st.text_input("T5 model path", value=DEFAULT_T5_PATH) if enable_t5 else None
    word_threshold = st.slider("Word threshold for summarization", 20, 300, DEFAULT_WORD_THRESHOLD)

    if st.button("🚀 Reload Models"):
        with st.spinner("Loading models on the backend..."):
            try:
                resp = api_post("/load-models", json={
                    "bert_model_path": bert_path,
                    "t5_model_path": t5_path,
                    "enable_t5": enable_t5,
                })
                if resp.status_code == 200:
                    info = resp.json()
                    st.success(f"BERT: {info['bert_loaded']} | T5: {info['t5_loaded']}")
                else:
                    st.error(f"Error: {resp.json().get('message', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Could not reach backend. Is Flask running on port 5000?")

# Compact status badge instead of a technical readout
try:
    status = api_get("/status").json()
    ready = status["bert_loaded"]
    badge = "🟢 System ready" if ready else "🟡 Models not loaded — open Admin Setup in the sidebar"
    st.sidebar.markdown(f"**{badge}**")
except requests.exceptions.ConnectionError:
    st.sidebar.error("⚠️ Backend not reachable. Is the Flask server running?")
    status = {"bert_loaded": False, "t5_loaded": False}



st.title("⚖️ JurisMind")
st.caption("AI-powered sentiment analysis and summarization for public policy comments")

tab_analyze, tab_history = st.tabs(["📂 Analyze Comments", "🗂️ History"])

# ---------------- Tab 1: Analyze ----------------

with tab_analyze:
    if not status["bert_loaded"]:
        st.warning("Models are still loading in the background — this can take 10-30 seconds on first launch. "
                    "If this persists, open **Admin Setup** in the sidebar and click **Reload Models**.")

    mode = st.radio("Input mode", ["Single comment", "CSV upload"], horizontal=True)

    if mode == "Single comment":
        text = st.text_area("Enter a citizen comment:", height=150)
        if st.button("Analyze comment", type="primary") and text.strip():
            with st.spinner("Sending to backend for analysis..."):
                resp = api_post("/analyze", json={"text": text, "word_threshold": word_threshold})
            if resp.status_code == 200:
                r = resp.json()
                st.success(f"Sentiment: **{r['sentiment'].upper()}** (confidence {r['confidence']:.3f})")
                if r["was_summarized"]:
                    st.info(f"T5 summary used for analysis: {r['summary_text']}")
            else:
                st.error(resp.json().get("error", "Analysis failed"))

    else:
        uploaded = st.file_uploader("Upload a CSV of comments", type=["csv"])
        if uploaded is not None:
            preview_df = pd.read_csv(uploaded)
            st.dataframe(preview_df.head(10), use_container_width=True)
            text_column = st.selectbox("Text column to analyze", options=preview_df.columns)
            batch_label = st.text_input(
                "Label this batch (optional)",
                placeholder="e.g. Draft Bill 42 - June consultation",
            )

            if st.button("🔍 Run batch analysis", type="primary"):
                uploaded.seek(0)
                with st.spinner(f"Analyzing {len(preview_df)} comments on the backend..."):
                    resp = api_post(
                        "/analyze-batch",
                        files={"file": (uploaded.name, uploaded, "text/csv")},
                        data={
                            "text_column": text_column,
                            "word_threshold": word_threshold,
                            "batch_label": batch_label,
                        },
                    )
                if resp.status_code == 200:
                    batch = resp.json()
                    st.session_state["last_batch"] = batch
                    st.success(f"Analyzed {batch['count']} comments.")
                else:
                    st.error(resp.json().get("error", "Batch analysis failed"))

        if "last_batch" in st.session_state:
            batch = st.session_state["last_batch"]
            results_df = pd.DataFrame(batch["results"])

            col1, col2, col3, col4 = st.columns(4)
            counts = batch["sentiment_counts"]
            total = batch["count"]
            with col1:
                st.metric("✅ Positive", counts.get("positive", 0),
                          f"{counts.get('positive', 0) / total * 100:.1f}%" if total else "0%")
            with col2:
                st.metric("🔴 Negative", counts.get("negative", 0),
                          f"{counts.get('negative', 0) / total * 100:.1f}%" if total else "0%")
            with col3:
                st.metric("🔵 Neutral", counts.get("neutral", 0),
                          f"{counts.get('neutral', 0) / total * 100:.1f}%" if total else "0%")
            with col4:
                st.metric("🎯 Avg Confidence", f"{results_df['confidence'].mean():.3f}" if total else "0")

            fig = px.pie(values=list(counts.values()), names=list(counts.keys()),
                         title="Sentiment Distribution",
                         color_discrete_map={"positive": "#00c853", "negative": "#ff5252", "neutral": "#448aff"})
            st.plotly_chart(fig, use_container_width=True)

            # Word cloud via backend
            with st.spinner("Generating word cloud on backend..."):
                wc_resp = api_post("/wordcloud", json={"texts": results_df["original_text"].tolist()})
            if wc_resp.status_code == 200:
                wc_data = wc_resp.json()
                if wc_data.get("image_base64"):
                    st.subheader("☁️ Word Cloud")
                    img_bytes = base64.b64decode(wc_data["image_base64"])
                    st.image(img_bytes, use_column_width=True)
                    top_words_df = pd.DataFrame(wc_data["words"][:20])
                    st.dataframe(top_words_df, use_container_width=True)

            st.subheader("💬 Comments")
            sentiment_filter = st.selectbox("Filter by sentiment", ["All"] + list(counts.keys()))
            display_df = results_df if sentiment_filter == "All" else results_df[results_df["sentiment"] == sentiment_filter]
            st.dataframe(
                display_df[["original_text", "sentiment", "confidence", "was_summarized", "summary_text"]],
                use_container_width=True,
            )

            st.download_button(
                "📥 Download results (CSV)",
                data=results_df.to_csv(index=False),
                file_name="jurismind_batch_results.csv",
                mime="text/csv",
            )

# ---------------- Tab 2: History ----------------

with tab_history:
    st.subheader("📁 Past Analysis Sessions")
    st.caption("Each CSV upload (or single comment) is stored as its own session below.")

    try:
        batches_resp = api_get("/batches")
    except requests.exceptions.ConnectionError:
        st.error("Backend not reachable.")
        batches_resp = None

    if batches_resp is not None and batches_resp.status_code == 200:
        batches = batches_resp.json()

        if not batches:
            st.info("No analyses yet — run something from the **Analyze Comments** tab first.")
        else:
            for b in batches:
                key = b["batch_id"] or "single"
                total = b["count"]
                counts = b["sentiment_counts"]
                title = b["batch_label"] or b["source_filename"] or (
                    "Individually analyzed comments" if key == "single" else f"Batch {key[:8]}"
                )

                with st.expander(f"🗂️ {title}  •  {total} comments  •  {b['created_at'][:10] if b['created_at'] else ''}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("✅ Positive", counts.get("positive", 0))
                    c2.metric("🔴 Negative", counts.get("negative", 0))
                    c3.metric("🔵 Neutral", counts.get("neutral", 0))

                    if b["source_filename"]:
                        st.caption(f"Source file: `{b['source_filename']}`")

                    show_details = st.checkbox("Show individual comments", key=f"show_{key}")
                    if show_details:
                        detail_resp = api_get("/history", params={"batch_id": b["batch_id"], "limit": 500} if b["batch_id"] else {"limit": 500})
                        if detail_resp.status_code == 200:
                            detail_df = pd.DataFrame(detail_resp.json())
                            if b["batch_id"] is None:
                                detail_df = detail_df[detail_df["batch_id"].isna()]
                            if not detail_df.empty:
                                st.dataframe(
                                    detail_df[["original_text", "sentiment", "confidence", "was_summarized", "created_at"]],
                                    use_container_width=True,
                                )
                                st.download_button(
                                    "📥 Download this session (CSV)",
                                    data=detail_df.to_csv(index=False),
                                    file_name=f"jurismind_{title.replace(' ', '_')}.csv",
                                    mime="text/csv",
                                    key=f"dl_{key}",
                                )

                    if st.button("🗑️ Delete this session", key=f"del_{key}"):
                        del_resp = requests.delete(f"{API_BASE}/batches/{key}", headers=auth_headers())
                        if del_resp.status_code == 200:
                            st.success(f"Deleted {del_resp.json()['deleted']} records.")
                            st.rerun()
                        else:
                            st.error("Could not delete this session.")