"""
JurisMind Flask backend.

Endpoints:
  POST /api/load-models   -> load LegalBERT (+ optional T5) from disk
  GET  /api/status        -> check what's currently loaded
  POST /api/analyze       -> analyze a single comment, saves to DB
  POST /api/analyze-batch -> analyze a CSV file of comments, saves to DB
  GET  /api/history       -> list past analysis records (filterable)
  POST /api/wordcloud     -> word-frequency + wordcloud image for a list of texts
"""
import os
# Fix for a common Windows crash where PyTorch + MKL both load Intel's OpenMP
# runtime and the process gets killed (OMP Error #15). Must be set before torch imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import uuid
import base64
import logging
from collections import Counter

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from database import init_db, SessionLocal
from models import AnalysisRecord
from user_model import User
from auth import issue_token, revoke_token, require_auth
from inference import EnhancedLegalBERTAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jurismind.app")

app = Flask(__name__)
CORS(app)  # allow the Streamlit frontend (different port) to call this API

init_db()


def seed_default_admin():
    """Creates a default admin/admin123 account on first run, if no users exist yet."""
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(username="admin", role="admin")
            admin.set_password("admin123")
            db.add(admin)
            db.commit()
            logger.warning("Seeded default account admin/admin123 — change this password before deploying.")
    finally:
        db.close()


seed_default_admin()

# Global analyzer instance — loaded via /api/load-models
analyzer = None

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those',
}


def save_record(result, db, batch_id=None, batch_label=None, source_filename=None):
    record = AnalysisRecord(
        batch_id=batch_id,
        batch_label=batch_label,
        source_filename=source_filename,
        original_text=result["original_text"],
        analysis_text=result["analysis_text"],
        word_count=result["word_count"],
        is_long_comment=result["is_long_comment"],
        was_summarized=result["was_summarized"],
        summary_text=result.get("summary_text") or None,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
    )
    db.add(record)
    return record


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400

    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            return jsonify({"error": "username already taken"}), 409

        user = User(username=username, role="analyst")
        user.set_password(password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return jsonify(user.to_dict())
    finally:
        db.close()


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid username or password"}), 401

        token = issue_token(user.id)
        return jsonify({"token": token, "user": user.to_dict()})
    finally:
        db.close()


@app.route("/api/logout", methods=["POST"])
@require_auth
def logout():
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.split(" ", 1)[1]
    revoke_token(token)
    return jsonify({"status": "logged out"})


@app.route("/api/load-models", methods=["POST"])
@require_auth
def load_models():
    global analyzer
    data = request.get_json(force=True) or {}
    bert_path = data.get("bert_model_path", "./legal-bert-sentiment-model-3class")
    t5_path = data.get("t5_model_path")  # None to skip T5
    enable_t5 = data.get("enable_t5", False)

    try:
        analyzer = EnhancedLegalBERTAnalyzer(bert_path, t5_path if enable_t5 else None)
        return jsonify({
            "status": "ok",
            "bert_loaded": analyzer.bert_loaded,
            "t5_loaded": analyzer.t5_loaded,
        })
    except Exception as e:
        logger.exception("Model loading failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    if analyzer is None:
        return jsonify({"bert_loaded": False, "t5_loaded": False})
    return jsonify({"bert_loaded": analyzer.bert_loaded, "t5_loaded": analyzer.t5_loaded})


@app.route("/api/analyze", methods=["POST"])
@require_auth
def analyze_single():
    if analyzer is None or not analyzer.bert_loaded:
        return jsonify({"error": "Model not loaded. Call /api/load-models first."}), 400

    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    word_threshold = data.get("word_threshold", 100)

    if not text.strip():
        return jsonify({"error": "text field is required"}), 400

    result = analyzer.analyze_text(text, word_threshold=word_threshold, use_summarization=True)

    db = SessionLocal()
    try:
        record = save_record(result, db)
        db.commit()
        db.refresh(record)
        return jsonify(record.to_dict())
    finally:
        db.close()


@app.route("/api/analyze-batch", methods=["POST"])
@require_auth
def analyze_batch():
    if analyzer is None or not analyzer.bert_loaded:
        return jsonify({"error": "Model not loaded. Call /api/load-models first."}), 400

    if "file" not in request.files:
        return jsonify({"error": "CSV file is required under 'file' field"}), 400

    file = request.files["file"]
    text_column = request.form.get("text_column")
    word_threshold = int(request.form.get("word_threshold", 100))
    batch_label = request.form.get("batch_label") or None
    source_filename = file.filename

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    if text_column not in df.columns:
        return jsonify({"error": f"Column '{text_column}' not found. Available: {list(df.columns)}"}), 400

    batch_id = str(uuid.uuid4())
    db = SessionLocal()
    results = []
    try:
        for _, row in df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            if not text.strip():
                continue
            result = analyzer.analyze_text(text, word_threshold=word_threshold, use_summarization=True)
            record = save_record(result, db, batch_id=batch_id, batch_label=batch_label,
                                  source_filename=source_filename)
            results.append(record)
        db.commit()
        for r in results:
            db.refresh(r)

        payload = [r.to_dict() for r in results]
        sentiment_counts = Counter(r["sentiment"] for r in payload)

        return jsonify({
            "batch_id": batch_id,
            "batch_label": batch_label,
            "source_filename": source_filename,
            "count": len(payload),
            "sentiment_counts": dict(sentiment_counts),
            "results": payload,
        })
    finally:
        db.close()


@app.route("/api/batches", methods=["GET"])
@require_auth
def list_batches():
    """
    Returns one summary row per CSV upload session (batch), instead of every
    individual comment — this is what powers the grouped History view.
    Also includes single (non-batch) analyses grouped under batch_id=None.
    """
    db = SessionLocal()
    try:
        records = db.query(AnalysisRecord).order_by(AnalysisRecord.id.desc()).all()
        grouped = {}
        for r in records:
            key = r.batch_id or "single"
            if key not in grouped:
                grouped[key] = {
                    "batch_id": r.batch_id,
                    "batch_label": r.batch_label,
                    "source_filename": r.source_filename,
                    "count": 0,
                    "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            grouped[key]["count"] += 1
            s = r.sentiment if r.sentiment in ("positive", "negative", "neutral") else "neutral"
            grouped[key]["sentiment_counts"][s] += 1

        return jsonify(list(grouped.values()))
    finally:
        db.close()


@app.route("/api/batches/<batch_id>", methods=["DELETE"])
@require_auth
def delete_batch(batch_id):
    db = SessionLocal()
    try:
        query = db.query(AnalysisRecord)
        if batch_id == "single":
            query = query.filter(AnalysisRecord.batch_id.is_(None))
        else:
            query = query.filter(AnalysisRecord.batch_id == batch_id)
        deleted = query.delete(synchronize_session=False)
        db.commit()
        return jsonify({"deleted": deleted})
    finally:
        db.close()


@app.route("/api/history", methods=["GET"])
@require_auth
def history():
    sentiment_filter = request.args.get("sentiment")
    batch_id = request.args.get("batch_id")
    limit = int(request.args.get("limit", 100))

    db = SessionLocal()
    try:
        query = db.query(AnalysisRecord)
        if sentiment_filter:
            query = query.filter(AnalysisRecord.sentiment == sentiment_filter)
        if batch_id:
            query = query.filter(AnalysisRecord.batch_id == batch_id)
        records = query.order_by(AnalysisRecord.id.desc()).limit(limit).all()
        return jsonify([r.to_dict() for r in records])
    finally:
        db.close()


@app.route("/api/wordcloud", methods=["POST"])
@require_auth
def wordcloud():
    """Returns top word frequencies + a base64 PNG word cloud image for the given texts."""
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import re as _re

    data = request.get_json(force=True) or {}
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "texts list is required"}), 400

    all_text = " ".join(str(t) for t in texts)
    clean_text = _re.sub(r"[^\w\s]", " ", all_text.lower())
    clean_text = _re.sub(r"\d+", "", clean_text)
    words = [w for w in clean_text.split() if w not in STOPWORDS and len(w) > 2]

    word_freq = Counter(words)
    most_common = word_freq.most_common(50)

    if not most_common:
        return jsonify({"words": [], "image_base64": None})

    wc = WordCloud(width=800, height=400, background_color="white",
                    max_words=100, colormap="viridis").generate_from_frequencies(dict(most_common))

    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "words": [{"word": w, "count": c} for w, c in most_common],
        "image_base64": image_base64,
    })


if __name__ == "__main__":
    # use_reloader=False is important: Flask's reloader watches every file in
    # site-packages too, and PyTorch touches its own internal files during
    # model loading/inference. That was triggering false "file changed" restarts
    # mid-request, which killed the connection with a ConnectionResetError.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)