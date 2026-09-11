"""
JurisMind inference engine.

This is your original EnhancedLegalBERTAnalyzer logic (from app.py), stripped of all
Streamlit UI calls so it can run inside a Flask backend as a plain Python service.
"""
import os
import re
import json
import logging

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("jurismind.inference")

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False


class SimpleTextPreprocessor:
    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '', text
        )
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class EnhancedLegalBERTAnalyzer:
    """
    Loads LegalBERT (sentiment classification) and, optionally, T5 (summarization),
    and exposes analyze_text() for single-comment analysis.
    """

    def __init__(self, bert_model_path, t5_model_path=None):
        self.bert_loaded = False
        self.t5_loaded = False
        self.preprocessor = SimpleTextPreprocessor()

        self.load_bert_model(bert_model_path)

        if t5_model_path and T5_AVAILABLE:
            self.load_t5_model(t5_model_path)

    # ---------- Model loading ----------

    def load_bert_model(self, model_path):
        try:
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)

            with open(os.path.join(model_path, "reverse_mapping.json"), "r") as f:
                self.id_to_label = json.load(f)

            with open(os.path.join(model_path, "preprocessing_config.json"), "r") as f:
                self.config = json.load(f)

            self.bert_model.eval()
            self.bert_loaded = True
            logger.info("LegalBERT model loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to load BERT model: %s", e)
            self.bert_loaded = False
            raise

    def load_t5_model(self, model_path):
        try:
            abs_path = os.path.abspath(model_path)
            if not os.path.exists(abs_path):
                logger.warning("T5 model directory not found: %s", abs_path)
                return False

            config = T5Config.from_pretrained(abs_path, local_files_only=True)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                abs_path, config=config, local_files_only=True, torch_dtype=torch.float32
            )
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(abs_path, local_files_only=True)
            except Exception:
                self.t5_tokenizer = AutoTokenizer.from_pretrained(abs_path, local_files_only=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.t5_model = self.t5_model.to(device)
            self.t5_model.eval()
            self.t5_loaded = True
            logger.info("T5 model loaded from %s on %s", abs_path, device)
            return True
        except Exception as e:
            logger.error("Failed to load T5 model: %s", e)
            self.t5_loaded = False
            return False

    # ---------- Inference ----------

    def count_words(self, text):
        return len(str(text).split())

    def summarize_with_t5(self, text, max_length=100, min_length=30):
        if not self.t5_loaded:
            return text
        try:
            input_text = f"summarize: {text}"
            inputs = self.t5_tokenizer(
                input_text, return_tensors="pt", max_length=512, truncation=True, padding=True
            )
            device = next(self.t5_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                summary_ids = self.t5_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
            summary = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary if summary and summary.strip() else text
        except Exception as e:
            logger.warning("T5 summarization failed: %s", e)
            return text

    def analyze_sentiment(self, text):
        processed_text = self.preprocessor.clean_text(text)
        if len(processed_text.strip()) < 3:
            return {"sentiment": "neutral", "confidence": 0.0, "error": "Text too short after processing"}

        try:
            inputs = self.bert_tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.get("max_length", 512),
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_id].item()

            predicted_label = self.id_to_label[str(predicted_id)]
            return {"sentiment": predicted_label, "confidence": confidence}
        except Exception as e:
            return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}

    def analyze_text(self, text, word_threshold=100, use_summarization=True,
                      max_length=100, min_length=30):
        original_text = str(text)
        word_count = self.count_words(original_text)
        is_long_comment = word_count > word_threshold

        result = {
            "original_text": original_text,
            "word_count": word_count,
            "is_long_comment": is_long_comment,
            "was_summarized": False,
            "summary_text": "",
            "analysis_text": original_text,
        }

        if is_long_comment and use_summarization and self.t5_loaded:
            summary = self.summarize_with_t5(original_text, max_length, min_length)
            if summary and summary != original_text and len(summary.strip()) > 10:
                result["was_summarized"] = True
                result["summary_text"] = summary
                result["analysis_text"] = summary

        sentiment_result = self.analyze_sentiment(result["analysis_text"])
        result.update(sentiment_result)
        return result