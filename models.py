"""
ORM models for JurisMind's SQLite database.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class AnalysisRecord(Base):
    """A single analyzed comment, whether submitted individually or as part of a batch CSV upload."""

    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, index=True, nullable=True)  # groups rows from the same CSV upload
    batch_label = Column(String, nullable=True)  # user-given name, e.g. "June consultation - Draft Bill 42"
    source_filename = Column(String, nullable=True)  # original CSV filename, if uploaded as a batch
    original_text = Column(Text, nullable=False)
    analysis_text = Column(Text, nullable=False)  # text actually fed to BERT (summary if long, else original)
    word_count = Column(Integer, default=0)
    is_long_comment = Column(Boolean, default=False)
    was_summarized = Column(Boolean, default=False)
    summary_text = Column(Text, nullable=True)
    sentiment = Column(String, index=True, nullable=False)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "batch_label": self.batch_label,
            "source_filename": self.source_filename,
            "original_text": self.original_text,
            "analysis_text": self.analysis_text,
            "word_count": self.word_count,
            "is_long_comment": self.is_long_comment,
            "was_summarized": self.was_summarized,
            "summary_text": self.summary_text,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }