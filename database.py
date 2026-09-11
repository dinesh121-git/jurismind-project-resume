"""
Database configuration for JurisMind.
Uses SQLite for simple, file-based persistence — no external DB server needed.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./jurismind.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Create all tables if they don't already exist."""
    import models  # noqa: F401 (ensures models are registered on Base)
    import user_model  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a DB session, ensuring it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()