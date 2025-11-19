from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator
from sqlmodel import SQLModel, Session, create_engine
import os

DB_PATH = os.path.join(os.getcwd(), "screening.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db() -> None:
    SQLModel.metadata.create_all(ENGINE)
    # Lightweight migrations for added columns
    with ENGINE.connect() as conn:
        # Check Candidate columns
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info('candidate')").fetchall()}
        if 'candidate_public_id' not in cols:
            conn.exec_driver_sql("ALTER TABLE candidate ADD COLUMN candidate_public_id TEXT")
        if 'status' not in cols:
            conn.exec_driver_sql("ALTER TABLE candidate ADD COLUMN status TEXT DEFAULT 'received'")
        # Backfill missing public ids
        try:
            rows = conn.exec_driver_sql("SELECT id, candidate_public_id FROM candidate").fetchall()
            import uuid
            for rid, pub in rows:
                if not pub:
                    new_id = uuid.uuid4().hex[:12]
                    conn.exec_driver_sql("UPDATE candidate SET candidate_public_id=? WHERE id=?", (new_id, rid))
        except Exception:
            pass


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(ENGINE) as session:
        yield session
