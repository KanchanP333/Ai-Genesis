from __future__ import annotations
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
import uuid


def new_public_id() -> str:
    return uuid.uuid4().hex[:12]


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    public_id: str = Field(default_factory=new_public_id, index=True, unique=True)
    title: str
    description: str
    constraints: Optional[str] = Field(default=None, description="Any specific requirements or constraints")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id")
    candidate_public_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], index=True, unique=True)
    name: str
    email: str
    resume_text: str
    extra_inputs: Optional[str] = Field(default=None, description="Free-form inputs from candidate")
    score: Optional[float] = Field(default=None, index=True)
    summary: Optional[str] = None
    fits: Optional[bool] = Field(default=None, index=True)
    status: str = Field(default="received", index=True, description="received|under_review|accepted|rejected")
    created_at: datetime = Field(default_factory=datetime.utcnow)
