from __future__ import annotations
import os
from contextlib import contextmanager
from sqlmodel import SQLModel, Session, create_engine, select
from passlib.context import CryptContext
from .models import UserAccount, Project, ProjectMembership

DB_URL = os.getenv("ACCOUNTS_DB_URL", "sqlite:///accounts.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def init_db():
    SQLModel.metadata.create_all(engine)
    # Seed demo data if empty
    with Session(engine) as s:
        if s.exec(select(UserAccount).limit(1)).first() is None:
            s.add(UserAccount(username="alice", password_hash=pwd_context.hash("alice123"), role="staff", level=2, dept="DivisionA/Dept1"))
            s.add(UserAccount(username="bob", password_hash=pwd_context.hash("bob123"), role="manager", level=4, dept="DivisionA/Dept1"))
            s.add(UserAccount(username="carol", password_hash=pwd_context.hash("carol123"), role="admin", level=5, dept="DivisionA"))
        if s.exec(select(Project).limit(1)).first() is None:
            s.add(Project(name="ProjectX"))
            s.add(Project(name="ProjectY"))
        # Seed memberships
        if s.exec(select(ProjectMembership).limit(1)).first() is None:
            s.add(ProjectMembership(username="alice", project_name="ProjectX"))
            s.add(ProjectMembership(username="bob", project_name="ProjectX"))
            s.add(ProjectMembership(username="bob", project_name="ProjectY"))
            s.add(ProjectMembership(username="carol", project_name="ProjectX"))
            s.add(ProjectMembership(username="carol", project_name="ProjectY"))
        s.commit()

@contextmanager
def get_session():
    with Session(engine) as s:
        yield s
