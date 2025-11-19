from __future__ import annotations
from typing import Optional
from sqlmodel import SQLModel, Field

class UserAccount(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default="staff")  # staff | manager | admin
    level: int = Field(default=1)
    dept: str = Field(default="")

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

class ProjectMembership(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True)
    project_name: str = Field(index=True)
