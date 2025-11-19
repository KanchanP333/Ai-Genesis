from pydantic import BaseModel, Field
from typing import List, Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class User(BaseModel):
    username: str
    role: str
    level: int
    dept: str
    project: str

class UploadResponse(BaseModel):
    ids: List[str]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 8

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] = Field(default_factory=list)
