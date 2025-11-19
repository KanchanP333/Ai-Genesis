import os
import time
from typing import Optional, Dict
from jose import jwt, JWTError
from passlib.context import CryptContext
try:
    from app.accounts.db import get_session as accounts_session
    from app.accounts.models import UserAccount, ProjectMembership
    from sqlmodel import select
    _HAS_ACCOUNTS = True
except Exception:
    _HAS_ACCOUNTS = False

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_SECONDS = 60*60*8

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Demo in-memory users: username -> {password_hash, role, level}
_USERS: Dict[str, Dict] = {
    "alice": {"password_hash": pwd_context.hash("alice123"), "role": "staff", "level": 2, "dept": "DivisionA/Dept1", "project": "ProjectX"},
    "bob": {"password_hash": pwd_context.hash("bob123"), "role": "manager", "level": 4, "dept": "DivisionA/Dept1", "project": "ProjectX"},
    "carol": {"password_hash": pwd_context.hash("carol123"), "role": "admin", "level": 5, "dept": "DivisionA", "project": "ProjectX"},
}

ROLE_WEIGHT = {"staff": 1, "manager": 2, "admin": 3}

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    # DB-backed auth (required)
    if not _HAS_ACCOUNTS:
        return None
    with accounts_session() as s:
        acct = s.exec(select(UserAccount).where(UserAccount.username == username)).first()
        if not acct:
            return None
        if not verify_password(password, acct.password_hash):
            return None
        # Pick first membership project if any
        proj = s.exec(select(ProjectMembership).where(ProjectMembership.username == username)).first()
        project = proj.project_name if proj else ""
        return {
            "username": username,
            "role": acct.role,
            "level": acct.level,
            "dept": acct.dept or "",
            "project": project,
            "role_weight": ROLE_WEIGHT.get(acct.role, 0)
        }

def create_access_token(sub: str, role: str, level: int, dept: str, project: str) -> str:
    now = int(time.time())
    payload = {
        "sub": sub,
        "role": role,
        "level": level,
        "dept": dept,
        "project": project,
        "exp": now + ACCESS_TOKEN_EXPIRE_SECONDS,
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> Optional[Dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError:
        return None
