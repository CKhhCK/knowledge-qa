"""SQLite-based user storage."""

import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Optional

from app.auth.security import hash_password, verify_password
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DB_PATH = "./data/users.db"


class UserStore:
    """Thread-safe SQLite user store."""

    def __init__(self, db_path: str = _DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
            conn.commit()
            conn.close()

    def create_user(self, username: str, password: str, email: str = "") -> dict:
        """Register a new user. Returns dict with user info or error."""
        username = username.strip().lower()

        if len(username) < 2:
            return {"success": False, "message": "用户名至少2个字符"}
        if len(password) < 6:
            return {"success": False, "message": "密码至少6个字符"}

        with self._lock:
            conn = sqlite3.connect(self._db_path)

            # Check duplicate
            existing = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                conn.close()
                return {"success": False, "message": f"用户名 '{username}' 已被注册"}

            user_id = uuid.uuid4().hex[:16]
            now = datetime.now().isoformat()

            conn.execute(
                "INSERT INTO users (user_id, username, password_hash, email, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, username, hash_password(password), email or "", now),
            )
            conn.commit()
            conn.close()

        logger.info(f"User registered: {username}")
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "email": email or "",
            "created_at": now,
        }

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Authenticate user. Returns user dict if valid, None otherwise."""
        username = username.strip().lower()

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT user_id, username, password_hash, email, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            conn.close()

        if not row:
            return None

        user_id, uname, pw_hash, email, created_at = row
        if not verify_password(password, pw_hash):
            return None

        return {
            "user_id": user_id,
            "username": uname,
            "email": email or "",
            "created_at": created_at,
        }

    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT user_id, username, email, created_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            conn.close()

        if not row:
            return None

        return {
            "user_id": row[0],
            "username": row[1],
            "email": row[2] or "",
            "created_at": row[3],
        }

    def count_users(self) -> int:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            conn.close()
            return count


# Global singleton
_user_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store
