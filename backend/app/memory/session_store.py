from __future__ import annotations
"""
SQLite-persisted conversation store — supports multiple sessions per user.

Tables:
  conversations (id, user_id, title, created_at, updated_at)
  messages (id, conversation_id, role, content, trace, created_at)

Survives server restarts. Multiple conversations per user.
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Optional

from app.models.chat import TraceInfo
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SessionStore:
    """SQLite-backed conversation store with multi-conversation support."""

    def __init__(self, db_path: str = "./data/conversations.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            # Remove corrupted/empty DB file so SQLite creates a clean one
            if os.path.exists(self._db_path):
                try:
                    test = sqlite3.connect(self._db_path)
                    test.execute("SELECT 1 FROM conversations LIMIT 1")
                    test.close()
                except Exception:
                    test.close()
                    os.remove(self._db_path)

            conn = sqlite3.connect(self._db_path)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    title TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    trace TEXT DEFAULT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
            """)
            conn.commit(); conn.close()

    # ================================================================
    # Conversation CRUD
    # ================================================================

    def create_conversation(self, user_id: str = "anonymous", title: str = "") -> dict:
        """Create a new conversation. Returns {conversation_id, ...}."""
        conv_id = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        if not title:
            title = "新对话"
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                (conv_id, user_id, title, now, now),
            )
            conn.commit(); conn.close()
        logger.info(f"Conversation created: {conv_id[:8]}...")
        return {"conversation_id": conv_id, "user_id": user_id, "title": title,
                "created_at": now, "message_count": 0}

    def list_conversations(self, user_id: str = "anonymous") -> list[dict]:
        """List all conversations for a user, newest first."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                """SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as msg_count
                   FROM conversations c WHERE c.user_id = ?
                   ORDER BY c.updated_at DESC""",
                (user_id,),
            ).fetchall()
            conn.close()
        return [{"conversation_id": r[0], "title": r[1], "created_at": r[2],
                 "updated_at": r[3], "message_count": r[4]} for r in rows]

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a single conversation by ID."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            conn.close()
        if not row:
            return None
        return {"conversation_id": row[0], "user_id": row[1], "title": row[2],
                "created_at": row[3], "updated_at": row[4]}

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            c = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            existed = c.rowcount > 0
            conn.commit(); conn.close()
        return existed

    # ================================================================
    # Messages
    # ================================================================

    def add_message(self, conversation_id: str, role: str, content: str,
                    trace: TraceInfo = None) -> None:
        """Add a single message to a conversation."""
        now = datetime.now().isoformat()
        trace_json = trace.model_dump_json() if trace else None
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, trace, created_at) VALUES (?,?,?,?,?)",
                (conversation_id, role, content, trace_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit(); conn.close()

    def add_qa_pair(self, conversation_id: str, user_msg: str, assistant_msg: str,
                    trace: TraceInfo = None) -> None:
        """Add a user+assistant message pair."""
        self.add_message(conversation_id, "user", user_msg)
        self.add_message(conversation_id, "assistant", assistant_msg, trace)

    def get_history(self, conversation_id: str) -> list[dict]:
        """Get all messages for a conversation."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT role, content, trace, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
            conn.close()
        return [{"role": r[0], "content": r[1],
                 "trace": json.loads(r[2]) if r[2] else None,
                 "timestamp": r[3]} for r in rows]

    def get_recent_history(self, conversation_id: str, limit: int = 10) -> list[dict]:
        """Get last N messages."""
        history = self.get_history(conversation_id)
        return history[-limit:]

    def get_user_messages(self, user_id: str = "anonymous", limit: int = 200) -> list[dict]:
        """Get recent messages across all conversations for a user (for semantic search)."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                """SELECT m.role, m.content, m.created_at, c.id as conversation_id
                   FROM messages m JOIN conversations c ON m.conversation_id = c.id
                   WHERE c.user_id = ? ORDER BY m.id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
            conn.close()
        # Return in chronological order
        return [{"role": r[0], "content": r[1], "timestamp": r[2], "conversation_id": r[3]}
                for r in reversed(rows)]

    def clear_conversation(self, conversation_id: str) -> bool:
        """Alias for delete_conversation."""
        return self.delete_conversation(conversation_id)

    def get_stats(self) -> dict:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            conn.close()
        return {"total_conversations": convs, "total_messages": msgs}

    def update_title(self, conversation_id: str, title: str) -> None:
        """Update conversation title."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
            conn.commit(); conn.close()
