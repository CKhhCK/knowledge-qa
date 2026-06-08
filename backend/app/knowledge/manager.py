"""
KnowledgeManager — RAG pipeline with semantic search, HyDE, MQE, and Rerank.

Features:
- Vector embedding search (sentence-transformers or TF-IDF fallback)
- HyDE: Hypothetical Document Embeddings — LLM generates a fake answer,
  embed that, and search with it (finds semantically similar docs)
- MQE: Multi-Query Expansion — LLM generates N query variants,
  search with each, merge results by max-score dedup
- Rerank: LLM scores retrieved chunks for relevance, re-orders top results
- Smart chunking with paragraph/sentence boundary awareness
- Multi-type memory (working, episodic, semantic) with SQLite persistence
"""

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

from app.config import Settings
from app.knowledge.embedding import (
    embed_texts, embed_query, embed_documents, cosine_similarity, get_embedding_dim,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ================================================================
# BM25 + Vector Hybrid Index
# ================================================================

from collections import Counter
from math import log

class BM25Index:
    """BM25 keyword search — exact matching for numbers, code, proper nouns."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1; self.b = b
        self._chunks: list[str] = []          # chunk text
        self._chunk_ids: list[tuple] = []      # (doc_id, chunk_idx)
        self._df: Counter = Counter()          # document frequency
        self._avg_len = 0
        self._total = 0

    def add_document(self, doc_id: str, chunks: list[str]):
        for i, c in enumerate(chunks):
            self._chunk_ids.append((doc_id, i))
            self._chunks.append(c)
            for term in set(self._tokenize(c)):
                self._df[term] += 1
        self._avg_len = sum(len(c) for c in self._chunks) / max(len(self._chunks), 1)
        self._total = len(self._chunks)

    def remove_document(self, doc_id: str):
        indices = [j for j, (d, _) in enumerate(self._chunk_ids) if d == doc_id]
        for j in reversed(indices):
            for term in set(self._tokenize(self._chunks[j])):
                self._df[term] = max(0, self._df[term] - 1)
            del self._chunks[j]; del self._chunk_ids[j]
        self._total = len(self._chunks)
        self._avg_len = sum(len(c) for c in self._chunks) / max(self._total, 1)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        terms = self._tokenize(query)
        if not terms: return []
        scores = []
        for idx, chunk in enumerate(self._chunks):
            score = 0.0
            doc_len = len(chunk)
            for term in terms:
                tf = chunk.lower().count(term)
                if tf == 0: continue
                df = self._df.get(term, 0)
                idf = log((self._total - df + 0.5) / (df + 0.5) + 1)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_len, 1)))
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"score": round(s, 4), "chunk_idx": self._chunk_ids[i][1],
                 "document_id": self._chunk_ids[i][0], "chunk": self._chunks[i]} for s, i in scores[:limit]]

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        for word in re.findall(r'[一-鿿]+|[a-z0-9]+|\d+\.?\d*%?', text.lower()):
            if len(word) >= 1: tokens.append(word)
        return tokens

    def clear(self):
        self._chunks.clear(); self._chunk_ids.clear(); self._df.clear()
        self._avg_len = 0; self._total = 0


class VectorIndex:
    """Semantic search index with SQLite persistence + BM25 hybrid search."""

    RRF_K = 60  # Reciprocal Rank Fusion constant

    def __init__(self, db_path: str = "./data/vector_index.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._bm25 = BM25Index()
        self._init_db()
        self._chunks: list[dict] = []
        self._documents: dict[str, dict] = {}
        self._load_from_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_chunks (
                    doc_id TEXT NOT NULL, chunk_idx INTEGER NOT NULL,
                    text TEXT NOT NULL, embedding TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    PRIMARY KEY (doc_id, chunk_idx)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc ON doc_chunks(doc_id)")
            conn.commit(); conn.close()

    def _load_from_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            rows = list(conn.execute("SELECT doc_id, chunk_idx, text, embedding, metadata FROM doc_chunks"))
            conn.close()
        for row in rows:
            self._chunks.append({"doc_id": row[0], "chunk_idx": row[1], "text": row[2], "embedding": json.loads(row[3])})
            if row[0] not in self._documents:
                self._documents[row[0]] = json.loads(row[4]) if row[4] else {}
        # Rebuild BM25 from loaded chunks
        for doc_id in self._documents:
            chunks = [c["text"] for c in self._chunks if c["doc_id"] == doc_id]
            if chunks: self._bm25.add_document(doc_id, chunks)
        if self._chunks:
            logger.info(f"Loaded {len(self._chunks)} chunks ({len(self._documents)} docs) + BM25")

    def add_document(self, doc_id: str, chunks: list[str], metadata: dict = None):
        if not chunks: return
        logger.info(f"Embedding {len(chunks)} chunks for '{doc_id}' ({get_embedding_dim()}d)...")
        t0 = time.perf_counter()
        embeddings = embed_documents(chunks)
        logger.info(f"Embedded {len(chunks)} chunks in {(time.perf_counter()-t0)*1000:.0f}ms")
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                conn.execute("INSERT OR REPLACE INTO doc_chunks VALUES (?,?,?,?,?)",
                             (doc_id, i, chunk, json.dumps(emb), meta_json))
                self._chunks.append({"doc_id": doc_id, "chunk_idx": i, "text": chunk, "embedding": emb})
            conn.commit(); conn.close()
        self._documents[doc_id] = metadata or {}
        self._bm25.add_document(doc_id, chunks)

    def remove_document(self, doc_id: str):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM doc_chunks WHERE doc_id = ?", (doc_id,))
            conn.commit(); conn.close()
        self._chunks = [c for c in self._chunks if c["doc_id"] != doc_id]
        self._documents.pop(doc_id, None)
        self._bm25.remove_document(doc_id)

    def search_vector(self, query_embedding: list[float], limit: int = 10,
                      score_threshold: float = 0.0) -> list[dict]:
        """Pure vector search."""
        if not self._chunks: return []
        scored = []
        for chunk in self._chunks:
            score = cosine_similarity(query_embedding, chunk["embedding"])
            if score >= score_threshold:
                scored.append({"document_id": chunk["doc_id"], "score": round(score, 4),
                               "chunk": chunk["text"], "chunk_idx": chunk["chunk_idx"],
                               "metadata": self._documents.get(chunk["doc_id"], {})})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def search_hybrid(self, query: str, query_embedding: list[float],
                      limit: int = 5) -> list[dict]:
        """BM25 + Vector hybrid search using Reciprocal Rank Fusion."""
        # Get results from both methods (more candidates)
        bm25_results = self._bm25.search(query, limit=limit * 3)
        vector_results = self.search_vector(query_embedding, limit=limit * 3)

        # Weighted RRF: vector 2x (better semantic understanding), BM25 1x (keyword precision)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, r in enumerate(bm25_results):
            key = f"{r['document_id']}_{r['chunk_idx']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.RRF_K + rank + 1)
            chunk_map[key] = r

        for rank, r in enumerate(vector_results):
            key = f"{r['document_id']}_{r['chunk_idx']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 2.0 / (self.RRF_K + rank + 1)  # 2x weight
            if key not in chunk_map:
                chunk_map[key] = r
            if key not in chunk_map:
                chunk_map[key] = r

        # Sort by RRF score
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, rrf_score in merged[:limit]:
            r = chunk_map[key]
            r["score"] = round(rrf_score, 4)
            results.append(r)
        return results

    def clear(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM doc_chunks"); conn.commit(); conn.close()
        self._chunks.clear(); self._documents.clear()
        self._bm25.clear()

    def stats(self) -> dict:
        return {"total_chunks": len(self._chunks), "total_documents": len(self._documents),
                "embedding_dimension": get_embedding_dim(), "document_ids": list(self._documents.keys())}


# ================================================================
# HyDE (Hypothetical Document Embeddings)
# ================================================================

HYDE_PROMPT = """根据用户问题，写一段可能的答案段落，用于改善检索效果。
要求：中等长度，客观，包含关键术语。直接写段落，不要写"答案是"之类的开头。

问题：{query}"""


# ================================================================
# MQE (Multi-Query Expansion)
# ================================================================

MQE_PROMPT = """你是检索查询扩展助手。生成语义等价或互补的多样化查询。
原始查询：{query}
请给出{n}个不同表述的查询，每行一个，简洁直接。"""


# ================================================================
# Rerank Prompt
# ================================================================

RERANK_PROMPT = """评估以下文档片段与用户问题的相关性。为每个片段打分（0-10）。

用户问题: {query}

文档片段:
{chunks}

请输出JSON数组: [{"idx": 0, "score": 8.5, "reason": "直接相关"}, ...]"""


# ================================================================
# Text Chunking Strategies
# ================================================================

CHUNK_STRATEGIES = ["fixed", "recursive", "markdown"]
CHUNK_LABELS = {
    "fixed": "固定字符",
    "recursive": "递归语义",
    "markdown": "Markdown标题",
}


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200,
               strategy: str = "recursive") -> list[str]:
    """Split text using the specified strategy."""
    if strategy == "fixed":
        return chunk_fixed(text, chunk_size, chunk_overlap)
    elif strategy == "markdown":
        return chunk_markdown(text, chunk_size)
    else:
        return chunk_recursive(text, chunk_size, chunk_overlap)


def chunk_fixed(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Fixed-size chunks with overlap, no structure awareness."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    return [c for c in chunks if c]


def chunk_recursive(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[str]:
    """
    Recursive chunking — tries paragraph first, then sentence, then fixed-size.

    The chunk_size is a TARGET, not a hard limit. It will:
    1. Merge paragraphs together until approaching chunk_size
    2. If a single paragraph exceeds chunk_size, split at sentences
    3. If a single sentence exceeds chunk_size, split at fixed-size
    """
    if len(text) <= chunk_size:
        return [text]

    # Phase 1: Split by paragraphs first
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size * 1.5:
            # Merge into current chunk
            current = (current + "\n\n" + para).strip()
        else:
            # Save current chunk
            if current.strip():
                chunks.append(current.strip())
            # If this paragraph alone is huge, split it further
            if len(para) > chunk_size * 1.5:
                sub_chunks = _split_by_sentences(para, chunk_size)
                chunks.extend(sub_chunks)
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _split_by_sentences(text: str, chunk_size: int) -> list[str]:
    """Split long text at sentence boundaries, keeping within chunk_size."""
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current = (current + sent).strip()
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(sent) > chunk_size:
                # Single sentence too long: fixed-size fallback
                for i in range(0, len(sent), chunk_size):
                    chunks.append(sent[i:i+chunk_size].strip())
            else:
                current = sent

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_markdown(text: str, max_chunk_size: int = 2000) -> list[str]:
    """
    Markdown-aware chunking: split by ## headers, keep sections together.
    If a section is still too large, fall back to recursive within it.
    """
    # Split by ## headers (h2)
    sections = re.split(r'\n(?=## )', text)

    chunks = []
    current = ""

    for section in sections:
        if len(current) + len(section) < max_chunk_size:
            # Merge into current chunk
            current = (current + "\n\n" + section).strip()
        else:
            # Save current chunk if any
            if current.strip():
                if len(current) > max_chunk_size * 1.5:
                    # Still too big, recursive-split it
                    chunks.extend(chunk_recursive(current, max_chunk_size, 200))
                else:
                    chunks.append(current.strip())
            current = section

    if current.strip():
        if len(current) > max_chunk_size * 1.5:
            chunks.extend(chunk_recursive(current, max_chunk_size, 200))
        else:
            chunks.append(current.strip())

    return [c for c in chunks if c]


# ================================================================
# Memory Store — follows hello-agents MemoryTool pattern
#
# Types (same as hello-agents):
#   working  — current session context (temporary)
#   episodic — conversation records, events (persistent, SQLite)
#   semantic — knowledge, facts (persistent, SQLite)
#
# Search uses embedding vectors (Qwen3), NOT keyword matching.
# ================================================================

class MemoryStore:
    """Multi-type memory store with embedding-based semantic search.

    Follows hello-agents MemoryTool pattern:
    - working memory: temporary session context
    - episodic memory: persistent conversation records (原文对话)
    - semantic memory: persistent knowledge facts
    - Search uses embedding vectors + cosine similarity
    """

    def __init__(self, db_path: str = "./data/memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'anonymous',
                    memory_type TEXT NOT NULL DEFAULT 'working', content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5, metadata TEXT DEFAULT '{}',
                    embedding TEXT DEFAULT NULL,
                    created_at TEXT NOT NULL, accessed_at TEXT NOT NULL
                )
            """)
            try: conn.execute("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'")
            except: pass
            try: conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT DEFAULT NULL")
            except: pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.commit(); conn.close()

    def add(self, content: str, memory_type: str = "working",
            importance: float = 0.5, user_id: str = "anonymous", **metadata) -> str:
        """Add a memory. Embedding stored as NULL, computed on-demand during search."""
        mem_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO memories (id, user_id, memory_type, content, importance, metadata, embedding, created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (mem_id, user_id, memory_type, content, importance,
                 json.dumps(metadata, ensure_ascii=False), now, now),
            )
            conn.commit(); conn.close()
        return mem_id

    def search(self, query: str, limit: int = 5, memory_type: str = None,
               user_id: str = "anonymous") -> list[dict]:
        """
        Semantic search — embed query once, compare with pre-computed memory embeddings.
        """
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            sql = "SELECT id, memory_type, content, importance, metadata, embedding, created_at FROM memories WHERE user_id = ?"
            params = [user_id]
            if memory_type:
                sql += " AND memory_type = ?"; params.append(memory_type)
            rows = list(conn.execute(sql, params))
            conn.close()

        if not rows:
            return []

        # Embed query once
        q_emb = None
        try:
            q_emb = embed_query(query)
        except Exception:
            pass

        results = []
        for row in rows:
            mem_id, mtype, content, importance, meta_str, emb_str, created_at = row
            score = 0.0

            # Compare with pre-computed embedding
            if q_emb and emb_str:
                try:
                    mem_emb = json.loads(emb_str)
                    score = cosine_similarity(q_emb, mem_emb)
                except Exception:
                    pass

            if score <= 0:
                ql = query.lower()
                score = 0.01 * sum(1 for w in ql.split() if w in content.lower())

            if score > 0.001:
                results.append({
                    "id": mem_id, "memory_type": mtype, "content": content[:1000],
                    "importance": importance,
                    "metadata": json.loads(meta_str) if meta_str else {},
                    "created_at": created_at, "score": round(score, 4),
                })

        results.sort(key=lambda x: (x["score"], x["importance"]), reverse=True)
        return results[:limit]

    def consolidate(self, from_type: str = "working", to_type: str = "episodic",
                    importance_threshold: float = 0.7) -> int:
        """Consolidate important working memories into episodic (hello-agents pattern)."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.execute("UPDATE memories SET memory_type=? WHERE memory_type=? AND importance>=?",
                             (to_type, from_type, importance_threshold))
            conn.commit(); conn.close()
            return c.rowcount

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1,
               max_age_days: int = 30) -> int:
        """Forget memories by strategy (hello-agents pattern)."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            if strategy == "importance_based":
                c = conn.execute("DELETE FROM memories WHERE importance < ?", (threshold,))
            else:
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
                c = conn.execute("DELETE FROM memories WHERE created_at < ?", (cutoff,))
            conn.commit(); conn.close()
            return c.rowcount

    def stats(self) -> str:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            conn.close()
        return f"{total} total memories"

    def clear(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM memories"); conn.commit(); conn.close()


# ================================================================
# KnowledgeManager (orchestrator with HyDE + MQE + Rerank)
# ================================================================

class KnowledgeManager:
    """RAG pipeline with semantic search, HyDE, MQE, Rerank, and memory."""

    def __init__(self, settings: Settings, llm=None):
        self.settings = settings
        self.llm = llm  # HelloAgentsLLM instance for HyDE/MQE/Rerank
        os.makedirs(settings.knowledge_base_path, exist_ok=True)
        os.makedirs("./data", exist_ok=True)

        self._index = VectorIndex()
        self._memory = MemoryStore()
        self._documents: dict[str, dict] = {}
        self._doc_chunks: dict[str, list[str]] = {}

        logger.info("KnowledgeManager initialized (embedding will load on first request)")

    # ================================================================
    # Document Ingestion
    # ================================================================

    def ingest_text(self, text: str, document_id: str, title: str = None,
                    chunk_strategy: str = "recursive") -> dict:
        """Ingest raw text with embedding-based indexing."""
        start = time.perf_counter()
        logger.info(f"[INGEST] Start | doc={document_id} | len={len(text)} | strategy={chunk_strategy}")
        try:
            chunks = chunk_text(text, self.settings.rag_chunk_size,
                               self.settings.rag_chunk_overlap, strategy=chunk_strategy)
            logger.info(f"[INGEST] Chunked into {len(chunks)} chunks")
            self._index.add_document(document_id, chunks, {
                "title": title or document_id, "source": "text",
                "chunk_strategy": chunk_strategy,
            })
            self._doc_chunks[document_id] = chunks
            elapsed = int((time.perf_counter() - start) * 1000)
            self._documents[document_id] = {
                "document_id": document_id, "title": title or document_id,
                "char_count": len(text), "chunk_count": len(chunks),
                "chunk_strategy": chunk_strategy,
                "ingested_at": datetime.now().isoformat(), "ingestion_time_ms": elapsed,
            }
            self._memory.add(
                content=f"Knowledge: {title or document_id}\n{text[:500]}...",
                memory_type="semantic", importance=0.85, concept=title or document_id, document_id=document_id,
            )
            return {"success": True, "message": f"Ingested ({len(chunks)} chunks, {elapsed}ms)",
                    "document_id": document_id, "char_count": len(text), "chunk_count": len(chunks),
                    "ingestion_time_ms": elapsed}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def ingest_document(self, file_path: str, document_id: str = None,
                        chunk_size: int = None, chunk_overlap: int = None,
                        chunk_strategy: str = "recursive") -> dict:
        """Ingest a file with embedding-based indexing."""
        if not os.path.exists(file_path):
            return {"success": False, "message": f"File not found: {file_path}"}
        doc_id = document_id or os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            return self.ingest_text(text, doc_id, os.path.basename(file_path),
                                   chunk_strategy=chunk_strategy)
        except Exception as e:
            return {"success": False, "message": str(e)}

    # ================================================================
    # HyDE — Hypothetical Document Embeddings
    # ================================================================

    async def _generate_hyde(self, query: str) -> Optional[str]:
        """Let LLM write a hypothetical answer, used as the search query."""
        if not self.llm:
            return None
        try:
            prompt = HYDE_PROMPT.format(query=query)
            messages = [{"role": "user", "content": prompt}]
            hyde_text = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.3)
            if hyde_text and len(hyde_text) > 10:
                logger.info(f"HyDE generated: {len(hyde_text)} chars")
                return hyde_text.strip()
        except Exception as e:
            logger.warning(f"HyDE failed: {e}")
        return None

    # ================================================================
    # MQE — Multi-Query Expansion
    # ================================================================

    async def _generate_mqe(self, query: str, n: int = 3) -> list[str]:
        """Let LLM generate N query variants for broader search coverage."""
        if not self.llm:
            return [query]
        try:
            prompt = MQE_PROMPT.format(query=query, n=n)
            messages = [{"role": "user", "content": prompt}]
            text = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.3)
            lines = [ln.strip("-• \t") for ln in (text or "").splitlines() if ln.strip()]
            queries = [ln for ln in lines if len(ln) > 2][:n]
            if queries:
                logger.info(f"MQE generated {len(queries)} expansions")
                return queries
        except Exception as e:
            logger.warning(f"MQE failed: {e}")
        return [query]

    # ================================================================
    # Rerank — LLM-based relevance scoring
    # ================================================================

    async def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Use LLM to re-score retrieved chunks for better relevance ordering."""
        if not self.llm or len(chunks) <= 1:
            return chunks

        try:
            # Format chunks for LLM
            chunk_texts = "\n\n".join([
                f"[{i}] {c['chunk'][:300]}" for i, c in enumerate(chunks[:10])
            ])
            prompt = RERANK_PROMPT.format(query=query, chunks=chunk_texts)
            messages = [{"role": "user", "content": prompt}]
            response = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.1)

            # Parse scores — handle various LLM output formats
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                score_map = {}
                for s in scores:
                    if not isinstance(s, dict): continue
                    # Handle different key names from LLM
                    idx = s.get("idx") or s.get("index") or s.get("id") or s.get("i", 0)
                    score_val = s.get("score") or s.get("relevance") or s.get("rating") or 5
                    try: idx = int(idx)
                    except: continue
                    score_map[idx] = float(score_val) / 10.0

                # Blend vector score (0.4) + LLM score (0.6)
                for i, c in enumerate(chunks):
                    llm_score = score_map.get(i, 0.5)
                    c["score"] = round(c["score"] * 0.4 + llm_score * 0.6, 4)
                    c["reranked"] = True

                chunks.sort(key=lambda x: x["score"], reverse=True)
                logger.info(f"Reranked {len(score_map)} chunks")
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")

        return chunks

    # ================================================================
    # Advanced Search (HyDE + MQE + Rerank)
    # ================================================================

    async def search_advanced(
        self, query: str, limit: int = 5,
        enable_hyde: bool = True, enable_mqe: bool = True, enable_rerank: bool = True,
    ) -> list[dict]:
        """
        Advanced semantic search combining HyDE + MQE + Rerank.

        Flow:
        1. Generate MQE expansions (N query variants)
        2. Optionally generate HyDE (hypothetical answer)
        3. Embed all queries and search
        4. Merge results by max-score dedup
        5. Rerank with LLM
        """
        if not self._index._chunks:
            return []

        # Collect all search queries
        queries = [query]

        if enable_mqe and self.llm:
            mqe_queries = await self._generate_mqe(query, n=2)
            queries.extend(mqe_queries)

        if enable_hyde and self.llm:
            hyde_text = await self._generate_hyde(query)
            if hyde_text:
                queries.append(hyde_text)

        if len(queries) == 1:
            # Simple search
            q_emb = embed_query(query)
            results = self._index.search_hybrid(query, q_emb, limit=limit * 2 if enable_rerank else limit)
        else:
            # Multi-query search with dedup
            all_results: dict[str, dict] = {}  # doc_id+chunk_idx -> result
            slots = max(limit * 4 // len(queries), 3)

            for q in queries:
                q_emb = embed_query(q)
                batch = self._index.search_vector(q_emb, limit=slots)
                for r in batch:
                    key = f"{r['document_id']}_{r['chunk_idx']}"
                    if key not in all_results or r["score"] > all_results[key]["score"]:
                        all_results[key] = r

            results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
            results = results[:limit * 2 if enable_rerank else limit]
            logger.info(f"Advanced search: {len(queries)} queries, {len(results)} unique results")

        # Rerank
        if enable_rerank and self.llm and len(results) > 1:
            results = await self._rerank(query, results)

        return results[:limit]

    def search_simple(self, query: str, limit: int = 5) -> list[dict]:
        """Pure vector search, no LLM calls."""
        q_emb = embed_query(query)
        return self._index.search_vector(q_emb, limit=limit)

    def search_multi_query(self, queries: list[str], limit: int = 5) -> list[dict]:
        """
        Search with multiple query variants independently, merge by max-score dedup.

        Each query gets its own vector search. Results are merged with
        max-score dedup (same chunk found by multiple queries keeps highest score).
        Ideal for compound questions decomposed into sub-queries.
        No LLM calls — embedding only.
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.search_simple(queries[0], limit=limit)

        # Search each query independently
        all_results: dict[str, dict] = {}  # key = "doc_id_chunk_idx" -> result
        slots = max(limit * 2 // len(queries), 3)  # per-query limit

        for q in queries:
            q_emb = embed_query(q)
            batch = self._index.search_vector(q_emb, limit=slots)
            for r in batch:
                key = f"{r['document_id']}_{r['chunk_idx']}"
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # Sort by score descending, take top-k
        merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        logger.info(
            f"Multi-query search: {len(queries)} queries, "
            f"{len(all_results)} unique chunks, returning top {limit}"
        )
        return merged[:limit]

    # ================================================================
    # Knowledge Retrieval (public API)
    # ================================================================

    def search_knowledge(self, query: str, limit: int = 5) -> str:
        """Sync wrapper: simple semantic search."""
        results = self.search_simple(query, limit)
        if not results:
            return "No relevant documents found."
        return self._format_results(query, results)

    async def ask_knowledge(
        self, question: str, limit: int = 5,
        enable_advanced: bool = True, include_citations: bool = True,
    ) -> str:
        """
        Retrieve knowledge for a question.

        Two modes:
        - Simple (enable_advanced=False): Pure vector search, fast, no LLM calls.
        - Advanced (enable_advanced=True): HyDE + MQE + Rerank, slower but better
          for complex or poorly-matched queries.

        The caller (FactualHandler) decides which mode to use. The pattern is:
        1. Try simple search first
        2. If answer is short/empty, retry with advanced
        """
        if enable_advanced and self.llm:
            results = await self.search_advanced(question, limit=limit)
        else:
            results = self.search_simple(question, limit=limit)

        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            doc_title = r.get("metadata", {}).get("title", r["document_id"])
            parts.append(f"【来源{i}: {doc_title}】\n{r['chunk']}")

        if include_citations:
            parts.append(f"\n---\n检索到 {len(results)} 个相关片段")

        return "\n\n".join(parts)

    async def ask_knowledge_multi(
        self, sub_questions: list[str], limit: int = 5,
        include_citations: bool = True,
    ) -> str:
        """
        Retrieve knowledge for a compound question by searching each
        sub-question independently, then merging results.

        No LLM calls — only embedding + vector search per sub-question.
        Results are deduplicated by document_id + chunk_idx (max-score).
        """
        results = self.search_multi_query(sub_questions, limit=limit)

        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            doc_title = r.get("metadata", {}).get("title", r["document_id"])
            parts.append(f"【来源{i}: {doc_title}】\n{r['chunk']}")

        if include_citations:
            parts.append(f"\n---\n检索到 {len(results)} 个相关片段（来自 {len(sub_questions)} 个子查询）")

        return "\n\n".join(parts)

    def has_knowledge(self) -> bool:
        return len(self._documents) > 0 or len(self._index._documents) > 0

    # ================================================================
    # Memory Operations (follows hello-agents MemoryTool pattern)
    # ================================================================

    def record_qa(self, question: str, answer: str, category: str = "",
                  user_id: str = "anonymous") -> None:
        """
        Record a Q&A exchange — follows hello-agents auto_record_conversation.

        Working memory: individual user/assistant messages
        Episodic memory: FULL conversation text (persistent, searchable)
        """
        # Working memory: current session context (temporary)
        self._memory.add(
            content=f"用户: {question}",
            memory_type="working", importance=0.6, user_id=user_id,
            role="user",
        )
        self._memory.add(
            content=f"助手: {answer[:1000]}",
            memory_type="working", importance=0.7, user_id=user_id,
            role="assistant",
        )

        # Episodic memory: FULL conversation record (persistent)
        # This is the key — 原文对话存到持久化记忆，后续可以向量搜索召回
        interaction = f"用户问: {question}\nAI回答: {answer[:2000]}"
        self._memory.add(
            content=interaction,
            memory_type="episodic", importance=0.8, user_id=user_id,
            event_type="qa_interaction", category=category,
            question=question[:200],
        )

    def recall(self, query: str, limit: int = 5, user_id: str = "anonymous") -> str:
        """
        Search episodic memory for relevant past conversations.
        Uses embedding-based semantic search (like hello-agents retrieve_memories).
        """
        results = self._memory.search(query, limit=limit, memory_type="episodic", user_id=user_id)
        if not results:
            return ""

        lines = [f"找到 {len(results)} 条相关记忆:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r['score']:.2f}] {r['content'][:300]}")
        return "\n".join(lines)

    def get_memory_context(self, query: str, limit: int = 3, user_id: str = "anonymous") -> str:
        """
        Get relevant memory context for a query.
        Searches episodic memory with embedding similarity (like hello-agents get_context_for_query).
        """
        results = self._memory.search(query, limit=limit, memory_type="episodic", user_id=user_id)
        if not results:
            return ""
        return "\n---\n".join(r["content"][:500] for r in results[:limit])

    def search_conversation_history(self, query: str, messages: list[dict],
                                     limit: int = 5) -> str:
        """
        Semantic + time-weighted search over conversation history.
        Uses batch embedding — one model call for all messages.
        """
        if not messages:
            return ""

        # Filter and prepare messages
        valid = [(i, m) for i, m in enumerate(reversed(messages)) if m.get("content")]
        if not valid:
            return ""

        contents = [f"{m['role']}: {m['content']}" for _, m in valid]

        try:
            # Batch embed: query + all messages in ONE call
            all_embs = embed_texts([query] + contents)
            q_emb = all_embs[0]
            msg_embs = all_embs[1:]
        except Exception:
            return ""

        scored = []
        for (i, msg), c_emb in zip(valid, msg_embs):
            sim = cosine_similarity(q_emb, c_emb)
            time_weight = 1.0 - (i / max(len(messages), 1)) * 0.5
            score = sim * 0.6 + time_weight * 0.4
            if score > 0.2:
                scored.append({
                    "content": contents[len(scored)][:500],
                    "score": round(score, 3),
                    "timestamp": msg.get("timestamp", ""),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        if not scored:
            return ""

        lines = ["以下是与用户问题相关的历史对话记录:\n"]
        for i, s in enumerate(scored[:limit], 1):
            lines.append(f"{i}. [相关度={s['score']:.2f}] {s['content']}")
        return "\n".join(lines)

    def consolidate_memories(self) -> str:
        n = self._memory.consolidate("working", "episodic", 0.7)
        return f"Consolidated {n} memories"

    def forget_stale_memories(self, max_age_days: int = 30) -> str:
        n = self._memory.forget(strategy="time_based", max_age_days=max_age_days)
        return f"Forgot {n} stale memories"

    # ================================================================
    # Management
    # ================================================================

    def list_documents(self) -> list[dict]:
        """List documents from both in-memory cache and persisted index."""
        # Merge with VectorIndex documents (survives restart)
        for doc_id in self._index._documents:
            if doc_id not in self._documents:
                self._documents[doc_id] = {
                    "document_id": doc_id,
                    "title": self._index._documents[doc_id].get("title", doc_id),
                    "chunk_count": len([c for c in self._index._chunks if c["doc_id"] == doc_id]),
                    "ingested_at": "",
                }
        return list(self._documents.values())

    def get_document(self, document_id: str) -> Optional[dict]:
        return self._documents.get(document_id)

    def get_chunks(self, document_id: str) -> Optional[list[str]]:
        """Get all text chunks for a document (from memory or index)."""
        if document_id in self._doc_chunks:
            return self._doc_chunks[document_id]
        # Fallback: rebuild from persisted index
        chunks = [c["text"] for c in self._index._chunks if c["doc_id"] == document_id]
        return chunks if chunks else None

    def delete_document(self, document_id: str) -> dict:
        if document_id not in self._documents:
            return {"success": False, "message": f"Document '{document_id}' not found"}
        self._index.remove_document(document_id)
        self._doc_chunks.pop(document_id, None)
        del self._documents[document_id]
        self._memory.add(content=f"Deleted document: {document_id}", memory_type="episodic", importance=0.5)
        return {"success": True}

    def get_stats(self) -> dict:
        return {
            "documents_ingested": len(self._documents),
            "document_ids": list(self._documents.keys()),
            "index": self._index.stats(),
            "memory": self._memory.stats(),
        }

    def clear_all(self) -> dict:
        self._index.clear(); self._memory.clear()
        self._documents.clear(); self._doc_chunks.clear()
        return {"success": True}

    @staticmethod
    def _format_results(query: str, results: list[dict]) -> str:
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r['document_id']}] (score: {r['score']:.4f})")
            lines.append(f"   {r['chunk'][:300]}"); lines.append("")
        return "\n".join(lines)
