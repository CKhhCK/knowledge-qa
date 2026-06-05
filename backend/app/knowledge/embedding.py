"""
Embedding module — semantic vector embeddings using Qwen3-Embedding-4B.

Model: D:/computerPro/EmbeddingLLM/qwen3-embedding-4b
- Architecture: Qwen3ForCausalLM (36 layers, 2560 hidden_size)
- Embedding dimension: 2560
- Query prompt: instruction-based (automatically applied via sentence-transformers)
- Similarity: cosine

Fallback: TF-IDF if model fails to load.
"""

import hashlib
import os
import numpy as np
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Model configuration — Qwen3-Embedding-4B (local, 2560d, batch-only inference)
_MODEL_PATH = r"D:\computerPro\EmbeddingLLM\qwen3-embedding-4b"
_embedding_model = None
_embedding_dim = 2560
_embedding_crashed = False  # Switch to TF-IDF after a crash to prevent repeated segfaults


def _load_model():
    """Lazy-load Qwen3-Embedding-4B. Thread-safe with lock."""
    global _embedding_model, _embedding_dim, _embedding_crashed, _model_loading
    if _embedding_model is not None:
        return

    with _model_lock:
        # Double-check after acquiring lock
        if _embedding_model is not None:
            return

        # After a crash, skip Qwen3
        if _embedding_crashed:
            _embedding_model = "tfidf"
            _embedding_dim = 512
            logger.warning("Qwen3 previously crashed, using TF-IDF fallback")
            return

        _model_loading = True

        # Qwen3-Embedding-4B (local, no network)
        if os.path.isdir(_MODEL_PATH):
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading Qwen3-Embedding-4B from {_MODEL_PATH}...")
                _embedding_model = SentenceTransformer(
                    _MODEL_PATH, device="cpu", trust_remote_code=True,
                )
                _embedding_dim = _embedding_model.get_sentence_embedding_dimension()
                _model_ready.set()
                logger.info(f"Qwen3-Embedding-4B loaded ({_embedding_dim}d)")
                return
            except Exception as e:
                logger.error(f"Qwen3-Embedding-4B failed: {e}")

    # TF-IDF fallback (no network, no GPU needed)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _embedding_model = "tfidf"
        _embedding_dim = 512
        logger.warning("Using TF-IDF fallback (512d)")
        return
    except ImportError:
        pass

    _embedding_model = "hash"
    _embedding_dim = 128
    logger.warning("Using hash fallback (128d)")


# Threading: signal when model is loaded, requests can wait for it
import threading
_model_lock = threading.Lock()
_model_ready = threading.Event()
_model_loading = False


def preload_model():
    """Pre-load embedding model in background (called at startup)."""
    logger.info("Pre-loading embedding model in background...")
    _load_model()  # Lock is inside _load_model


def get_embedding_dim() -> int:
    _load_model()
    return _embedding_dim


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    """
    global _embedding_model, _embedding_dim, _embedding_crashed
    # Wait for background model loading (up to 30s)
    if _model_loading and not _model_ready.is_set():
        _model_ready.wait(timeout=30.0)
    _load_model()

    if _embedding_model is None or _embedding_model == "hash":
        return [_hash_embedding(t, _embedding_dim) for t in texts]

    if _embedding_model == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=_embedding_dim)
        try:
            matrix = vectorizer.fit_transform(texts)
            return [matrix[i].toarray()[0].tolist() for i in range(matrix.shape[0])]
        except:
            return [_hash_embedding(t, _embedding_dim) for t in texts]

    # sentence-transformers model (Qwen3 or fallback)
    try:
        if is_query:
            embeddings = _embedding_model.encode(
                texts, prompt_name="query", show_progress_bar=False, batch_size=1,
            )
        else:
            embeddings = _embedding_model.encode(
                texts, show_progress_bar=False, batch_size=1,
            )
        result = embeddings.tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}, switching to TF-IDF fallback")
        _embedding_crashed = True
        _embedding_model = "tfidf"
        _embedding_dim = 512
        return [_hash_embedding(t, _embedding_dim) for t in texts]

    # Cleanup to prevent OOM segfault
    import gc
    gc.collect()

    return result


def embed_query(text: str) -> list[float]:
    """Generate embedding for a search query (with instruction prompt)."""
    return embed_texts([text], is_query=True)[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for documents (no instruction prompt)."""
    return embed_texts(texts, is_query=False)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _hash_embedding(text: str, dim: int) -> list[float]:
    """Fallback: hash-based pseudo-embedding."""
    vec = [0.0] * dim
    for i, word in enumerate(text.lower().split()):
        h = int(hashlib.md5(f"{word}_{i}".encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    total = sum(v * v for v in vec) ** 0.5
    return [v / max(total, 0.001) for v in vec]
