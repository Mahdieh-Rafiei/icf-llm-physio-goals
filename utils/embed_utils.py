"""
Embedding utilities using Ollama's /api/embed endpoint.

Model:    qwen3-embedding:8b  (configurable via EMBED_MODEL in .env)
          — multilingual, 4096-dim, 40K context, 100+ languages.

Uses asymmetric retrieval:
  Queries (concepts being looked up): prefixed with the task instruction
  Passages (stored index texts):      plain text

Credentials are loaded from .env (OLLAMA_API_URL, COSYBIO_API_KEY, EMBED_MODEL).
IMPORTANT: .env always overrides shell environment variables (override=True).
           If you change EMBED_MODEL, delete output/*.pkl and rebuild the index.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import urllib3
import numpy as np
import requests
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# Use EMBED_API_URL if set (allows embeddings and LLM on different servers)
OLLAMA_API_URL = os.getenv("EMBED_API_URL", os.environ["OLLAMA_API_URL"]).rstrip("/")
API_KEY        = os.getenv("EMBED_API_KEY", os.environ["COSYBIO_API_KEY"])
EMBED_MODEL    = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")
OUTPUT_DIR     = Path(__file__).parent.parent.parent / "output"

# Arctic Embed 2 query instruction for asymmetric retrieval
_CONCEPT_QUERY_INSTRUCTION = (
    "Represent this German rehabilitation concept for searching relevant items: "
)
_CODE_QUERY_INSTRUCTION = (
    "Represent this German rehabilitation concept for searching ICF code descriptions: "
)

BATCH_SIZE = 32  # texts per API call


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _embed_batch(texts: list[str], retries: int = 5) -> np.ndarray:
    """Call /api/embed for a single batch with retry on transient errors."""
    import time
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_API_URL}/api/embed",
                headers=_headers(),
                json={"model": EMBED_MODEL, "input": texts},
                timeout=120,
                verify=False,
            )
            resp.raise_for_status()
            vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return vecs / norms
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"\n  Embed batch error ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)


def embed_texts(
    texts: list[str],
    as_query: bool = False,
    instruction: str = "",
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts. Returns L2-normalised float32 array of shape (N, dim).

    as_query=True  → prepend instruction prefix (for concept queries)
    as_query=False → plain text (for stored index passages)
    """
    if as_query and instruction:
        texts = [instruction + t for t in texts]

    all_vecs = []
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    iterator = tqdm(batches, desc="Embedding") if show_progress else batches

    for batch in iterator:
        all_vecs.append(_embed_batch(batch))

    return np.vstack(all_vecs)


# ─── Index classes ────────────────────────────────────────────────────────────

class ConceptIndex:
    """Cosine similarity index over training concepts."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        assert len(embeddings) == len(metadata)
        self.embeddings = embeddings
        self.metadata = metadata

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 8,
        exclude_konzept: Optional[str] = None,
    ) -> list[dict]:
        sims = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        ranked_idx = np.argsort(sims)[::-1]
        results = []
        for idx in ranked_idx:
            item = self.metadata[idx].copy()
            item["similarity"] = float(sims[idx])
            if exclude_konzept and item["konzept"] == exclude_konzept:
                continue
            results.append(item)
            if len(results) == top_k:
                break
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: Path) -> "ConceptIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["embeddings"], data["metadata"])


class ICFCodeIndex:
    """Cosine similarity index over ICF code descriptions."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        self.embeddings = embeddings
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[dict]:
        sims = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            item = self.metadata[idx].copy()
            item["similarity"] = float(sims[idx])
            results.append(item)
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: Path) -> "ICFCodeIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["embeddings"], data["metadata"])


# ─── Index builders ──────────────────────────────────────────────────────────

def build_concept_index(train_df) -> ConceptIndex:
    konzepte = train_df["konzept"].tolist()
    print(f"Embedding {len(konzepte)} training concepts (passages)...")
    embeddings = embed_texts(konzepte, as_query=False)

    metadata = [
        {
            "konzept": row["konzept"],
            "gold_codes": list(row["gold_codes"]),
            "cohort": row["cohort"],
        }
        for _, row in train_df.iterrows()
    ]
    return ConceptIndex(embeddings, metadata)


def build_icf_code_index(icf_df) -> ICFCodeIndex:
    texts = [
        f"{row['code']}: {row['title_de']}"
        for _, row in icf_df.iterrows()
    ]
    print(f"Embedding {len(texts)} ICF code descriptions (passages)...")
    embeddings = embed_texts(texts, as_query=False)

    metadata = [
        {"code": row["code"], "title_de": row["title_de"]}
        for _, row in icf_df.iterrows()
    ]
    return ICFCodeIndex(embeddings, metadata)


def query_concept_index(
    konzept: str,
    concept_index: ConceptIndex,
    top_k: int = 8,
) -> list[dict]:
    q_emb = embed_texts(
        [konzept],
        as_query=True,
        instruction=_CONCEPT_QUERY_INSTRUCTION,
        show_progress=False,
    )[0]
    return concept_index.search(q_emb, top_k=top_k, exclude_konzept=konzept)


def query_icf_code_index(
    konzept: str,
    code_index: ICFCodeIndex,
    top_k: int = 20,
) -> list[dict]:
    q_emb = embed_texts(
        [konzept],
        as_query=True,
        instruction=_CODE_QUERY_INSTRUCTION,
        show_progress=False,
    )[0]
    return code_index.search(q_emb, top_k=top_k)
