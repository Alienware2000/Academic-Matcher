import os
import json
import numpy as np
from typing import List, Dict, Tuple

def load_embeddings(emb_path: str) -> np.ndarray:
    """
    Load the (N x D) embedding matrix from a .npy file.
    Returns a float32 numpy array.
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    emb = np.load(emb_path)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    return emb

def load_metadata_jsonl(meta_path: str) -> List[Dict]:
    """
    Load metadata aligned with the embedding rows.
    JSON Lines format: one JSON object per line.
    Returns a list of dicts, length N, where N matches embedding rows.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    records: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def sanity_check_alignment(emb: np.ndarray, metas: List[Dict]) -> None:
    """
    Ensures len(metas) == emb.shape[0]. Raises AssertionError otherwise.
    This catches accidental misalignment early.
    """
    n_vecs = emb.shape[0]
    n_meta = len(metas)
    assert n_vecs == n_meta, f"Mismatch: {n_vecs} embeddings vs {n_meta} metadata rows."