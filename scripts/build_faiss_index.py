import os
import json
import numpy as np
import faiss

EMB_PATH = "data/embeddings/professor_embeddings.npy"   # (N, D) float32
META_PATH = "data/embeddings/professor_metadata.jsonl"  # N lines, aligned to embeddings
INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.ivf")  # just a filename; "ivf" is arbitrary here
IDMAP_PATH = os.path.join(INDEX_DIR, "id_map.npy")

def load_embeddings(path: str) -> np.ndarray:
    emb = np.load(path).astype(np.float32)
    assert emb.ndim == 2 and emb.shape[0] > 0, "embeddings must be 2D with >0 rows"
    return emb

def count_metadata_lines(path: str) -> int:
    """Lightweight check that metadata rows == embedding rows."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

def build_faiss_index(emb: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for inner-product (cosine when vectors are L2-normalized).
    - If emb vectors are L2-normalized (which we did), use IndexFlatIP.
    """
    n, d = emb.shape
    # FAISS expects shape (N, D), contiguous float32
    emb = np.ascontiguousarray(emb, dtype=np.float32)

    # IndexFlatIP = exact search via inner product (cosine if vectors are normalized)
    index = faiss.IndexFlatIP(d)
    index.add(emb)  # add all vectors
    assert index.ntotal == n, "FAISS index size mismatch after add()"
    return index

def save_index(index: faiss.Index, index_path: str) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

def save_id_map(n: int, idmap_path: str) -> None:
    """
    Here our 'ids' are just [0..N-1] — row positions in the embedding matrix.
    Saving it keeps our options open if we ever change how we order/assign ids.
    """
    os.makedirs(os.path.dirname(idmap_path), exist_ok=True)
    ids = np.arange(n, dtype=np.int32)
    np.save(idmap_path, ids)

if __name__ == "__main__":
    # 1) Load artifacts
    emb = load_embeddings(EMB_PATH)
    n, d = emb.shape
    print(f"[embeddings] loaded matrix: {emb.shape}")

    # 2) Sanity: metadata line count should match N
    meta_lines = count_metadata_lines(META_PATH)
    print(f"[metadata] {meta_lines} lines in {META_PATH}")
    assert meta_lines == n, f"metadata lines ({meta_lines}) != embeddings rows ({n})"

    # 3) Build index
    index = build_faiss_index(emb)
    print(f"[faiss] built IndexFlatIP with {index.ntotal} vectors (dim={d})")

    # 4) Save artifacts
    save_index(index, INDEX_PATH)
    save_id_map(n, IDMAP_PATH)
    print(f"[save] index -> {INDEX_PATH}")
    print(f"[save] id_map -> {IDMAP_PATH}")
