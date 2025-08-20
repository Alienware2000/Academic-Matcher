import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/faiss_index.ivf"
IDMAP_PATH = "data/index/id_map.npy"
META_PATH = "data/embeddings/professor_metadata.jsonl"

def load_index(index_path: str) -> faiss.Index:
    assert os.path.exists(index_path), f"missing index at {index_path}"
    return faiss.read_index(index_path)

def load_id_map(idmap_path: str) -> np.ndarray:
    ids = np.load(idmap_path)
    assert ids.ndim == 1, "id_map should be 1D"
    return ids

def load_metadata(meta_path: str):
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

if __name__ == "__main__":
    # 1) Load FAISS artifacts + metadata
    index = load_index(INDEX_PATH)
    id_map = load_id_map(IDMAP_PATH)
    metas = load_metadata(META_PATH)
    print(f"[load] index.ntotal={index.ntotal}, id_map={len(id_map)}, metas={len(metas)}")

    # 2) Load the same embedding model as used for the corpus
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # 3) Get a query from user
    query = input("\nEnter a research interest (e.g., 'robotics and human-robot interaction'): ").strip()
    if not query:
        print("Empty query. Exit.")
        exit(0)

    # 4) Encode + normalize query
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    # FAISS expects shape (n_queries, dim)
    assert q.ndim == 2 and q.shape[0] == 1

    # 5) Search — return inner products (cosine) and indices
    top_k = 5
    scores, idxs = index.search(q, top_k)  # scores shape: (1, k), idxs shape: (1, k)

    print(f"\nTop {top_k} results for: \"{query}\"")
    for rank, (score, faiss_id) in enumerate(zip(scores[0], idxs[0]), start=1):
        # faiss_id is the row index in the original embeddings (since we used default ids)
        meta = metas[faiss_id]  # because row order = metadata line order
        print(f"  {rank:>2}. {meta['name']:30s} | score={score: .3f} | areas={', '.join(meta['areas'])}")
        print(f"      Title: {meta['title']}")
        print(f"      URL:   {meta['url']}\n")
