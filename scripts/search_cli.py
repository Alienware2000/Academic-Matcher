import os
import numpy as np
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer

# Reuse the utilities we wrote:
from utils_io import load_embeddings, load_metadata_jsonl, sanity_check_alignment

# ---- Paths (same as your embed script wrote) ----
EMB_PATH = "data/embeddings/professor_embeddings.npy"
META_PATH = "data/embeddings/professor_metadata.jsonl"

# ---- Cosine top-k (assumes normalized vectors) ----
def top_k_cosine(query_vec: np.ndarray, emb_matrix: np.ndarray, k: int = 5):
    """
    query_vec: (1, D) normalized
    emb_matrix: (N, D) normalized
    Returns list of (index, score) sorted descending.
    """
    sims = (query_vec @ emb_matrix.T).ravel()                 # dot products
    k = min(k, len(sims))
    idx = np.argpartition(-sims, kth=k-1)[:k]                 # approximate top-k
    idx_sorted = idx[np.argsort(-sims[idx])]                  # sort top-k
    return [(int(i), float(sims[i])) for i in idx_sorted]

def format_snippet(text: str, max_chars: int = 220) -> str:
    """
    Return a short snippet from 'perspectives' or other text.
    Avoid printing very long blocks in CLI output.
    """
    if not text:
        return ""
    t = " ".join(text.split())  # clean excessive whitespace/newlines
    return (t[:max_chars] + "…") if len(t) > max_chars else t

def main():
    # 1) Load artifacts
    print("[search] loading artifacts …")
    emb = load_embeddings(EMB_PATH)           # shape (N, D)
    metas = load_metadata_jsonl(META_PATH)    # length N
    sanity_check_alignment(emb, metas)
    print(f"[search] embeddings: {emb.shape}, metadata rows: {len(metas)}")

    # 2) Load the SAME embedding model used for indexing
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[model] loading {model_name} …")
    model = SentenceTransformer(model_name)

    # 3) Interactive loop
    print("\nType a query (or 'q' to quit). Examples:")
    print("  - 'robotics and human-robot interaction'")
    print("  - 'graph neural networks for biology'")
    print("  - 'distributed systems secure consensus'")
    while True:
        query = input("\nquery> ").strip()
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            print("bye!")
            break

        # 4) Encode the query (normalize=True ensures unit-length vector)
        q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        # shape (1, D)

        # 5) Retrieve top-k
        topk = top_k_cosine(q_vec, emb, k=5)

        print(f"\nTop results for: {query!r}")
        for rank, (i, score) in enumerate(topk, start=1):
            m = metas[i]
            name = m.get("name", "Unknown")
            areas = ", ".join(m.get("areas", []))
            snippet = format_snippet(m.get("perspectives", ""))

            print(f"{rank:>2}. {name}  | score={score: .3f}")
            if areas:
                print(f"    Areas: {areas}")
            if snippet:
                print(f"    {snippet}")
            print(f"    Profile: {m.get('url', '')}")

if __name__ == "__main__":
    main()
