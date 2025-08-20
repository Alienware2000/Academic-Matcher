# app/streamlit_app.py
# --------------------
# Minimal Streamlit UI for Academic Matcher MVP.
# Loads precomputed embeddings + metadata, encodes a user query with the SAME model,
# computes cosine similarity, and shows the top matches with links & snippets.

import os
import json
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


# --------- Config: paths + model -----------
EMB_PATH = "data/embeddings/professor_embeddings.npy"
META_PATH = "data/embeddings/professor_metadata.jsonl"

# IMPORTANT: This must match the model used to build the embeddings!
# If your embeddings were built with 'all-MiniLM-L6-v2', keep this as is.
# If you later rebuild with E5, update this to "intfloat/e5-base-v2".
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------- Small utility functions ----------
def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row so cosine similarity becomes a dot product."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def load_jsonl(path: str) -> List[Dict]:
    """Load one JSON object per line, returning a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# --------- Caching: prevent reloading on every rerun ----------
@st.cache_resource(show_spinner=False)
def load_model(name: str) -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def load_corpus(emb_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict]]:
    """Load embeddings (N x D) and aligned metadata list."""
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    emb = np.load(emb_path).astype(np.float32)
    metas = load_jsonl(meta_path)

    if emb.shape[0] != len(metas):
        raise ValueError(f"Embeddings rows ({emb.shape[0]}) != metadata rows ({len(metas)})")

    return emb, metas


def search_top_k(query: str, model: SentenceTransformer, emb: np.ndarray, metas: List[Dict], k: int = 5):
    """
    Encode the query, L2-normalize it, compute cosine similarity with all docs,
    and return (indices, scores) of the top-k most similar docs.
    """
    # Encode the query (must match how corpus was embedded)
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)  # (1, D)

    # Embeddings we loaded should already be normalized in your pipeline.
    # But if you're not sure, uncomment the next line to normalize emb here:
    # emb = l2_normalize(emb)

    # Cosine similarity = dot product if both are normalized
    sims = (q_vec @ emb.T).ravel()  # shape (N,)

    # Top-k indices by similarity
    k = min(k, len(sims))
    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx_sorted = idx[np.argsort(-sims[idx])]
    scores = sims[idx_sorted]

    return idx_sorted, scores


# --------- Streamlit UI ----------
st.set_page_config(page_title="Academic Matcher", page_icon="🎓", layout="wide")

st.title("🎓 Academic Matcher — MVP")
st.caption("Find professors whose research aligns with your interests using semantic search.")

# Sidebar: data/model info
with st.sidebar:
    st.header("Configuration")
    st.write(f"**Embeddings file:** `{EMB_PATH}`")
    st.write(f"**Metadata file:** `{META_PATH}`")
    st.write(f"**Model:** `{MODEL_NAME}`")
    k = st.slider("Top K results", min_value=3, max_value=15, value=5, step=1)

# Load resources once
try:
    model = load_model(MODEL_NAME)
    emb, metas = load_corpus(EMB_PATH, META_PATH)
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Main input
query = st.text_input("Enter your research interest (e.g., 'computational biology', 'robotics and HRI')", "")

if st.button("Search") or query.strip():
    q = query.strip()
    if not q:
        st.warning("Please type a query.")
    else:
        with st.spinner("Searching..."):
            idxs, scores = search_top_k(q, model, emb, metas, k=k)

        st.markdown(f"### Top {len(idxs)} results for: `{q}`")
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            m = metas[int(i)]
            name = m.get("name", "Unknown")
            title = m.get("title", "")
            areas = ", ".join(m.get("areas", []))
            url = m.get("url", "")
            email = m.get("email", "")
            website = m.get("website", "")

            perspectives = m.get("perspectives", "")
            if perspectives and len(perspectives) > 400:
                perspectives = perspectives[:400] + "..."

            # Card-like layout
            st.markdown(f"**{rank}. {name}**  |  _score: {s:.3f}_")
            if title:
                st.write(title)
            if areas:
                st.write(f"**Areas**: {areas}")
            if perspectives:
                st.write(f"**Perspectives**: {perspectives}")

            # Links in a tidy row
            cols = st.columns(3)
            with cols[0]:
                if email:
                    st.write(f"📧 [{email}](mailto:{email})")
            with cols[1]:
                if website:
                    st.write(f"🌐 [Website]({website})")
            with cols[2]:
                if url:
                    st.write(f"🔗 [Profile]({url})")

            st.divider()
