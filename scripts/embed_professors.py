import os       
import json     
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


# --------- Config (file paths) ----------
PROFILES_PATH = "data/professor_profiles.json"   # input produced by your scraper
EMB_DIR = "data/embeddings"                      # where we'll later save outputs


# --------- Step 1: Load profiles ----------
def load_profiles(path: str) -> List[Dict]:
    """
    Read the JSON file that contains the list of professor profiles.
    Expect: a list of dicts, each with keys like "name", "title", "areas", "perspectives".
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "profiles JSON must be a list"
    return data


# --------- Step 2: Construct document text ----------
def _safe_join(parts: List[str]) -> str:
    """
    Helper: join non-empty strings with spaces, avoiding 'None' strings.
    This keeps our final document clean and human-readable.
    """
    return " ".join([p.strip() for p in parts if p and p.strip()])


def profile_to_document(p: Dict) -> Tuple[str, Dict]:
    """
    Convert ONE professor profile dict -> (document_text, metadata).

    document_text:
      We want a short, natural paragraph that captures the research "signal":
      - name, title
      - areas (keywords like 'robotics', 'NLP', etc.)
      - perspectives (the meat — sentences describing their research)

    metadata:
      Extra info we keep alongside the embedding row for display later.
    """
    name = p.get("name") or ""
    title = p.get("title") or ""
    areas = p.get("areas") or []                 # list of strings
    perspectives = p.get("perspectives") or ""   # long text or bullet points

    areas_text = ", ".join(areas)                # turn list into "A, B, C"

    # Build a succinct, natural-sounding document string.
    # (This is what the embedding model will "read")
    doc = _safe_join([
        f"{name}.",
        f"{title}.",
        f"Research areas: {areas_text}." if areas_text else "",
        f"Research interests: {perspectives}" if perspectives else "",
    ])

    # Metadata aligned with this document (we’ll store one metadata per embedding row)
    meta = {
        "url": p.get("url"),
        "name": name,
        "title": title,
        "areas": areas,
        "perspectives": perspectives,
        "website": p.get("website"),
        "email": p.get("email"),
        "office_address": p.get("office_address"),
        "room_office": p.get("room_office"),
    }
    return doc, meta


def build_corpus(profiles: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Convert ALL profiles -> (documents, metadata_list)
    documents[i] and metadata_list[i] refer to the same professor.
    """
    documents, metas = [], []
    for p in profiles:
        doc, meta = profile_to_document(p)
        if doc.strip():          # skip any empty texts just in case
            documents.append(doc)
            metas.append(meta)
    assert len(documents) == len(metas), "documents and metadata size mismatch"
    return documents, metas

def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize each row so cosine similarity equals dot product.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)  # avoid divide-by-zero
    return mat / norms


def save_metadata_jsonl(metas: List[Dict], path: str) -> None:
    """
    Save metadata (aligned to embeddings) as JSON Lines (one JSON per line).
    JSONL is convenient for streaming and line-by-line loading.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    profiles = load_profiles(PROFILES_PATH)
    documents, metas = build_corpus(profiles)
    print(f"[load/build] {len(documents)} documents")

    # Load model once
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[model] loading {model_name} ...")
    model = SentenceTransformer(model_name)

    # Encode ALL documents (now we use a larger batch size)
    emb = model.encode(
        documents,
        batch_size=64,             # tune based on your machine
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype(np.float32)

    print("[encode] matrix shape:", emb.shape)

    # Normalize (recommended for cosine similarity)
    emb = l2_normalize(emb)
    print("row norms (first 5):", np.linalg.norm(emb[:5], axis=1))

    # Save artifacts
    os.makedirs(EMB_DIR, exist_ok=True)
    emb_path = os.path.join(EMB_DIR, "professor_embeddings.npy")
    meta_path = os.path.join(EMB_DIR, "professor_metadata.jsonl")
    np.save(emb_path, emb)
    save_metadata_jsonl(metas, meta_path)
    print(f"[save] {emb_path} and {meta_path}")
