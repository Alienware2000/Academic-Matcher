import os       
import json     
from typing import List, Dict, Tuple  


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


# --------- Lesson A main ---------
if __name__ == "__main__":
    # 1) Load profiles
    profiles = load_profiles(PROFILES_PATH)
    print(f"[load] profiles: {len(profiles)} from {PROFILES_PATH}")

    # 2) Build corpus text + metadata
    documents, metas = build_corpus(profiles)
    print(f"[build] documents: {len(documents)} (non-empty)")

    # 3) Peek at some examples to understand what we created
    for i in range(3):
        print(f"\n--- DOC {i} ---")
        print(documents[i][:400])   # first 400 chars
        print("--- META ---")
        print({k: metas[i][k] for k in ["name", "areas", "url"]})
