# scripts/search_faiss_cli.py
import os
import re
import json
import sys
import shutil
import numpy as np
import faiss
import textwrap
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/faiss_index.ivf"
IDMAP_PATH = "data/index/id_map.npy"
META_PATH  = "data/embeddings/professor_metadata.jsonl"

# ---- Simple stopword list to avoid highlighting trivial words ----
STOPWORDS = {
    "and","or","the","a","an","of","in","to","for","with","on","at","by","from",
    "about","into","over","after","before","between","across","through","during",
    "without","within","along","around","since","until","than","as","is","are","be"
}

# ---- ANSI formatting (set to False to disable all colors/bold) ----
USE_COLORS = True and sys.stdout.isatty()

def color(text: str, style: str = "bold") -> str:
    """Minimal formatting: bold or yellow.
    - style='bold'   -> bright/bold
    - style='yellow' -> yellow text
    """
    if not USE_COLORS:
        return text
    if style == "bold":
        return f"\033[1m{text}\033[0m"
    if style == "yellow":
        return f"\033[93m{text}\033[0m"
    return text

def get_term_width(default: int = 100) -> int:
    """Determine terminal width for pretty wrapping."""
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def wrap_paragraphs(text: str, width: int) -> str:
    """Wrap paragraphs separated by blank lines."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    wrapped = []
    for p in paras:
        wrapped.append(textwrap.fill(p, width=width))
    return "\n\n".join(wrapped)

def find_bullets(persp: str) -> list[str]:
    """Extract bullet-like lines starting with '-'/'•'/'–' etc."""
    bullets = []
    for line in persp.splitlines():
        s = line.strip()
        if re.match(r"^[-–•]\s+", s):
            bullets.append(s)
    return bullets

def first_sentences(text: str, max_sentences: int = 2) -> str:
    """Return the first 1–2 sentences (very simple split)."""
    # Split on period/question/exclamation + whitespace
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    return " ".join(parts[:max_sentences]) if parts else text.strip()

def summarize_perspectives(persp: str, max_chars: int = 350, max_bullets: int = 3) -> str:
    """Summarize perspectives:
       - If bullets exist, show the first few as bullets.
       - Else show the first 1–2 sentences.
    """
    if not persp:
        return "—"

    bullets = find_bullets(persp)
    if bullets:
        bullets = bullets[:max_bullets]
        text = "\n".join(bullets)
    else:
        text = first_sentences(persp, max_sentences=2)

    # Truncate long text but avoid cutting mid-word
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text

def highlight_terms(text: str, terms: set[str]) -> str:
    """Highlight occurrences of terms (>=4 letters) in text."""
    if not USE_COLORS or not terms:
        return text

    # Build one regex that matches any term, case-insensitive
    # Escape terms to avoid regex issues, sort longest-first to avoid partial overshadow
    safe_terms = sorted({t for t in terms if len(t) >= 4}, key=len, reverse=True)
    if not safe_terms:
        return text
    pattern = r"(" + "|".join(re.escape(t) for t in safe_terms) + r")"

    def repl(m):
        return color(m.group(0), "yellow")

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def tokenize_query(q: str) -> set[str]:
    """Lowercase, extract alphabetic tokens, drop stopwords."""
    toks = re.findall(r"[a-zA-Z]{3,}", q.lower())
    return {t for t in toks if t not in STOPWORDS}

# --------------------- IO helpers ---------------------

def load_index(index_path: str) -> faiss.Index:
    assert os.path.exists(index_path), f"Missing index at {index_path}"
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

# --------------------- Main CLI -----------------------

if __name__ == "__main__":
    # 1) Load artifacts
    index = load_index(INDEX_PATH)
    id_map = load_id_map(IDMAP_PATH)
    metas = load_metadata(META_PATH)
    print(f"[load] index.ntotal={index.ntotal}, id_map={len(id_map)}, metas={len(metas)}")

    # 2) Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # 3) Query input (optionally support k)
    try:
        k = int(input("How many results (k)? [default 5]: ").strip() or "5")
        k = max(1, min(k, 20))
    except ValueError:
        k = 5

    query = input("\nEnter a research interest: ").strip()
    if not query:
        print("Empty query. Exit."); sys.exit(0)

    # 4) Encode + normalize query
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # 5) Search
    scores, idxs = index.search(q_vec, k)

    # 6) Display nicely
    width = get_term_width(100)
    q_terms = tokenize_query(query)

    print(f"\nTop {k} results for: {color(query, 'bold')}\n")
    for rank, (score, faiss_id) in enumerate(zip(scores[0], idxs[0]), start=1):
        meta = metas[faiss_id]
        name  = meta.get("name", "Unknown")
        title = meta.get("title", "")
        url   = meta.get("url", "")
        areas = ", ".join(meta.get("areas", []))
        email = meta.get("email") or "—"
        web   = meta.get("website") or "—"

        # Prepare perspectives summary
        persp = meta.get("perspectives") or ""
        persp_summary = summarize_perspectives(persp, max_chars=350, max_bullets=3)

        # Build “where matched” highlighting
        areas_h = highlight_terms(areas, q_terms)
        persp_h = highlight_terms(persp_summary, q_terms)

        # Header line (rank, score, name)
        header = f"{rank:>2}. {color(name, 'bold')}  |  score={score: .3f}"
        print(header)
        print(textwrap.fill(title, width=width))
        print(f"Areas: {textwrap.fill(areas_h, width=width)}")

        print("Perspectives:")
        print(wrap_paragraphs(persp_h, width=width))

        print(f"Email: {email}")
        print(f"Site:  {web}")
        print(f"URL:   {url}")
        print("-" * width)
