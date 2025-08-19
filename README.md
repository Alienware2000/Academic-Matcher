# Academic Matcher 🎓🔍

A retrieval-augmented tool to help students find research labs that align with their interests, using embeddings and large language models (LLMs).

---

## 🧠 How It Works (Planned Pipeline)

1. **Scrape** lab descriptions and faculty profiles from university sites ✅  
2. **Preprocess & Store** structured profile data (JSON) ✅  
3. **Embed** the text using transformer models  
4. **Build Retrieval**: perform semantic search on user queries  
5. **Match & Explain**: return best-matching labs + optional LLM-based summaries  

---

## ✅ Current Progress (as of August 2025)

We have completed **Week 2 (Data Collection & Scraping)**:

- Scraped **16 research areas** from Yale’s CS/EAS site.  
- Extracted **40 unique professor profiles** across those areas.  
- Saved structured data to `data/professor_profiles.json`, including:  
  - `name`, `title`, `email`, `website`  
  - `room_office`, `office_address`  
  - `perspectives` (main research description)  
  - associated `areas`  
- All raw HTML snapshots are saved under `data/raw/professors/` for reproducibility/debugging.  
- Scraper supports **resume functionality**, error handling, and polite delays.  

👉 Out of 40 profiles, **38 had “Perspectives” sections extracted successfully**. The 2 missing ones simply didn’t contain that section (confirmed by manual inspection).

---

## 🔧 Tech Stack

- **Python** for scraping, data handling, and embeddings  
- **BeautifulSoup4** + **Requests** for HTML parsing  
- **tqdm** for progress tracking  
- **Hugging Face Transformers / Sentence Transformers** (planned)  
- **Streamlit** (planned) for simple UI  
- **OpenAI / other LLM APIs** (planned)  
- **Git + GitHub** for version control  

---

## 🚀 Next Steps (Week 3 and beyond)

- **Week 3 (Embeddings & Semantic Search)**  
  - Choose embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`)  
  - Generate embeddings for professor perspectives + areas  
  - Store embeddings in a vector database (likely FAISS or Chroma)  

- **Week 4 (Matching Engine)**  
  - Implement semantic search for user queries → return top-k professors/labs  
  - Add optional LLM explanation layer (e.g., GPT/Claude rephrasing why match makes sense)  

- **Week 5+ (UI + Deployment)**  
  - Build simple Streamlit interface for query + result display  
  - Deploy demo app  

---

## 🗂️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Alienware2000/Academic-Matcher.git
   cd Academic-Matcher


