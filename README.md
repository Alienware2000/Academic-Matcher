# Academic Matcher 🎓🔍

A retrieval-augmented tool to help students find research labs that align with their interests, using embeddings and large language models (LLMs).

## 🧠 How It Works

1. Scrape lab descriptions from university sites
2. Embed the text using a transformer model
3. Perform semantic search on user queries
4. Show best-matching labs + LLM-based explanations (optional)

## 🔧 Tech Stack

- Python
- Hugging Face Transformers
- Sentence Transformers
- Streamlit
- OpenAI / LLM APIs
- Git + GitHub

## 📁 Project Structure

academic-matcher/
│
├── data/                         # Store raw + cleaned data (JSON/CSV files)
│   ├── scraped_labs.json         # Output of scraping: lab/course info
│   └── embeddings.npy            # Numpy array of vectorized lab descriptions
│
├── scripts/                      # Python scripts that do scraping, preprocessing, etc.
│   ├── scrape_labs.py            # Script to scrape lab/course info
│   ├── clean_data.py             # Script to clean or structure scraped data (optional)
│   └── embed_data.py             # Script to compute sentence embeddings
│
├── match_engine/                 # Core logic that matches user input to data
│   └── matcher.py                # Loads embeddings + computes similarity
│
├── llm_explainer/               # Optional: LLMs to generate natural language explanations
│   └── explainer.py              # Uses an LLM (e.g., OpenAI/HF) to explain a match
│
├── frontend/                     # Streamlit (or web app) user interface
│   └── app.py                    # Main Streamlit app UI
│
├── logs/                         # Track progress of project
│   └── weekend-progress.md       # Main log
│
├── requirements.txt              # List of packages to install
├── README.md                     # Project overview
└── .gitignore                    # Ignore large files, envs, etc.


