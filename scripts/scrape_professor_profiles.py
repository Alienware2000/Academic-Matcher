import json
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm


# Path to your JSON file
file_path = "data/research_areas.json"

# Open and load it
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Now `data` is a Python object (usually a list or dict)
print(f"Loaded {len(data)} research_areas from {file_path}")

print(json.dumps(data[0], indent=2))

