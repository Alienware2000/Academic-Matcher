import json
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import defaultdict

# Path to your JSON file
file_path = "data/research_areas.json"

# Open and load it
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Now `data` is a Python object (usually a list or dict)
print(f"Loaded {len(data)} research_areas from {file_path}")

# Build a mapping: profile_url -> set(areas)
prof_to_areas = defaultdict(set)

for area in data:  # data = research_areas loaded from JSON
    area_name = area.get("area")
    for prof in area.get("professors", []):
        url = prof.get("profile_url")
        if url:
            prof_to_areas[url].add(area_name)

print(f"Unique professor profile URLs: {len(prof_to_areas)}")


