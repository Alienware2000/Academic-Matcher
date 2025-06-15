import requests
from bs4 import BeautifulSoup
import re
import json
import os

def fetch_html(url: str) -> str:
    """Fetch HTML content from the given URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if request failed
    return response.text

def save_raw_html(html: str, output_path: str):
    """Save raw HTML to a local file for reproducibility/debugging."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved raw HTML to {output_path}")

def parse_research_areas(html: str) -> list:
    """Extract research areas and associated faculty from Yale CS page."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("div", class_="side-nav-blocks")
    print(f"Found {len(blocks)} research areas.")

    data = []

    for block in blocks:
        title_tag = block.find("h2")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)

        if re.match(r"^\d{4}\s*[-–]\s*\d{4}$", title):
            continue  # Skip technical report years

        # Extract description paragraphs
        description_paragraphs = []
        for sibling in title_tag.find_next_siblings():
            if sibling.name == "h3" and "Faculty" in sibling.get_text():
                break
            if sibling.name == "p":
                description_paragraphs.append(sibling.get_text(strip=True))

        description = " ".join(description_paragraphs)

        # Extract professors
        professors = []
        faculty_tag = block.find("div", class_="faculty-member-list")
        if faculty_tag:
            for tag in faculty_tag.find_all("a"):
                href = tag.get("href")
                if href.startswith("/"):
                    href = "https://engineering.yale.edu" + href
                p = tag.find("p")
                if p:
                    name = p.find("strong").get_text(strip=True)
                    title_text = p.get_text(strip=True).replace(name, "").strip()
                    professors.append({
                        "name": name,
                        "title": title_text,
                        "profile_url": href
                    })

        data.append({
            "area": title,
            "description": description,
            "professors": professors
        })

    return data

def save_json(data: list, output_path: str):
    """Save extracted data to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} research areas to {output_path}")

# ----------- MAIN PIPELINE -------------------

if __name__ == "__main__":
    url = "https://engineering.yale.edu/academic-study/departments/computer-science/research-areas"
    html = fetch_html(url)

    save_raw_html(html, "data/raw/research_areas.html")
    data = parse_research_areas(html)
    save_json(data, "data/research_areas.json")
