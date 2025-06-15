import requests
from bs4 import BeautifulSoup
import re
import json

# -------------------------------
# Step 1: Fetch the web page HTML
# -------------------------------

url = "https://engineering.yale.edu/academic-study/departments/computer-science/research-areas"
response = requests.get(url)
html = response.text  # Get raw HTML as string

# -------------------------------
# Step 2: Parse HTML with BeautifulSoup
# -------------------------------

soup = BeautifulSoup(html, "html.parser")

# -------------------------------
# Step 3: Find all research area blocks
# -------------------------------

blocks = soup.find_all("div", class_="side-nav-blocks")
print(f"Found {len(blocks)} research areas.")

data = []

for block in blocks:
    # -------------------------------
    # Extract research area title
    # -------------------------------
    title_tag = block.find("h2")
    if not title_tag:
        continue

    title = title_tag.get_text(strip=True)

    # Skip "technical report" blocks with date ranges like "2025–2020"
    if re.match(r"^\d{4}\s*[-–]\s*\d{4}$", title):
        continue

    # -------------------------------
    # Extract area description
    # -------------------------------

    description_paragraphs = []

    # Iterate over siblings after <h2> until "Associated Faculty" section
    for sibling in title_tag.find_next_siblings():
        if sibling.name == "h3" and "Faculty" in sibling.get_text():
            break
        if sibling.name == "p":
            text = sibling.get_text(strip=True)
            description_paragraphs.append(text)

    # Join all paragraphs into one string
    description = " ".join(description_paragraphs)

    # -------------------------------
    # Extract associated professors
    # -------------------------------

    professors = []

    faculty_tag = block.find("div", class_="faculty-member-list")
    if faculty_tag:
        a_tags = faculty_tag.find_all("a", class_="faculty-link")

        for tag in a_tags:
            # Get profile link (make absolute if needed)
            href = tag["href"]
            if href.startswith("/"):
                href = "https://engineering.yale.edu" + href

            # Get professor name and title from <p> tag
            p = tag.find("p")
            name = p.find("strong").get_text(strip=True)
            title_text = p.get_text(strip=True).replace(name, "").strip()

            professors.append({
                "name": name,
                "title": title_text,
                "profile_url": href
            })

    # -------------------------------
    # Append this research area to results
    # -------------------------------

    data.append({
        "area": title,
        "description": description,
        "professors": professors
    })

# -------------------------------
# Step 4: Save output to JSON file
# -------------------------------

output_path = "data/research_areas.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(data)} research areas to {output_path}")
