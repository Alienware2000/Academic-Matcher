import requests
from bs4 import BeautifulSoup
import re
import json

# Fetch the webpage HTML

url = "https://engineering.yale.edu/academic-study/departments/computer-science/research-areas"
response = requests.get(url)        # Send GET request to URL
html = response.text                # Extract the HTML text content

# print(html[:1000])                  # Print the first 1000 characters just to inspect it

# Parse the HTML using BeautifulSoup

soup = BeautifulSoup(html, "html.parser")

# Loop through all research area blocks

blocks = soup.find_all("div", class_="side-nav-blocks")
print(f"Found {len(blocks)} research areas.")

data = []

for block in blocks:
    title_tag = block.find("h2")
    if not title_tag:
        continue

    title = title_tag.get_text(strip=True)

    # Skip "technical report" year-range blocks
    if re.match(r"^\d{4}\s*[-–]\s*\d{4}$", title):
        continue  # e.g., "2025–2020"

    # Extract description paragraphs

    description_paragraphs = []

    # We start from the h2, and walk through the tags that come after it
    for sibling in title_tag.find_next_siblings():
        if sibling.name == "h3" and "Faculty" in sibling.get_text():
            break  # stop when we reach the "Associated Faculty" section
        if sibling.name == "p":
            text = sibling.get_text(strip=True)
            description_paragraphs.append(text)

    # Join all the description <p> tags into one string
    description = " ".join(description_paragraphs)
    # print("Description:", description[:200] + "...\n")  # Print first 200 characters to preview

    # Extract Associated Professors

    professors = []

    faculty_tag = block.find("div", class_="faculty-member-list")
    a_tag = faculty_tag.find_all("a")

    for tag in a_tag:
        href = tag["href"]
        if href.startswith("/"):
            href = "https://engineering.yale.edu" + href

        p = tag.find("p")
        name = p.find("strong").get_text(strip=True)
        name_title = p.get_text(strip=True).replace(name, "").strip()
        
        professors.append({
            "name": name,
            "title": name_title,
            "profile_url": href
        })

    data.append({
    "area": title,
    "description": description,
    "professors": professors
    })

# Save results to JSON
output_path = "data/research_areas.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(data)} research areas to {output_path}")
