import requests
from bs4 import BeautifulSoup
import re

# Step 1: Fetch the webpage HTML

url = "https://engineering.yale.edu/academic-study/departments/computer-science/research-areas"

response = requests.get(url)        # Send GET request to URL
html = response.text                # Extract the HTML text content

# print(html[:1000])                  # Print the first 1000 characters just to inspect it


# Step 2: Parse the HTML using BeautifulSoup

# Assume you already fetched 'html' using requests
soup = BeautifulSoup(html, "html.parser")

# Print out the first few tags just to explore
# print(soup.prettify()[:1000])  # Optional: visualize the parsed HTML nicely formatted

# Let's try finding the first research area block

# block = soup.find("div", class_="side-nav-blocks")
# print(block.h2.text)  # Should print "Artificial Intelligence & Machine Learning"

# Step 3: Loop through all research area blocks

blocks = soup.find_all("div", class_="side-nav-blocks")
print(f"Found {len(blocks)} research areas.")

for block in blocks:
    title_tag = block.find("h2")
    if not title_tag:
        continue

    title = title_tag.get_text(strip=True)

    # Skip "technical report" year-range blocks
    if re.match(r"^\d{4}\s*[-–]\s*\d{4}$", title):
        continue  # e.g., "2025–2020"

    print("Area:", title)

    # Step 4: Extract description paragraphs

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
    print("Description:", description[:200] + "...\n")  # Print first 200 characters to preview

