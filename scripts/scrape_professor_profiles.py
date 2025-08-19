import os
import re
import json
import time
import requests
from tqdm import tqdm
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import defaultdict

# # Path to your JSON file
# file_path = "data/research_areas.json"

# # Open and load it
# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Now `data` is a Python object (usually a list or dict)
# print(f"Loaded {len(data)} research_areas from {file_path}")

# sample_urls = list(prof_to_areas.keys())[:2]
# print(sample_urls)

HEADERS = {
    # A User-Agent helps sites not reject “unknown” bots
    "User-Agent": "Mozilla/5.0 (compatible; AcademicMatcherBot/0.1; +https://github.com/Alienware2000/Academic-Matcher)"
}

def url_to_slug(url: str) -> str:
    """
    Convert a profile URL into a safe slug for filenames.
    e.g., https://.../faculty-directory/nisheeth-vishnoi -> nisheeth-vishnoi
    """
    path = urlparse(url).path  # '/research-and-faculty/faculty-directory/nisheeth-vishnoi'
    slug = path.rstrip('/').split('/')[-1]
    return slug or "profile"

def fetch_html(url: str, timeout: int = 20) -> str:
    """
    Fetch HTML for a page, with basic error handling.
    Returns the HTML text if successful; raises for non-200 status.
    """
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()  # raise error if 4xx/5xx so we notice failures
    return resp.text

def save_raw_html(html: str, slug: str) -> str:
    """
    Save the raw HTML so you can debug parsers later if needed.
    """
    raw_dir = os.path.join("data", "raw", "professors")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"{slug}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def extract_name(soup: BeautifulSoup) -> str | None:
    """
    The professor's name appears as <h1> at the top.
    """
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else None


def extract_title(soup: BeautifulSoup) -> str | None:
    """
    The title appears in a big header area as an <h2>.
    We'll find the first <h2> after <h1>.
    """
    h1 = soup.find("h1")
    if not h1:
        return None
    h2 = h1.find_next("h2")
    return h2.get_text(strip=True) if h2 else None

def extract_email(soup: BeautifulSoup) -> str | None:
    """
    Find a link whose href starts with 'mailto:'.
    Extract the email part after 'mailto:'.
    """
    a = soup.find("a", href=lambda x: isinstance(x, str) and x.startswith("mailto:"))
    if not a:
        return None
    mailto = a.get("href", "")
    return mailto.replace("mailto:", "").strip() or None


def extract_website(soup: BeautifulSoup) -> str | None:
    """
    There is often a blue 'Website: Research Website' button.
    We'll look for an <a> that contains 'Website:' in the text.
    """
    for a in soup.find_all("a", href=True):
        text = a.get_text(separator=" ", strip=True)
        if "Website:" in text:
            return a["href"]
    return None

def extract_labeled_fields(soup: BeautifulSoup) -> dict:
    """
    Extract fields like:
      'Room / Office', 'Office Address', 'Mailing Address'
    Returns a dict with any that are found.
    """
    results = {}

    # Strategy: find all <p> that have a <strong> inside -> label
    for p in soup.find_all("p"):
        strong = p.find("strong")
        if not strong:
            continue

        label = strong.get_text(strip=True).rstrip(":")
        # Remove the label text from this <p> to get the inline value, if any
        inline_text = p.get_text(separator=" ", strip=True)
        # e.g. "Room / Office: Room 319" -> remove "Room / Office:" -> "Room 319"
        after_label = inline_text.replace(strong.get_text(strip=True), "").lstrip(":").strip()

        if after_label:
            # We got an inline value (e.g., 'Room 319')
            results[label] = after_label
        else:
            # Sometimes the value is in the next sibling <p> (especially for addresses)
            next_p = p.find_next_sibling("p")
            if next_p and not next_p.find("strong"):
                # Grab the full text (addresses may contain <br> -> join with spaces)
                value = next_p.get_text(separator=" ", strip=True)
                if value:
                    results[label] = value

    return results

def extract_perspectives(soup: BeautifulSoup) -> str | None:
    """
    Find the section headed by <h3>Perspectives</h3>, then read the content
    in the adjacent column (usually paragraphs).
    """
    # find <h3> whose text is exactly 'Perspectives' ignoring case/spaces
    h3 = None
    for tag in soup.find_all("h3"):
        if tag.get_text(strip=True).lower() == "perspectives":
            h3 = tag
            break
    if not h3:
        return None

    # The page uses a 3-column grid; the content is in the second/third column.
    # Walk up to a reasonable container (the grid row), then find the large column.
    row = h3.find_parent(class_=re.compile(r"grid|col-span"))
    if not row:
        # Fallback: just collect paragraphs after the h3 until the next h3
        texts = []
        for sib in h3.find_all_next():
            if sib.name == "h3":
                break
            if sib.name == "p":
                texts.append(sib.get_text(" ", strip=True))
        return "\n\n".join(texts) if texts else None

    # Preferred: within this row, find the wider column
    content_div = None
    for div in row.find_all("div", recursive=False):
        # Heuristic: pick the div that is not the label column and has paragraphs
        ps = div.find_all("p")
        if ps and (div.get("class") and any("col-span-2" in c for c in div.get("class")) or len(ps) >= 1):
            content_div = div
            break

    # Collect paragraphs from the content div
    texts = []
    if content_div:
        for p in content_div.find_all("p"):
            texts.append(p.get_text(" ", strip=True))
    else:
        # Fallback if structure differs
        for p in row.find_all("p"):
            texts.append(p.get_text(" ", strip=True))

    return "\n\n".join(texts).strip() if texts else None

def parse_professor_profile(url: str, html: str) -> dict:
    """
    Given the profile page HTML, extract key fields into a dict.
    """
    soup = BeautifulSoup(html, "html.parser")

    name = extract_name(soup)
    title = extract_title(soup)
    email = extract_email(soup)
    website = extract_website(soup)
    labeled = extract_labeled_fields(soup)
    perspectives = extract_perspectives(soup)

    # Normalize the fields we care about, pulling from labeled where needed
    room_office = labeled.get("Room / Office") or labeled.get("Room/Office")
    office_address = labeled.get("Office Address")
    # Note: We're intentionally ignoring "Mailing Address" for now

    return {
        "url": url,
        "name": name,
        "title": title,
        "email": email,
        "website": website,
        "room_office": room_office,
        "office_address": office_address,
        "perspectives": perspectives,
    }

def scrape_all_profiles(prof_to_areas,
                        output_path="data/professor_profiles.json",
                        raw_dir="data/raw/professors",
                        delay=1.0):
    """
    Scrape every professor profile URL in prof_to_areas (dict url -> set(areas)).
    - Saves raw HTML snapshots to raw_dir.
    - Writes incremental progress to output_path after each successful profile.
    - Resumes if output_path already exists.
    """

    # Make sure the raw HTML dir exists
    os.makedirs(raw_dir, exist_ok=True)

    # --- Resume support: load already-scraped profiles if file exists ---
    scraped_map = {}   # url -> record
    results = []       # list of records we'll write out
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_list = json.load(f)
            # create quick lookup to avoid re-scraping
            for r in existing_list:
                if r.get("url"):
                    scraped_map[r["url"]] = r
            results = existing_list[:]  # start from existing
            print(f"[resume] Loaded {len(existing_list)} existing profiles from {output_path}")
        except Exception as e:
            print(f"[warn] Could not load existing {output_path}: {e}. Starting fresh.")
            results = []
            scraped_map = {}

    total = len(prof_to_areas)
    scraped_count = len(scraped_map)
    failed = []

    # Iterate with a progress bar
    for url, areas in tqdm(prof_to_areas.items(), total=total, desc="scrape profiles"):
        # Skip if already scraped
        if url in scraped_map:
            tqdm.write(f"[skip] already scraped: {url}")
            continue

        try:
            # 1) Fetch
            html = fetch_html(url)

            # 2) Save raw HTML snapshot (for debugging / reproducibility)
            slug = url_to_slug(url)
            raw_path = save_raw_html(html, slug)  # returns the path where saved
            tqdm.write(f"[saved raw] {raw_path}")

            # 3) Parse the page into structured fields
            record = parse_professor_profile(url, html)

            # 4) Attach additional metadata
            record["areas"] = sorted(list(areas))
            record["last_scraped_at"] = datetime.utcnow().isoformat() + "Z"
            record["raw_html_path"] = raw_path

            # 5) Append to results and persist immediately
            results.append(record)
            scraped_map[url] = record
            with open(output_path, "w", encoding="utf-8") as fout:
                json.dump(results, fout, indent=2, ensure_ascii=False)
            tqdm.write(f"[ok] scraped and saved: {url}")

        except Exception as exc:
            # Catch exceptions so one bad page doesn't stop the whole run
            err = str(exc)
            tqdm.write(f"[fail] {url} -> {err}")
            failed.append({"url": url, "error": err})
            # also persist failures so you can inspect later
            with open("data/failed_professors.json", "w", encoding="utf-8") as ferr:
                json.dump(failed, ferr, indent=2, ensure_ascii=False)

        # Politeness: pause between requests
        time.sleep(delay)

    # Final summary
    print("\n--- scrape finished ---")
    print(f"Total candidate URLs: {total}")
    print(f"Successfully scraped: {len(scraped_map)}")
    print(f"Failed: {len(failed)}")
    print(f"Output written to: {output_path}")
    if failed:
        print("Failed examples saved to data/failed_professors.json")

if __name__ == "__main__":
    # Path to your JSON file
    file_path = "data/research_areas.json"

    # Load your research_areas.json:
    with open(file_path, "r", encoding="utf-8") as f:
        areas = json.load(f)

    # Now `areas` is a Python object (usually a list or dict)
    print(f"Loaded {len(areas)} research_areas from {file_path}")

    # Build a mapping: profile_url -> set(areas)
    prof_to_areas = defaultdict(set)

    for area in areas:  # data = research_areas loaded from JSON
        area_name = area.get("area")
        for prof in area.get("professors", []):
            url = prof.get("profile_url")
            if url:
                prof_to_areas[url].add(area_name)

    print(f"Unique professor profile URLs: {len(prof_to_areas)}")

    small_map = dict(list(prof_to_areas.items())[:3])   # only 3 for testing
    scrape_all_profiles(small_map, output_path="data/professor_profiles_test.json", delay=1.0)

    # # --- Pick ONE URL to start ---
    # test_url = unique_urls[0]
    # print(f"\nFetching one profile: {test_url}")

    # # Fetch + save raw HTML
    # html = fetch_html(test_url)
    # slug = url_to_slug(test_url)
    # raw_path = save_raw_html(html, slug)
    # print(f"Saved raw HTML to {raw_path}")

    # # Parse fields
    # profile = parse_professor_profile(test_url, html)

    # # Pretty print results for inspection
    # print("\nParsed profile:")
    # for k, v in profile.items():
    #     if isinstance(v, str) and len(v) > 300:
    #         print(f"- {k}: {v[:300]}... [truncated]")
    #     else:
    #         print(f"- {k}: {v}")
