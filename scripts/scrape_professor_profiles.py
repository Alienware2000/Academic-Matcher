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

def _clean_text(node) -> str:
    """Turn <br> into newlines and collapse whitespace."""
    txt = node.get_text(separator="\n", strip=True)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)   # at most 2 consecutive newlines
    return txt.strip()

def _collect_paras_and_lists(container) -> list[str]:
    """Collect paragraphs and list items as separate blocks (bullets for <li>)."""
    blocks = []
    # paragraphs
    for p in container.find_all("p"):
        t = _clean_text(p)
        if t:
            blocks.append(t)
    # lists
    for ul in container.find_all(["ul", "ol"]):
        for li in ul.find_all("li"):
            t = _clean_text(li)
            if t:
                blocks.append(f"- {t}")
    # fallback: if nothing matched, take raw text of container
    if not blocks:
        t = _clean_text(container)
        if t:
            blocks.append(t)
    return blocks

def extract_perspectives_strict(soup: BeautifulSoup) -> str | None:
    """
    Find <h3>Perspectives</h3>, then:
      - go up to the nearest ancestor with 'grid' in class
      - from that grid row, take the sibling col with 'col-span-2' (the content column)
      - collect <p> and <li> as blocks
    """
    # 1) find the <h3>
    h3 = None
    for tag in soup.find_all("h3"):
        if tag.get_text(strip=True).lower() == "perspectives":
            h3 = tag
            break
    if not h3:
        return None

    # 2) climb to the grid row
    grid_row = h3.find_parent(lambda t: t.name == "div" and t.get("class") and any("grid" in c for c in t.get("class")))
    if not grid_row:
        return None  # structure unexpected

    # 3) find the right/large column (content)
    content_div = None
    for div in grid_row.find_all("div", recursive=False):
        classes = " ".join(div.get("class", []))
        if "col-span-2" in classes or "lg:col-span-2" in classes:
            content_div = div
            break
    # fallback: any child div that actually has p/ul/ol
    if content_div is None:
        for div in grid_row.find_all("div", recursive=False):
            if div.find(["p", "ul", "ol"]):
                content_div = div
                break
    if not content_div:
        return None

    # 4) collect paragraphs and lists
    blocks = _collect_paras_and_lists(content_div)
    combined = "\n\n".join(blocks).strip()
    return combined or None

def extract_heading_until_next_h3(soup: BeautifulSoup, heading_text: str) -> str | None:
    """Find an <h3> with given text (case-insensitive); collect content until next <h3>."""
    target = None
    for h3 in soup.find_all("h3"):
        if h3.get_text(strip=True).lower() == heading_text.lower():
            target = h3
            break
    if not target:
        return None

    blocks = []
    for sib in target.find_all_next():
        if sib.name == "h3":  # stop at next section
            break
        if sib.name in ("p", "ul", "ol"):
            if sib.name == "p":
                t = _clean_text(sib)
                if t:
                    blocks.append(t)
            else:
                for li in sib.find_all("li"):
                    t = _clean_text(li)
                    if t:
                        blocks.append(f"- {t}")
    combined = "\n\n".join(blocks).strip()
    return combined or None

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
    # perspectives = extract_perspectives(soup)

    # NEW: get Perspectives robustly
    perspectives = (
        extract_perspectives_strict(soup)
        or extract_heading_until_next_h3(soup, "Perspectives")
        # Optional fallbacks if some pages use other labels:
        or extract_heading_until_next_h3(soup, "Research Interests")
        or extract_heading_until_next_h3(soup, "About")          # sometimes “About <Name>”
        or extract_heading_until_next_h3(soup, f"About {extract_name(soup) or ''}".strip())
        or extract_heading_until_next_h3(soup, "Biography")
        or None
    )

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

def validate_json(path: str):
    print()
    if not os.path.exists(path):
        print(f"[validate] not found: {path}")
        return
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    n_persp = sum(1 for r in rows if (r.get("perspectives") and r["perspectives"].strip()))
    n_name  = sum(1 for r in rows if (r.get("name") and r["name"].strip()))
    print(f"[validate] {path}")
    print(f"  records: {n}")
    print(f"  with perspectives: {n_persp}")
    print(f"  with name: {n_name}")
    print(f"  sample 2: {[(r.get('name'), bool(r.get('perspectives'))) for r in rows[:2]]}")

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

    # sample_urls = list(prof_to_areas.keys())[:2]
    # print(sample_urls)

    # small_map = dict(list(prof_to_areas.items())[:3])   # only 3 for testing
    # scrape_all_profiles(small_map, output_path="data/professor_profiles_test.json", delay=1.0)

    scrape_all_profiles(
    prof_to_areas,
    output_path="data/professor_profiles.json",
    delay=1.5  # be polite to the server
)
    
    # print()
    # rows = json.load(open("data/professor_profiles.json", encoding="utf-8"))
    # missing = [(r["name"], r["url"]) for r in rows if not (r.get("perspectives") and r["perspectives"].strip())]
    # print(len(missing), "missing")
    # for name, url in missing:
    #     print("-", name, "->", url)

    validate_json("data/professor_profiles.json")


    
