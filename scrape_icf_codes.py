"""
Crawl the BfArM ICF Version 2005 website (klassifikationen.bfarm.de) and build a
flat reference table of all ICF codes with their German titles.

Output: dataset/icf_codes_reference.csv  (columns: code, title, level, source_page)
"""
import csv
import re
import time
import urllib.request

BASE = "https://klassifikationen.bfarm.de/icf/icfhtml2005/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script; ICF code list extraction)"}

CODE_RE = re.compile(
    r'<a name="([^"]+)" id="[^"]+" class="code">[^<]*</a>\s*<span class="label">([^<]*)</span>'
)
BLOCKLIST_LINK_RE = re.compile(
    r'<a href="([^"#]+)\.htm[^"]*"[^>]*class="code">[^<]*</a>\s*<span class="label">([^<]*)</span>'
)
CHAPTER_HEADER_RE = re.compile(
    r'<h2>Kapitel ([a-z]\d+)<br/>([^<]*)</h2>'
)

CHAPTERS = (
    [f"b{i}" for i in range(1, 9)]
    + [f"s{i}" for i in range(1, 9)]
    + [f"d{i}" for i in range(1, 10)]
    + [f"e{i}" for i in range(1, 6)]
)


def fetch(page):
    url = BASE + page
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main():
    results = {}  # code -> (title, source_page)
    block_pages = set()

    for chapter in CHAPTERS:
        page = f"chapter-{chapter}.htm"
        html = fetch(page)
        time.sleep(0.3)

        m = CHAPTER_HEADER_RE.search(html)
        if m:
            code, title = m.group(1), m.group(2).strip()
            results.setdefault(code, (title, page))

        for href, title in BLOCKLIST_LINK_RE.findall(html):
            block_pages.add(href + ".htm")

    print(f"Found {len(block_pages)} block/singles pages")

    for page in sorted(block_pages):
        try:
            html = fetch(page)
        except Exception as e:
            print(f"  failed {page}: {e}")
            continue
        time.sleep(0.3)

        for code, title in CODE_RE.findall(html):
            results.setdefault(code, (title.strip(), page))

    print(f"Collected {len(results)} ICF codes total")

    out_path = "dataset/icf_codes_reference.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "title_de", "source_page"])
        for code in sorted(results):
            title, source = results[code]
            writer.writerow([code, title, source])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
