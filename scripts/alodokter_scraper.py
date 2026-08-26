#!/usr/bin/env python
"""Scrape Alodokter community Q&A threads for the topic tag "depresi".

Collects real patient questions (and the doctors' public answers) from
https://www.alodokter.com/komunitas/topic-tag/depresi for the deployed-chatbot
evaluation. Two stages: Stage 1 enumerates thread URLs from the listing pages;
Stage 2 opens each thread and extracts the full question (``<detail-topic>``)
and all doctor answers (``<doctor-topic>``).

    python scripts/alodokter_scraper.py                       # pages 1-25
    python scripts/alodokter_scraper.py --end-page 5 --limit 20
    python scripts/alodokter_scraper.py --cookie "$(cat cookie.txt)"
    python scripts/alodokter_scraper.py --browser             # Playwright fallback

Dependencies (not part of the core pipeline): requests, beautifulsoup4, lxml,
and optionally playwright (`playwright install chromium`) for --browser.

Anti-bot note: the site sits behind Akamai; datacenter IPs are often served an
empty page shell. The script checks the first thread and aborts with advice if
the content is missing. Fixes, in order: run from a residential connection,
pass your logged-in browser Cookie header via --cookie (DevTools -> Network ->
any alodokter request -> copy the "cookie:" value), or use --browser to render
pages with a real headless browser.

Ethics: these are sensitive mental-health posts. Use for private research only,
take only what you need, and never republish raw posts. Anonymization is ON by
default but only scrubs emails and phone numbers - free-text names/ages/places
are NOT detected, so manually review anything that will be quoted. The output
goes to data/ (not tracked in git). Consider asking Alodokter for permission
(support@alodokter.com).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
BASE_HOST = "https://www.alodokter.com"
BASE_URL = f"{BASE_HOST}/komunitas/topic-tag/depresi"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    "Referer": BASE_URL,
    "Upgrade-Insecure-Requests": "1",
}

_PHONE = re.compile(r"(?:\+62|62|0)8\d{7,11}")
_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")

FIELDS = ["id", "title", "doctor_listing", "replies", "listing_time", "url",
          "question_title", "question", "question_date",
          "answer", "answer_doctors", "n_answers"]


def scrub(text: str) -> str:
    """Redact emails and Indonesian phone numbers (regex-level anonymization)."""
    if not text:
        return text
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text))


def decode_content(raw: str) -> str:
    """detail/doctor content is a JSON string literal containing HTML - decode
    both layers (\\u003c -> <, \\n -> newline) down to plain text."""
    if not raw:
        return ""
    try:
        decoded = json.loads(raw)
    except Exception:
        decoded = html.unescape(raw)
    text = BeautifulSoup(decoded, "lxml").get_text(" ", strip=True)
    text = _fix_mojibake(text)
    # normalize non-breaking spaces (the un-mojibaked remnant of "\u00c2\u00a0")
    text = text.replace("\u00a0", " ")
    return re.sub(r" {2,}", " ", text).strip()


def _fix_mojibake(text: str) -> str:
    """Repair the site's double-encoded UTF-8 ("\u00c2\u00b2" -> "\u00b2", curly
    quotes, dashes). Applies the inverse transform (latin-1 bytes re-read as
    UTF-8) only when the telltale lead chars are present AND the whole string
    survives the round trip; otherwise the text is returned unchanged."""
    if not any(m in text for m in ("\u00c2", "\u00e2", "\u00c3")):
        return text
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # a lone "\u00c2" before whitespace is leftover non-breaking-space mojibake
    return re.sub("\u00c2(?=[ \u00a0])|\u00c2$", "", text)


def parse_listing(html_text: str) -> list[dict]:
    soup = BeautifulSoup(html_text, "lxml")
    rows = []
    for c in soup.find_all("card-topic"):
        href = c.get("href", "")
        rows.append({
            "id": c.get("id-data", ""),
            "title": c.get("title", "").strip(),
            "doctor_listing": c.get("pickup-name", "").strip(),
            "replies": c.get("counter-reply", ""),
            "listing_time": c.get("text-time", "").strip(),
            "url": urljoin(BASE_HOST, href) if href else "",
        })
    return rows


def parse_detail(html_text: str, anonymize: bool) -> dict:
    soup = BeautifulSoup(html_text, "lxml")
    dt = soup.find("detail-topic")
    q_title = q_text = q_date = ""
    if dt:
        q_title = dt.get("member-topic-title", "").strip()
        q_text = decode_content(dt.get("member-topic-content", ""))
        q_date = dt.get("member-post-date", "").strip()
    answers, doctors = [], []
    for d in soup.find_all("doctor-topic"):
        a = decode_content(d.get("doctor-topic-content", ""))
        if a:
            answers.append(a)
        doc = d.get("by-doctor", "").strip()
        if doc:
            doctors.append(doc)
    if anonymize:
        q_text = scrub(q_text)
        answers = [scrub(a) for a in answers]
    return {
        "question_title": q_title,
        "question": q_text,
        "question_date": q_date,
        "answer": "\n\n---\n\n".join(answers),
        "answer_doctors": "; ".join(dict.fromkeys(doctors)),
        "n_answers": len(answers),
    }


_EMPTY_DETAIL = {"question_title": "", "question": "", "question_date": "",
                 "answer": "", "answer_doctors": "", "n_answers": 0}


class RequestsFetcher:
    def __init__(self, cookie: str = "") -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        if cookie:
            self.session.headers["Cookie"] = cookie

    def fetch(self, url: str) -> str:
        r = self.session.get(url, timeout=25)
        r.raise_for_status()
        return r.text

    def close(self) -> None:
        self.session.close()


class BrowserFetcher:
    """Playwright fallback: renders pages in headless Chromium, which passes
    the JS/anti-bot rendering that plain requests may not."""

    def __init__(self) -> None:
        from playwright.sync_api import sync_playwright  # lazy: optional dep

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page(locale="id-ID")

    def fetch(self, url: str) -> str:
        self._page.goto(url, wait_until="networkidle", timeout=45_000)
        return self._page.content()

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()


def scrape_listing(fetcher, start: int, end: int, delay: float) -> list[dict]:
    out: list[dict] = []
    for p in range(start, end + 1):
        url = BASE_URL if p == 1 else f"{BASE_URL}/page/{p}"
        print(f"listing page {p} ...", end=" ", flush=True)
        try:
            rows = parse_listing(fetcher.fetch(url))
        except requests.HTTPError as e:
            print(f"HTTP error {e} - stopping.")
            break
        if not rows:
            print("empty (last page) - stopping.")
            break
        print(f"{len(rows)} threads")
        out.extend(rows)
        time.sleep(delay)
    return out


def content_served(fetcher, url: str) -> bool:
    """True if the page contains real thread markup (not an anti-bot shell)."""
    try:
        body = fetcher.fetch(url)
    except Exception as e:
        print(f"diagnostic fetch failed: {e}")
        return False
    return "detail-topic" in body and "doctor-topic" in body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alodokter-scraper", description=__doc__)
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--end-page", type=int, default=25)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=None, help="max threads to scrape")
    ap.add_argument("--cookie", default="", help="browser Cookie header (anti-bot bypass)")
    ap.add_argument("--browser", action="store_true",
                    help="render with headless Chromium (Playwright) instead of requests")
    ap.add_argument("--no-anonymize", action="store_true",
                    help="keep emails/phone numbers (NOT recommended)")
    ap.add_argument("--out", default=str(ROOT / "data" / "derived" / "alodokter_depresi_qa.csv"))
    args = ap.parse_args(argv)
    anonymize = not args.no_anonymize

    fetcher = BrowserFetcher() if args.browser else RequestsFetcher(args.cookie)
    try:
        listing = scrape_listing(fetcher, args.start_page, args.end_page, args.delay)
        print(f"\nenumerated {len(listing)} threads")
        if not listing:
            print("no threads found - the listing itself is likely blocked; "
                  "try --cookie or --browser.")
            return 1
        if args.limit:
            listing = listing[: args.limit]

        if not content_served(fetcher, listing[0]["url"]):
            print("BLOCKED: thread pages are served as an empty shell (anti-bot). "
                  "Re-run with --cookie '<your browser cookie header>' or --browser.")
            return 1

        records = []
        for i, row in enumerate(listing, 1):
            print(f"thread {i}/{len(listing)} ...", end=" ", flush=True)
            try:
                detail = parse_detail(fetcher.fetch(row["url"]), anonymize)
                print(f"{detail['n_answers']} answer(s)")
            except Exception as e:
                detail = dict(_EMPTY_DETAIL)
                print(f"error: {e}")
            records.append({**row, **detail})
            time.sleep(args.delay)
    finally:
        fetcher.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(records)
    ok = sum(1 for r in records if r["question"])
    print(f"\nwrote {len(records)} rows ({ok} with question text, "
          f"anonymized={anonymize}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
