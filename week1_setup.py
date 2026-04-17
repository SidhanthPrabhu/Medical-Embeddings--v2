import os
import json
import time
import datetime
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR     = "data"
OUTPUT_FILE  = os.path.join(DATA_DIR, "pubmed_abstracts.jsonl")

# NCBI Credentials
NCBI_API_KEY = "083329a150863384d479edac2c955b255508"  # <-- Paste your exact API key here
EMAIL        = "john.daley267@gmail.com"

# Target & Limits
TARGET_TOTAL = 450_000   # Sufficient to hit ~4M sentences after preprocessing
BATCH_SIZE   = 100       # Smaller batch size to prevent server timeouts
WINDOW_SIZE  = 9_000     
NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_QUERY = (
    "drug interactions[MeSH Terms] OR "
    "drug therapy[MeSH Terms] OR "
    "pharmacology[MeSH Terms] OR "
    "adverse effects[MeSH Terms] OR "
    "clinical[Title/Abstract]"
)

# Threading Controls
MAX_CONCURRENT_REQUESTS = 10 
api_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
file_write_lock = threading.Lock()

# ── Date Generation ──────────────────────────────────────────────────────────
def generate_monthly_windows(start_year, end_year):
    windows = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d_from = datetime.date(year, month, 1)
            if month == 12:
                d_to = datetime.date(year, 12, 31)
            else:
                d_to = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            windows.append((d_from.strftime("%Y/%m/%d"), d_to.strftime("%Y/%m/%d")))
    return windows[::-1]

# Expanded range to guarantee enough records
DATE_WINDOWS = generate_monthly_windows(2005, 2026)

# ── Networking ───────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def search_window(session, query: str, d_from: str, d_to: str) -> tuple:
    params = {
        "db": "pubmed", "term": f"{query} AND {d_from}:{d_to}[dp]",
        "retmax": 0, "usehistory": "y", "retmode": "json",
        "email": EMAIL, "api_key": NCBI_API_KEY
    }
    resp = session.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["esearchresult"]
    return data["webenv"], data["querykey"], min(int(data["count"]), WINDOW_SIZE)

def parse_xml_batch(xml_text: str) -> list:
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError: return records
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="").strip()
        parts = article.findall(".//AbstractText")
        if not parts: continue
        abstract = " ".join((p.text or "").strip() for p in parts if p.text)
        if abstract:
            records.append({
                "pmid": pmid, 
                "title": article.findtext(".//ArticleTitle", default="").strip(), 
                "abstract": abstract
            })
    return records

def fetch_and_save_batch(session, web_env, qk, start, fout, seen_set):
    """Worker function for threads. Fetches XML and safely writes to JSONL."""
    params = {
        "db": "pubmed", "WebEnv": web_env, "query_key": qk,
        "retstart": start, "retmax": BATCH_SIZE, "rettype": "abstract",
        "retmode": "xml", "email": EMAIL, "api_key": NCBI_API_KEY
    }
    
    with api_semaphore:
        # Enforce rate limit (10 requests per second max)
        time.sleep(0.1) 
        
        for attempt in range(5):
            try:
                resp = session.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=120)
                resp.raise_for_status()
                batch = parse_xml_batch(resp.text)
                
                new_count = 0
                # Thread-safe file writing and set updating
                with file_write_lock:
                    for rec in batch:
                        if rec["pmid"] not in seen_set:
                            fout.write(json.dumps(rec) + "\n")
                            seen_set.add(rec["pmid"])
                            new_count += 1
                return new_count
                
            # Catch ALL requests-related exceptions (Timeouts, ChunkedEncoding, 502s)
            except requests.exceptions.RequestException as e:
                if attempt == 4:
                    print(f"\n[!] Offset {start} failed permanently after 5 attempts: {e}")
                    return 0
                time.sleep(2 ** attempt)  # Exponential backoff
        return 0

# ── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    session = make_session()
    seen_pmids = set()
    total_downloaded = 0

    print(f"[1/2] Searching {len(DATE_WINDOWS)} date windows...")
    windows = []
    for d_from, d_to in DATE_WINDOWS:
        try:
            we, qk, cnt = search_window(session, SEARCH_QUERY, d_from, d_to)
            if cnt > 0: 
                windows.append((we, qk, cnt, d_from[:7]))
            time.sleep(0.1)
        except Exception as e: 
            continue

    print(f"[2/2] Parallel Fetching to {OUTPUT_FILE} (Target: {TARGET_TOTAL:,})")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for we, qk, cnt, label in windows:
            if total_downloaded >= TARGET_TOTAL: break
            
            offsets = list(range(0, cnt, BATCH_SIZE))
            
            with tqdm(total=len(offsets), desc=label, unit="batch", leave=False) as pbar:
                with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                    
                    # Submit all batch jobs for this month to the thread pool
                    futures = [
                        executor.submit(fetch_and_save_batch, session, we, qk, start, fout, seen_pmids) 
                        for start in offsets
                    ]
                    
                    for future in as_completed(futures):
                        added = future.result()
                        total_downloaded += added
                        pbar.set_postfix(total=f"{total_downloaded:,}")
                        pbar.update(1)
                        
                        # Hard stop if target reached
                        if total_downloaded >= TARGET_TOTAL:
                            # Cancel remaining futures in this executor
                            for f in futures: f.cancel()
                            break

    print(f"\nDone! {total_downloaded:,} unique records saved successfully.")
    print("Next step -> run week2_preprocess.py")