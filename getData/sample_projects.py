import argparse
import math
import os
import random
import time
import sqlite3
import requests
from collections import Counter
from typing import Dict, Optional
import pandas as pd
import xml.etree.ElementTree as ET
import json

API_BASE = "https://gtr.ukri.org/gtr/api/projects"
API_NS   = "http://gtr.rcuk.ac.uk/gtr/api"
HEADERS  = {"Accept": "application/xml", "User-Agent": "Mozilla/5.0"}

DB_PATH  = "projects_sample.db"
RATE     = 0.05
MIN_KEEP = 70
PAGE_SIZE = 100
MAX_PAGE = 1680  # API page limit

def init_db():
    """Initializes the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        funder TEXT,
        api_url TEXT
    );
    
    CREATE TABLE IF NOT EXISTS sampling_progress (
        id INTEGER PRIMARY KEY,
        last_page INTEGER,
        funder_counts TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS funder_stats (
        funder TEXT PRIMARY KEY,
        target_count INTEGER,
        current_count INTEGER,
        total_encountered INTEGER,
        from_excel BOOLEAN DEFAULT 0
    );
    """)
    conn.commit()
    return conn, cur

def save_progress(cur, page, taken_counts):
    """Saves the progress to the database."""
    counts_json = json.dumps(dict(taken_counts))
    cur.execute("DELETE FROM sampling_progress")
    cur.execute("INSERT INTO sampling_progress (last_page, funder_counts) VALUES (?, ?)", 
                (page, counts_json))

def load_progress(cur):
    """Loads progress from the database."""
    cur.execute("SELECT last_page, funder_counts FROM sampling_progress ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    if row:
        last_page, counts_json = row
        taken_counts = Counter(json.loads(counts_json))
        return last_page, taken_counts
    return 0, Counter()

def update_funder_stats(cur, funder, target=0, current=0, encountered=0, from_excel=False):
    """Updates funder statistics."""
    cur.execute("""
    INSERT OR REPLACE INTO funder_stats 
    (funder, target_count, current_count, total_encountered, from_excel) 
    VALUES (?, ?, ?, ?, ?)
    """, (funder, target, current, encountered, from_excel))

def get_attr(elem: ET.Element, name: str):
    """Gets an element attribute."""
    return elem.attrib.get(name) or elem.attrib.get(f"{{{API_NS}}}{name}")

def safe_get(url, retries=5, backoff=10):
    """Safe HTTP request with retry mechanism."""
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait_time = backoff * (2 ** i)
                print(f"  API rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  HTTP {r.status_code}, retrying {i+1}/{retries}")
                time.sleep(backoff * (i + 1))
        except requests.RequestException as e:
            print(f"  Network error: {e}, retrying {i+1}/{retries}")
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    
    print(f"  Failed after multiple retries, skipping URL: {url}")
    return None

def normalize_funder_name(name: str) -> str:
    """Normalizes funder names."""
    if not name:
        return ""
    
    normalized = name.upper().strip()
    
    replacements = {
        "RESEARCH COUNCILS UK": "RCUK",
        "ENGINEERING AND PHYSICAL SCIENCES RESEARCH COUNCIL": "EPSRC",
        "BIOTECHNOLOGY AND BIOLOGICAL SCIENCES RESEARCH COUNCIL": "BBSRC",
        "ECONOMIC AND SOCIAL RESEARCH COUNCIL": "ESRC",
        "NATURAL ENVIRONMENT RESEARCH COUNCIL": "NERC",
        "MEDICAL RESEARCH COUNCIL": "MRC",
        "ARTS AND HUMANITIES RESEARCH COUNCIL": "AHRC",
        "SCIENCE AND TECHNOLOGY FACILITIES COUNCIL": "STFC",
        "NATIONAL CENTRE FOR THE REPLACEMENT REFINEMENT AND REDUCTION OF ANIMALS IN RESEARCH": "NC3RS",
        "NATIONAL CENTRE FOR THE 3RS": "NC3RS",
        "TECHNOLOGY MEANS BUSINESS": "TMF",
        "UNITED KINGDOM RESEARCH AND INNOVATION": "UKRI",
    }
    
    for full_name, abbrev in replacements.items():
        if full_name in normalized:
            normalized = normalized.replace(full_name, abbrev)
    
    normalized = normalized.replace("-", " ").replace("_", " ").replace(".", "")
    normalized = " ".join(normalized.split())
    
    return normalized

def create_funder_mapping(excel_funders: Dict[str, int]) -> Dict[str, str]:
    """Creates a mapping from API funder names to Excel funder names."""
    mapping = {}
    
    for excel_funder in excel_funders.keys():
        normalized = normalize_funder_name(excel_funder)
        mapping[normalized] = excel_funder
    
    known_mappings = {
        normalize_funder_name("Ayrton Fund"): "AYRTON FUND",
        normalize_funder_name("Horizon Europe Guarantee"): "HORIZON EUROPE GUARANTEE", 
        normalize_funder_name("Infrastructure Fund"): "INFRASTRUCTURE FUND",
        normalize_funder_name("Innovate UK"): "INNOVATE UK",
        normalize_funder_name("NC3Rs"): "NC3RS",
        normalize_funder_name("Newton Fund"): "NEWTON FUND",
        normalize_funder_name("Open Access Block Grant"): "OPEN ACCESS BLOCK GRANT",
        normalize_funder_name("TMF"): "TMF",
    }
    
    for normalized_api, excel_name in known_mappings.items():
        if excel_name in excel_funders:
            mapping[normalized_api] = excel_name
    
    return mapping

def load_counts_table(path: str, delimiter: Optional[str]=None) -> Dict[str,int]:
    """Loads the funder counts table."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx",".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path, sep=delimiter if delimiter else None, engine="python")
    
    name_c = "FundingOrgName"
    num_c  = "Count"
    df[num_c] = df[num_c].astype(str).str.replace(r"[^\d]", "", regex=True)
    df = df[df[num_c].str.isnumeric()]
    return dict(zip(df[name_c].str.strip(), df[num_c].astype(int)))

def has_outcomes(project_elem, ns):
    """Checks if a project has outcomes."""
    links = project_elem.find('ns1:links', ns)
    if links is not None:
        for link in links.findall('ns1:link', ns):
            href = get_attr(link, 'href')
            if href and "/outcomes/" in href:
                return True
    return False

def sample_projects(counts_path: str, sep=None, delay=(0,0), resume=False):
    """Samples projects from the API."""
    conn, cur = init_db()
    
    excel_funders = load_counts_table(counts_path, sep)
    totals = Counter(excel_funders)
    targets = {f: max(math.ceil(n * RATE), MIN_KEEP)
               for f, n in totals.items()}
    
    funder_mapping = create_funder_mapping(excel_funders)
    
    if resume:
        start_page, taken = load_progress(cur)
        if start_page > 0:
            print(f"Resuming sampling from page {start_page + 1}...")
            print(f"Current sampled project stats:")
            for funder, count in taken.most_common(10):
                target = targets.get(funder, MIN_KEEP)
                print(f"  {funder}: {count}/{target}")
            if len(taken) > 10:
                print(f"  ... and {len(taken) - 10} more funders")
        else:
            print("No previous progress found, starting from the beginning...")
            taken = Counter()
            start_page = 0
    else:
        taken = Counter()
        start_page = 0
        cur.execute("DELETE FROM sampling_progress")
        conn.commit()
    
    encountered = Counter()
    
    for funder, target in targets.items():
        update_funder_stats(cur, funder, target=target, from_excel=True)
    
    print(f"Loaded quotas for {len(targets)} funders from Excel:")
    for funder, target in list(targets.items())[:10]:
        print(f"  {funder}: {target}")
    if len(targets) > 10:
        print(f"  ... and {len(targets)-10} more funders")
    print()
    
    consecutive_failures = 0
    max_failures = 10
    
    page = start_page + 1
    while True:
        url = f"{API_BASE}?p={page}&s={PAGE_SIZE}"
        resp = safe_get(url)
        
        if not resp:
            consecutive_failures += 1
            print(f"Failed to fetch page {page} (consecutive failures: {consecutive_failures}/{max_failures})")
            
            if consecutive_failures >= max_failures:
                print("Too many consecutive failures, saving progress and exiting.")
                save_progress(cur, page - 1, taken)
                conn.commit()
                break
            
            time.sleep(30)
            page += 1
            continue
        
        consecutive_failures = 0
            
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            print(f"Invalid XML on page {page}, skipping.")
            page += 1
            continue
            
        ns = {'ns1': API_NS, 'ns2': 'http://gtr.rcuk.ac.uk/gtr/api/project'}
        projects = root.findall('ns2:project', ns)
        if not projects:
            print("No more projects found, sampling complete.")
            break
        
        page_added = 0
        for pr in projects:
            fund_tag = pr.find('ns2:leadFunder', ns)
            api_funder = fund_tag.text if fund_tag is not None else "UNKNOWN"
            encountered[api_funder] += 1
            
            normalized_api_funder = normalize_funder_name(api_funder)
            excel_funder = funder_mapping.get(normalized_api_funder)
            
            if excel_funder:
                working_funder = excel_funder
                if working_funder not in targets:
                    targets[working_funder] = max(math.ceil(excel_funders[excel_funder] * RATE), MIN_KEEP)
                    print(f"  Matched: '{api_funder}' -> '{excel_funder}' (Quota: {targets[working_funder]})")
            else:
                working_funder = api_funder
                if working_funder not in targets:
                    targets[working_funder] = MIN_KEEP
                    update_funder_stats(cur, working_funder, target=MIN_KEEP, from_excel=False)
                    print(f"  New funder found: {working_funder} -> {MIN_KEEP}")
            
            if taken[working_funder] >= targets[working_funder]:
                continue
            
            pid = get_attr(pr, 'id')
            href = get_attr(pr, 'href')
            if not pid or not href:
                continue
            
            if has_outcomes(pr, ns):
                cur.execute("INSERT OR IGNORE INTO projects VALUES (?,?,?)", 
                           (pid, working_funder, href))
                taken[working_funder] += 1
                page_added += 1
                
                update_funder_stats(cur, working_funder, 
                                   target=targets[working_funder],
                                   current=taken[working_funder], 
                                   encountered=encountered[api_funder],
                                   from_excel=excel_funder is not None)
                
                if taken[working_funder] % 25 == 1 or taken[working_funder] == targets[working_funder]:
                    status = "DONE" if taken[working_funder] >= targets[working_funder] else "IN PROGRESS"
                    print(f"  {status} {working_funder}: {taken[working_funder]}/{targets[working_funder]}")
        
        conn.commit()
        
        if page % 10 == 0:
            save_progress(cur, page, taken)
            conn.commit()
            total_sampled = sum(taken.values())
            completed_funders = sum(1 for f in targets if taken[f] >= targets[f])
            print(f"Page {page} complete. Added {page_added} projects on this page.")
            print(f"   Total sampled: {total_sampled} projects. {completed_funders}/{len(targets)} funders have met their quota.")
        
        important_funders = [f for f, target in targets.items() if target > MIN_KEEP]
        if all(taken[f] >= targets[f] for f in important_funders):
            remaining = [f for f in targets if taken[f] < targets[f]]
            if len(remaining) <= 5:
                print(f"Major funders complete. Remaining: {remaining}")
        
        page += 1
        
        if delay[0] > 0 or delay[1] > 0:
            time.sleep(random.uniform(*delay))
    
    save_progress(cur, page - 1, taken)
    conn.commit()
    
    total_projects = sum(taken.values())
    print(f"\nFinished! Sampled a total of {total_projects} projects.")
    
    print("\n=== Sampling Results Summary ===")
    for funder, count in taken.most_common():
        target = targets[funder]
        status = "MET" if count >= target else "UNMET"
        source = "(Excel)" if funder in excel_funders else "(API)"
        print(f"  {status} {funder}: {count}/{target} {source}")
    
    print(f"\n=== Funder Coverage ===")
    excel_funders_set = set(excel_funders.keys())
    matched_funders = set()
    
    for api_name in encountered.keys():
        normalized = normalize_funder_name(api_name)
        if normalized in funder_mapping:
            matched_funders.add(funder_mapping[normalized])
    
    print(f"Total funders in Excel: {len(excel_funders_set)}")
    print(f"Total unique funders found in API: {len(encountered)}")
    print(f"Successfully matched Excel funders: {len(matched_funders)}")
    print(f"Funders with at least one sampled project: {len(taken)}")
    
    matched_but_not_sampled = matched_funders - set(taken.keys())
    if matched_but_not_sampled:
        print(f"\nMatched funders with no sampled projects ({len(matched_but_not_sampled)}):")
        for excel_funder in sorted(matched_but_not_sampled):
            api_names = []
            for api_name in encountered.keys():
                if funder_mapping.get(normalize_funder_name(api_name)) == excel_funder:
                    api_names.append(api_name)
            
            if api_names:
                api_name = api_names[0]
                count = encountered[api_name]
                print(f"  - {excel_funder} (API name: {api_name}): Encountered {count} projects, but none had outcomes.")
    
    unmatched_excel = excel_funders_set - matched_funders
    if unmatched_excel:
        print(f"\nUnmatched funders from Excel ({len(unmatched_excel)}):")
        for funder in sorted(unmatched_excel):
            print(f"  - {funder}: No corresponding projects found in the API.")
    
    unmatched_api = []
    for api_name in encountered.keys():
        normalized = normalize_funder_name(api_name)
        if normalized not in funder_mapping and api_name not in excel_funders_set:
            unmatched_api.append(api_name)
    
    if unmatched_api:
        print(f"\nAdditional funders found in API ({len(unmatched_api)}):")
        for api_name in sorted(unmatched_api):
            count = encountered[api_name]
            taken_count = taken.get(api_name, 0)
            print(f"  + {api_name}: Encountered {count} projects, sampled {taken_count} with outcomes.")
    
    conn.close()

def show_status():
    """Displays the current sampling status."""
    if not os.path.exists(DB_PATH):
        print("Database file not found. Please run the sampling script first.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM projects")
    total_projects = cur.fetchone()[0]
    
    cur.execute("SELECT funder, COUNT(*) FROM projects GROUP BY funder ORDER BY COUNT(*) DESC")
    results = cur.fetchall()
    
    print(f"Current Database Status:")
    print(f"   Total Projects: {total_projects}")
    print(f"   Number of Funders: {len(results)}")
    print()
    
    print("=== Projects per Funder ===")
    for funder, count in results:
        cur.execute("SELECT target_count, from_excel FROM funder_stats WHERE funder = ?", (funder,))
        stats = cur.fetchone()
        if stats:
            target, from_excel = stats
            source = "(Excel)" if from_excel else "(API)"
            status = "MET" if count >= target else "IN PROGRESS"
            print(f"  {status} {funder}: {count}/{target} {source}")
        else:
            print(f"    {funder}: {count}")
    
    cur.execute("SELECT last_page, timestamp FROM sampling_progress ORDER BY timestamp DESC LIMIT 1")
    progress = cur.fetchone()
    if progress:
        last_page, timestamp = progress
        print(f"\nLast progress saved at page: {last_page} ({timestamp})")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTR Project Sampling with Resume Support")
    parser.add_argument("--counts", help="Path to the funder counts file (.xlsx/.csv)")
    parser.add_argument("--sep", help="Delimiter for CSV files (e.g., '\\t')")
    parser.add_argument("--delay", nargs=2, type=float, default=[0.5, 1.0], 
                       help="Request delay range in seconds (e.g., --delay 0.5 1.5)")
    parser.add_argument("--resume", action="store_true", help="Resume from the last session")
    parser.add_argument("--status", action="store_true", help="Show current sampling status")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.counts:
        sample_projects(args.counts, args.sep, tuple(args.delay), args.resume)
    else:
        print("Please provide the --counts argument or use --status to check the current state.")
        parser.print_help()