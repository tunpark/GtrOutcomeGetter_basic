import argparse
import math
import os
import random
import re
import time
import sqlite3
import requests
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import threading
from queue import Queue, Empty

API_NS = "http://gtr.rcuk.ac.uk/gtr/api"
HEADERS = {"Accept": "application/xml", "User-Agent": "Mozilla/5.0"}

DB_PATH = "project_outcomes.db"

# API limit configurations
MAX_REQUESTS_PER_MINUTE = 60
MAX_CONCURRENT_REQUESTS = 3
BACKOFF_MULTIPLIER = 2
MAX_BACKOFF_TIME = 300  # seconds

# Explicitly define the 7 required outcome types
OUTCOME_PATHS = {
    "artisticandcreativeproducts": "ARTISTIC_AND_CREATIVE_PRODUCT",
    "keyfindings": "KEY_FINDINGS",
    "researchdatabaseandmodels": "RESEARCH_DATABASES_AND_MODELS",
    "softwareandtechnicalproducts": "SOFTWARE_AND_TECHNICAL_PRODUCT",
    "researchmaterials": "RESEARCH_TOOLS_AND_METHODS",
    "disseminations": "ENGAGEMENT_ACTIVITY",
    "intellectualproperties": "INTELLECTUAL_PROPERTY"
}

class RateLimiter:
    """API request rate limiter."""
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Waits if necessary until a request can be sent."""
        with self.lock:
            now = time.time()
            # Clean up request records older than one minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                # Need to wait
                sleep_time = 60 - (now - self.requests[0]) + 1
                print(f"Rate limit reached, waiting for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.requests = []

            self.requests.append(now)

class OutcomeFetcher:
    """Fetches project outcomes."""
    def __init__(self, db_path: str, max_concurrent: int = 3):
        self.db_path = db_path
        self.rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)
        self.max_concurrent = max_concurrent
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        self.stats = {
            'total_projects': 0,
            'processed_projects': 0,
            'total_outcomes': 0,
            'failed_requests': 0,
            'rate_limited': 0,
            'by_type': defaultdict(int),
            'skipped_types': defaultdict(int)
        }

        self.retry_config = {
            'max_retries': 5,
            'initial_backoff': 1,
            'max_backoff': MAX_BACKOFF_TIME,
            'backoff_multiplier': BACKOFF_MULTIPLIER
        }

    def get_attr(self, elem: ET.Element, name: str):
        """Gets element attribute, compatible with different namespace formats."""
        return elem.attrib.get(name) or elem.attrib.get(f"{{{API_NS}}}{name}")

    def safe_get(self, url: str, retries: int = None) -> Optional[requests.Response]:
        """Safe HTTP request with retries and rate limiting."""
        if retries is None:
            retries = self.retry_config['max_retries']

        backoff = self.retry_config['initial_backoff']

        for attempt in range(retries):
            try:
                self.rate_limiter.wait_if_needed()
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    self.stats['rate_limited'] += 1
                    retry_after = int(response.headers.get('Retry-After', backoff))
                    print(f"API rate limit hit, waiting {retry_after} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(retry_after)
                    backoff = min(backoff * self.retry_config['backoff_multiplier'], self.retry_config['max_backoff'])
                elif response.status_code == 404:
                    print(f"Resource not found: {url}")
                    return None
                elif response.status_code >= 500:
                    print(f"Server error {response.status_code}, retrying... (Attempt {attempt + 1}/{retries})")
                    time.sleep(backoff)
                    backoff = min(backoff * self.retry_config['backoff_multiplier'], self.retry_config['max_backoff'])
                else:
                    print(f"HTTP error {response.status_code}: {url}")
                    return None

            except requests.RequestException as e:
                print(f"Request exception: {e} (Attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * self.retry_config['backoff_multiplier'], self.retry_config['max_backoff'])

        self.stats['failed_requests'] += 1
        return None

    def get_project_outcomes_links(self, project_url: str) -> List[Tuple[str, str]]:
        """Gets all outcome links for a project (limited to the 7 specified types)."""
        outcome_links = []
        response = self.safe_get(project_url)
        if not response:
            return outcome_links

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return outcome_links

        ns = {'ns1': API_NS, 'ns2': 'http://gtr.rcuk.ac.uk/gtr/api/project'}
        links = root.find('ns1:links', ns)

        if links is not None:
            for link in links.findall('ns1:link', ns):
                href = self.get_attr(link, 'href')
                if href and "/outcomes/" in href:
                    outcome_type = self.identify_outcome_type(href)
                    if outcome_type:
                        outcome_links.append((href, outcome_type))
                    else:
                        skipped_type = self.extract_type_from_url(href)
                        if skipped_type:
                            self.stats['skipped_types'][skipped_type] += 1
        
        return outcome_links

    def extract_type_from_url(self, href: str) -> Optional[str]:
        """Extracts the outcome type from a URL (for tracking skipped types)."""
        match = re.search(r'/outcomes/([^/]+)/', href)
        return match.group(1) if match else None

    def identify_outcome_type(self, href: str) -> Optional[str]:
        """Identifies the outcome type (only returns one of the 7 required types)."""
        for path, type_name in OUTCOME_PATHS.items():
            if f"/outcomes/{path}/" in href:
                return type_name
        return None

    def parse_outcome_detail(self, project_id: str, outcome_url: str, outcome_type: str) -> bool:
        """Parses outcome details and stores them in the database."""
        response = self.safe_get(outcome_url)
        if not response:
            return False

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            print(f"Outcome XML parse error: {e}")
            return False

        outcome_id = outcome_url.split("/")[-1]
        ns = {'ns2': 'http://gtr.rcuk.ac.uk/gtr/api/project/outcome'}
        
        title_elem = root.find('ns2:title', ns)
        desc_elem = root.find('ns2:description', ns)
        url_elem = root.find('ns2:supportingUrl', ns)
        prot_elem = root.find('ns2:protection', ns)
        specific_type_elem = root.find('ns2:type', ns)

        title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
        description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else None
        support_url = url_elem.text.strip() if url_elem is not None and url_elem.text else None
        protection = prot_elem.text.strip() if prot_elem is not None and prot_elem.text else None
        specific_type = specific_type_elem.text.strip() if specific_type_elem is not None and specific_type_elem.text else outcome_type

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT OR REPLACE INTO outcomes 
                (outcome_id, project_id, type, outcome_type, title, description, 
                 support_url, protection, api_url, created_at) 
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                outcome_id, project_id, outcome_type, specific_type, title, description,
                support_url, protection, outcome_url, datetime.now().isoformat()
            ))
            conn.commit()
            self.stats['total_outcomes'] += 1
            self.stats['by_type'][outcome_type] += 1
            return True
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def setup_database(self):
        """Sets up the database table structure."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("PRAGMA table_info(projects)")
        existing_columns = [row[1] for row in cur.fetchall()]

        cur.executescript("""
            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                type TEXT,
                outcome_type TEXT,
                title TEXT,
                description TEXT,
                support_url TEXT,
                protection TEXT,
                api_url TEXT,
                created_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS processing_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL, -- 'started', 'completed', 'failed'
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_outcomes_project_id ON outcomes(project_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_type ON outcomes(type);
            CREATE INDEX IF NOT EXISTS idx_processing_status_project ON processing_status(project_id);
        """)

        if 'processed_at' not in existing_columns:
            cur.execute("ALTER TABLE projects ADD COLUMN processed_at TIMESTAMP")
        if 'outcome_count' not in existing_columns:
            cur.execute("ALTER TABLE projects ADD COLUMN outcome_count INTEGER DEFAULT 0")
        if 'last_error' not in existing_columns:
            cur.execute("ALTER TABLE projects ADD COLUMN last_error TEXT")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_processed ON projects(processed_at)")
        
        conn.commit()
        conn.close()

    def get_unprocessed_projects(self, limit: int = None) -> List[Tuple[str, str]]:
        """Gets a list of unprocessed projects."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("PRAGMA table_info(projects)")
            columns = [row[1] for row in cursor.fetchall()]
            query = "SELECT id, api_url FROM projects WHERE processed_at IS NULL ORDER BY id" if 'processed_at' in columns else "SELECT id, api_url FROM projects ORDER BY id"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            projects = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error: {e}. Falling back to selecting all projects.")
            query = "SELECT id, api_url FROM projects ORDER BY id"
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
            projects = cursor.fetchall()

        conn.close()
        return projects

    def mark_project_processed(self, project_id: str, outcome_count: int, error: str = None):
        """Marks a project as processed."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        if error:
            cur.execute("UPDATE projects SET processed_at = ?, outcome_count = ?, last_error = ? WHERE id = ?", 
                        (datetime.now().isoformat(), outcome_count, error, project_id))
            cur.execute("INSERT INTO processing_status (project_id, status, notes) VALUES (?, 'failed', ?)", 
                        (project_id, error))
        else:
            cur.execute("UPDATE projects SET processed_at = ?, outcome_count = ?, last_error = NULL WHERE id = ?", 
                        (datetime.now().isoformat(), outcome_count, project_id))
            cur.execute("INSERT INTO processing_status (project_id, status, notes) VALUES (?, 'completed', ?)", 
                        (project_id, f"Processed {outcome_count} outcomes"))
        
        conn.commit()
        conn.close()

    def get_processing_summary(self) -> dict:
        """Gets a summary of the processing status."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM projects")
        total_projects = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM projects WHERE processed_at IS NOT NULL")
        processed_projects = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM outcomes")
        total_outcomes = cur.fetchone()[0]
        cur.execute("SELECT type, COUNT(*) FROM outcomes GROUP BY type ORDER BY COUNT(*) DESC")
        by_type = dict(cur.fetchall())
        cur.execute("SELECT COUNT(*) FROM projects WHERE last_error IS NOT NULL")
        failed_projects = cur.fetchone()[0]
        
        conn.close()
        
        return {
            'total_projects': total_projects,
            'processed_projects': processed_projects,
            'remaining_projects': total_projects - processed_projects,
            'total_outcomes': total_outcomes,
            'failed_projects': failed_projects,
            'by_type': by_type
        }

    def fetch_outcomes_for_projects(self, limit: int = None, resume: bool = True):
        """Fetches outcomes for projects."""
        self.setup_database()
        
        if resume:
            projects = self.get_unprocessed_projects(limit)
            print(f"Found {len(projects)} unprocessed projects.")
            summary = self.get_processing_summary()
            print(f"Current status: {summary['processed_projects']}/{summary['total_projects']} projects processed.")
            print(f"Collected outcomes: {summary['total_outcomes']}")
            if summary['by_type']:
                print("Distribution by type:")
                for outcome_type, count in summary['by_type'].items():
                    print(f"   {outcome_type}: {count}")
        else:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("UPDATE projects SET processed_at = NULL, outcome_count = 0, last_error = NULL")
            cur.execute("DELETE FROM processing_status")
            cur.execute("DELETE FROM outcomes")
            conn.commit()
            conn.close()
            projects = self.get_unprocessed_projects(limit)
            print(f"Reprocessing {len(projects)} projects.")
        
        if not projects:
            print("No projects to process.")
            summary = self.get_processing_summary()
            self.print_summary_stats(summary)
            return
        
        self.stats['total_projects'] = len(projects)
        
        for i, (project_id, project_url) in enumerate(projects, 1):
            print(f"\nProcessing project {i}/{len(projects)}: {project_id}")
            
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute("INSERT INTO processing_status (project_id, status, notes) VALUES (?, 'started', ?)",
                            (project_id, f"Starting processing {i}/{len(projects)}"))
                conn.commit()
                conn.close()
                
                outcome_links = self.get_project_outcomes_links(project_url)
                
                if not outcome_links:
                    print(f"  Project {project_id} has no target outcomes.")
                    self.mark_project_processed(project_id, 0)
                    continue
                
                print(f"  Found {len(outcome_links)} target outcomes.")
                
                processed_outcomes = 0
                for outcome_url, outcome_type in outcome_links:
                    print(f"    Fetching {outcome_type}: {outcome_url.split('/')[-1]}")
                    if self.parse_outcome_detail(project_id, outcome_url, outcome_type):
                        processed_outcomes += 1
                    time.sleep(0.1)
                
                self.mark_project_processed(project_id, processed_outcomes)
                self.stats['processed_projects'] += 1
                print(f"  Finished, processed {processed_outcomes} outcomes.")
                
                if i % 10 == 0:
                    self.print_progress_stats()
                    
            except KeyboardInterrupt:
                print("\nUser interrupted, saving current progress...")
                self.mark_project_processed(project_id, 0, "User interrupted")
                break
            except Exception as e:
                error_msg = f"Processing exception: {str(e)}"
                print(f"  Error processing project {project_id}: {e}")
                self.mark_project_processed(project_id, 0, error_msg)
                continue
        
        self.print_final_stats()

    def print_progress_stats(self):
        """Prints progress statistics."""
        print("\nProgress Stats:")
        print(f"  Processed projects: {self.stats['processed_projects']}/{self.stats['total_projects']}")
        print(f"  Total outcomes: {self.stats['total_outcomes']}")
        print(f"  Failed requests: {self.stats['failed_requests']}")
        print(f"  Rate limited count: {self.stats['rate_limited']}")
        
        if self.stats['skipped_types']:
            print("  Skipped types:")
            for skipped_type, count in self.stats['skipped_types'].items():
                print(f"    {skipped_type}: {count}")

    def print_final_stats(self):
        """Prints final statistics."""
        print("\nFinished! Final Stats:")
        print(f"  Processed projects: {self.stats['processed_projects']}/{self.stats['total_projects']}")
        print(f"  Total outcomes: {self.stats['total_outcomes']}")
        print(f"  Failed requests: {self.stats['failed_requests']}")
        print(f"  Rate limited count: {self.stats['rate_limited']}")
        
        print("\nCollected outcome distribution (7 target types):")
        for outcome_type, count in sorted(self.stats['by_type'].items()):
            print(f"  {outcome_type}: {count}")
        
        if self.stats['skipped_types']:
            print("\nOther skipped types:")
            for skipped_type, count in sorted(self.stats['skipped_types'].items()):
                print(f"  {skipped_type}: {count}")

    def print_summary_stats(self, summary: dict):
        """Prints summary statistics."""
        print("\nDatabase Overall Status:")
        print(f"  Total projects: {summary['total_projects']}")
        print(f"  Processed: {summary['processed_projects']}")
        print(f"  Remaining: {summary['remaining_projects']}")
        print(f"  Failed: {summary['failed_projects']}")
        print(f"  Total outcomes: {summary['total_outcomes']}")
        
        if summary['by_type']:
            print("\nOutcomes distribution by type:")
            for outcome_type, count in summary['by_type'].items():
                print(f"  {outcome_type}: {count}")

def main():
    global MAX_REQUESTS_PER_MINUTE, MAX_CONCURRENT_REQUESTS
    
    parser = argparse.ArgumentParser(description="Fetch details for 7 specific types of GTR project outcomes.")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    parser.add_argument("--limit", type=int, help="Limit the number of projects to process")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from previous progress, start over")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, help="Maximum concurrent requests")
    parser.add_argument("--rate-limit", type=int, default=MAX_REQUESTS_PER_MINUTE, help="Maximum requests per minute")
    parser.add_argument("--status", action="store_true", help="Only display the current processing status")
    
    args = parser.parse_args()
    
    MAX_REQUESTS_PER_MINUTE = args.rate_limit
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    
    fetcher = OutcomeFetcher(args.db, args.max_concurrent)
    
    if args.status:
        fetcher.setup_database()
        summary = fetcher.get_processing_summary()
        fetcher.print_summary_stats(summary)
        return
    
    fetcher.fetch_outcomes_for_projects(args.limit, not args.no_resume)

if __name__ == "__main__":
    main()
