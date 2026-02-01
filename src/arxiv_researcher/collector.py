"""
arXiv Paper Collector
=====================

Collects papers from arXiv and enriches with data from external sources.
"""

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import re


# API Endpoints
ARXIV_API = "http://export.arxiv.org/api/query"
S2_GRAPH_API = "https://api.semanticscholar.org/graph/v1"

# CS Categories
CS_CATEGORIES = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

ALL_CATEGORIES = CS_CATEGORIES + [
    "stat.ML", "stat.TH", "stat.ME", "stat.AP", "stat.CO",
    "math.OC", "math.ST", "math.NA", "math.CO", "math.PR",
    "quant-ph", "physics.comp-ph", "physics.data-an",
    "eess.SP", "eess.IV", "eess.AS", "eess.SY",
]


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self) -> None:
        self.last_call: Dict[str, float] = {}

    def wait(self, api_name: str, min_interval: float) -> None:
        """Wait if needed to respect rate limits."""
        now = time.time()
        if api_name in self.last_call:
            elapsed = now - self.last_call[api_name]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_call[api_name] = time.time()


class ArxivCollector:
    """Collector for arXiv papers with multi-source enrichment."""

    def __init__(
        self,
        output_dir: str = "arxiv_data",
        skip_s2: bool = False,
        skip_openalex: bool = True,
        skip_pwc: bool = True,
    ) -> None:
        """Initialize the collector.

        Args:
            output_dir: Directory to save collected data.
            skip_s2: Skip Semantic Scholar enrichment.
            skip_openalex: Skip OpenAlex enrichment.
            skip_pwc: Skip Papers With Code enrichment.
        """
        self.output_dir = output_dir
        self.skip_s2 = skip_s2
        self.skip_openalex = skip_openalex
        self.skip_pwc = skip_pwc
        self.rate_limiter = RateLimiter()

        os.makedirs(output_dir, exist_ok=True)

    def collect(
        self,
        categories: Optional[Union[List[str], str]] = None,
        max_results: int = 500,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
        """Collect papers from arXiv.

        Args:
            categories: List of arXiv categories or "all" for all categories.
            max_results: Maximum papers per category.
            start_date: Start of date range.
            end_date: End of date range.
            resume: Resume from checkpoint if available.

        Returns:
            List of paper dictionaries.
        """
        # Set defaults
        if categories is None:
            categories = CS_CATEGORIES
        elif categories == "all":
            categories = ALL_CATEGORIES

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        # Check for checkpoint
        all_papers: List[Dict[str, Any]] = []
        completed_categories: List[str] = []
        checkpoint_file = os.path.join(self.output_dir, "checkpoint.json")

        if resume and os.path.exists(checkpoint_file):
            print(f"Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            all_papers = checkpoint.get("papers", [])
            completed_categories = checkpoint.get("completed_categories", [])
            categories = [c for c in categories if c not in completed_categories]
            print(f"  Loaded {len(all_papers):,} papers, {len(completed_categories)} categories done")

        # Generate date chunks for large ranges
        total_days = (end_date - start_date).days
        if total_days > 730:  # More than 2 years
            date_chunks = self._generate_date_chunks(start_date, end_date, months=24)
            per_chunk_limit = min(500, max(100, max_results // len(date_chunks)))
        else:
            date_chunks = [(start_date, end_date)]
            per_chunk_limit = max_results

        print(f"\nCollecting from {len(categories)} categories")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        if len(date_chunks) > 1:
            print(f"Split into {len(date_chunks)} time chunks")

        # Collect from each category
        for i, category in enumerate(categories, 1):
            print(f"\n[{i}/{len(categories)}] {category}")
            category_papers: List[Dict[str, Any]] = []

            for chunk_start, chunk_end in date_chunks:
                papers = self._fetch_arxiv_papers(
                    category=category,
                    start_date=chunk_start.strftime("%Y%m%d"),
                    end_date=chunk_end.strftime("%Y%m%d"),
                    max_results=per_chunk_limit,
                )
                category_papers.extend(papers)

                if len(category_papers) >= max_results:
                    category_papers = category_papers[:max_results]
                    break

            print(f"  Found {len(category_papers)} papers")

            # Enrich with Semantic Scholar
            if not self.skip_s2 and category_papers:
                print("  Enriching with Semantic Scholar...")
                category_papers = self._enrich_with_s2(category_papers)

            all_papers.extend(category_papers)

            # Save checkpoint
            completed_categories.append(category)
            self._save_checkpoint(all_papers, completed_categories)

        # Deduplicate
        seen = set()
        unique_papers = []
        for p in all_papers:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                unique_papers.append(p)

        # Save final results
        self._save_results(unique_papers)

        return unique_papers

    def _generate_date_chunks(
        self, start_date: datetime, end_date: datetime, months: int = 24
    ) -> List[tuple]:
        """Generate date range chunks."""
        chunks = []
        current = start_date
        while current < end_date:
            chunk_end = current + timedelta(days=months * 30)
            if chunk_end > end_date:
                chunk_end = end_date
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks

    def _fetch_arxiv_papers(
        self,
        category: str,
        start_date: str,
        end_date: str,
        max_results: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv API."""
        papers = []
        batch_size = 100
        start_index = 0

        date_query = f"submittedDate:[{start_date}0000 TO {end_date}2359]"
        search_query = f"cat:{category} AND {date_query}"

        while start_index < max_results:
            params = {
                "search_query": search_query,
                "start": start_index,
                "max_results": min(batch_size, max_results - start_index),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

            for retry in range(5):
                try:
                    self.rate_limiter.wait("arxiv", 5.0)
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "arxiv-researcher/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        xml_data = response.read().decode("utf-8")

                    root = ET.fromstring(xml_data)
                    ns = {
                        "atom": "http://www.w3.org/2005/Atom",
                        "arxiv": "http://arxiv.org/schemas/atom",
                    }

                    entries = root.findall("atom:entry", ns)
                    if not entries:
                        return papers

                    for entry in entries:
                        paper = self._parse_arxiv_entry(entry, ns, category)
                        if paper:
                            papers.append(paper)

                    start_index += batch_size
                    break

                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        wait_time = 10 * (2**retry)
                        print(f"    Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break
                except Exception as e:
                    if retry == 4:
                        print(f"    Error: {e}")
                    time.sleep(10 * (2**retry))

        return papers

    def _parse_arxiv_entry(
        self, entry: ET.Element, ns: dict, default_category: str
    ) -> Optional[Dict[str, Any]]:
        """Parse an arXiv XML entry into a paper dict."""
        id_elem = entry.find("atom:id", ns)
        paper_id = id_elem.text.split("/abs/")[-1] if id_elem is not None else ""

        title_elem = entry.find("atom:title", ns)
        title = re.sub(r"\s+", " ", title_elem.text.strip()) if title_elem is not None else ""

        summary = entry.find("atom:summary", ns)
        abstract = re.sub(r"\s+", " ", summary.text.strip()) if summary is not None else ""

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text)

        published = entry.find("atom:published", ns)
        pub_date = published.text[:10] if published is not None else ""

        categories = [
            cat.get("term")
            for cat in entry.findall("atom:category", ns)
            if cat.get("term")
        ]

        primary_cat = entry.find("arxiv:primary_category", ns)
        primary_category = (
            primary_cat.get("term") if primary_cat is not None else default_category
        )

        return {
            "arxiv_id": paper_id.split("v")[0] if "v" in paper_id else paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published_date": pub_date,
            "primary_category": primary_category,
            "categories": categories,
            "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
        }

    def _enrich_with_s2(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich papers with Semantic Scholar data."""
        batch_size = 100

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            arxiv_ids = [f"ARXIV:{p['arxiv_id']}" for p in batch]

            try:
                self.rate_limiter.wait("s2", 1.0)

                data = json.dumps({"ids": arxiv_ids}).encode("utf-8")
                req = urllib.request.Request(
                    f"{S2_GRAPH_API}/paper/batch",
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "arxiv-researcher/1.0",
                    },
                    method="POST",
                )

                params = urllib.parse.urlencode(
                    {
                        "fields": "citationCount,influentialCitationCount,referenceCount,fieldsOfStudy,tldr,venue,year"
                    }
                )
                req.full_url = f"{S2_GRAPH_API}/paper/batch?{params}"

                with urllib.request.urlopen(req, timeout=30) as response:
                    results = json.loads(response.read().decode("utf-8"))

                for paper, result in zip(batch, results):
                    if result:
                        paper["citation_count"] = result.get("citationCount", 0)
                        paper["influential_citations"] = result.get(
                            "influentialCitationCount", 0
                        )
                        paper["reference_count"] = result.get("referenceCount", 0)
                        paper["fields_of_study"] = result.get("fieldsOfStudy", [])
                        paper["venue"] = result.get("venue", "")
                        tldr = result.get("tldr")
                        paper["tldr"] = tldr.get("text", "") if tldr else ""

            except Exception as e:
                print(f"    S2 batch error: {e}")

        return papers

    def _save_checkpoint(
        self, papers: List[Dict[str, Any]], completed_categories: List[str]
    ) -> None:
        """Save checkpoint file."""
        checkpoint_file = os.path.join(self.output_dir, "checkpoint.json")
        checkpoint = {
            "papers": papers,
            "completed_categories": completed_categories,
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f)

    def _save_results(self, papers: List[Dict[str, Any]]) -> None:
        """Save final results to JSON and CSV."""
        # JSON
        json_file = os.path.join(self.output_dir, "papers.json")
        with open(json_file, "w") as f:
            json.dump(papers, f, indent=2)

        # CSV
        try:
            import pandas as pd

            df = pd.DataFrame(papers)
            csv_file = os.path.join(self.output_dir, "papers.csv")
            df.to_csv(csv_file, index=False)
        except ImportError:
            pass

        print(f"\nSaved {len(papers):,} papers to {self.output_dir}/")
