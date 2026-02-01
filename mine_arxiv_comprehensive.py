#!/usr/bin/env python3
"""
Comprehensive arXiv Research Mining System
==========================================
Multi-source data collection for research analysis and opportunity identification.

Data Sources:
- arXiv API: Core paper metadata
- Semantic Scholar: Citations, references, author metrics, related papers
- OpenAlex: Institutions, funders, concepts, citation trends
- Papers With Code: GitHub repos, datasets, benchmarks, SOTA

Computed Metrics:
- Citation velocity and trajectory
- Author influence scores
- Topic trends and saturation
- Cross-disciplinary potential
- Research gap indicators

Usage:
    uv run python mine_arxiv_comprehensive.py [options]
"""

import argparse
import csv
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional, Dict, List, Any, Tuple
import os
import re
import hashlib

# =============================================================================
# API ENDPOINTS
# =============================================================================

ARXIV_API = "http://export.arxiv.org/api/query"
S2_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
S2_BATCH_API = f"{S2_GRAPH_API}/paper/batch"
S2_AUTHOR_API = f"{S2_GRAPH_API}/author"
OPENALEX_API = "https://api.openalex.org"
PAPERS_WITH_CODE_API = "https://paperswithcode.com/api/v1"

# =============================================================================
# CATEGORIES
# =============================================================================

CS_CATEGORIES = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

ML_STATS_CATEGORIES = [
    "stat.ML", "stat.TH", "stat.ME", "stat.AP", "stat.CO"
]

MATH_CATEGORIES = [
    "math.OC", "math.ST", "math.NA", "math.CO", "math.PR"
]

PHYSICS_CATEGORIES = [
    "quant-ph", "physics.comp-ph", "physics.data-an", "cond-mat.dis-nn"
]

OTHER_CATEGORIES = [
    "eess.SP", "eess.IV", "eess.AS", "eess.SY",
    "q-bio.QM", "q-bio.NC", "q-bio.BM",
    "econ.EM", "econ.TH", "econ.GN"
]

ALL_CATEGORIES = CS_CATEGORIES + ML_STATS_CATEGORIES + MATH_CATEGORIES + PHYSICS_CATEGORIES + OTHER_CATEGORIES


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self):
        self.last_call = {}

    def wait(self, api_name: str, min_interval: float):
        """Wait if needed to respect rate limits."""
        now = time.time()
        if api_name in self.last_call:
            elapsed = now - self.last_call[api_name]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_call[api_name] = time.time()


RATE_LIMITER = RateLimiter()


# =============================================================================
# ARXIV DATA COLLECTION
# =============================================================================

def fetch_arxiv_papers(category: str, start_date: str, end_date: str,
                       max_results: int = 500, rate_limit: float = 3.0) -> List[Dict]:
    """Fetch papers from arXiv with full metadata and robust retry logic.

    Args:
        category: arXiv category (e.g., "cs.AI")
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        max_results: Maximum papers to fetch
        rate_limit: Seconds between API calls
    """
    papers = []
    batch_size = 100
    start_index = 0
    max_retries = 5
    base_backoff = 10  # Base backoff in seconds

    # Build search query with date range
    # arXiv uses submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM] format
    date_query = f"submittedDate:[{start_date}0000 TO {end_date}2359]"
    search_query = f"cat:{category} AND {date_query}"

    while start_index < max_results:
        params = {
            "search_query": search_query,
            "start": start_index,
            "max_results": min(batch_size, max_results - start_index),
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

        success = False
        for retry in range(max_retries):
            try:
                RATE_LIMITER.wait("arxiv", rate_limit)

                req = urllib.request.Request(url, headers={"User-Agent": "research-miner/3.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    xml_data = response.read().decode("utf-8")

                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

                entries = root.findall("atom:entry", ns)
                if not entries:
                    # No more entries, we're done
                    return papers

                for entry in entries:
                    paper = extract_arxiv_entry(entry, ns, start_date, end_date, category)
                    if paper:
                        papers.append(paper)

                start_index += batch_size
                print(f"      Fetched {min(start_index, max_results)}/{max_results}...")
                success = True
                break  # Success, exit retry loop

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # Rate limited - exponential backoff
                    wait_time = base_backoff * (2 ** retry)
                    print(f"      Rate limited (429). Waiting {wait_time}s... (attempt {retry+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"      HTTP error {e.code}: {e.reason}")
                    if retry == max_retries - 1:
                        break
                    wait_time = base_backoff * (2 ** retry)
                    time.sleep(wait_time)

            except Exception as e:
                error_msg = str(e)
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
                    wait_time = base_backoff * (2 ** retry)
                    print(f"      Timeout. Waiting {wait_time}s... (attempt {retry+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"      arXiv error: {e}")
                    if retry == max_retries - 1:
                        break
                    wait_time = base_backoff
                    time.sleep(wait_time)

        if not success:
            print(f"      Failed after {max_retries} retries. Moving to next batch...")
            start_index += batch_size  # Skip this batch and continue

    return papers


def extract_arxiv_entry(entry, ns: dict, start_date: str, end_date: str,
                        default_category: str) -> Optional[Dict]:
    """Extract comprehensive metadata from arXiv entry."""

    # Basic extraction
    id_elem = entry.find("atom:id", ns)
    paper_id = id_elem.text.split("/abs/")[-1] if id_elem is not None else ""

    version_match = re.search(r'v(\d+)$', paper_id)
    version = int(version_match.group(1)) if version_match else 1
    base_id = paper_id.split("v")[0] if "v" in paper_id else paper_id

    # Dates
    published = entry.find("atom:published", ns)
    pub_date = published.text[:10] if published is not None else ""

    if pub_date:
        pub_date_clean = pub_date.replace("-", "")
        if pub_date_clean < start_date or pub_date_clean > end_date:
            return None

    updated = entry.find("atom:updated", ns)
    updated_date = updated.text[:10] if updated is not None else ""

    # Title and abstract
    title_elem = entry.find("atom:title", ns)
    title = re.sub(r'\s+', ' ', title_elem.text.strip()) if title_elem is not None else ""

    summary = entry.find("atom:summary", ns)
    abstract = re.sub(r'\s+', ' ', summary.text.strip()) if summary is not None else ""

    # Authors
    authors = []
    author_affiliations = []
    for author in entry.findall("atom:author", ns):
        name = author.find("atom:name", ns)
        if name is not None:
            authors.append(name.text)
        affil = author.find("arxiv:affiliation", ns)
        if affil is not None and affil.text:
            author_affiliations.append(affil.text)

    # Categories
    primary_cat = entry.find("arxiv:primary_category", ns)
    primary_category = primary_cat.get("term") if primary_cat is not None else default_category

    categories = [cat.get("term") for cat in entry.findall("atom:category", ns) if cat.get("term")]

    # Additional metadata
    comment = entry.find("arxiv:comment", ns)
    arxiv_comment = comment.text.strip() if comment is not None and comment.text else ""

    journal_ref = entry.find("arxiv:journal_ref", ns)
    journal_reference = journal_ref.text.strip() if journal_ref is not None and journal_ref.text else ""

    doi_elem = entry.find("arxiv:doi", ns)
    doi = doi_elem.text if doi_elem is not None else ""

    # Extract useful info from comment
    page_count = None
    figure_count = None
    table_count = None
    accepted_venue = ""

    if arxiv_comment:
        # Pages
        page_match = re.search(r'(\d+)\s*pages?', arxiv_comment, re.I)
        if page_match:
            page_count = int(page_match.group(1))

        # Figures
        fig_match = re.search(r'(\d+)\s*figures?', arxiv_comment, re.I)
        if fig_match:
            figure_count = int(fig_match.group(1))

        # Tables
        table_match = re.search(r'(\d+)\s*tables?', arxiv_comment, re.I)
        if table_match:
            table_count = int(table_match.group(1))

        # Accepted venue
        venue_patterns = [
            r'accepted\s+(?:at|to|by|in)\s+([A-Z][A-Za-z0-9\s\-\']+(?:\s+\d{4})?)',
            r'to\s+appear\s+(?:at|in)\s+([A-Z][A-Za-z0-9\s\-\']+(?:\s+\d{4})?)',
            r'published\s+(?:at|in)\s+([A-Z][A-Za-z0-9\s\-\']+(?:\s+\d{4})?)',
        ]
        for pattern in venue_patterns:
            match = re.search(pattern, arxiv_comment, re.I)
            if match:
                accepted_venue = match.group(1).strip()
                break

    # Paper age
    paper_age_days = None
    if pub_date:
        try:
            pub_dt = datetime.strptime(pub_date, "%Y-%m-%d")
            paper_age_days = (datetime.now() - pub_dt).days
        except ValueError:
            pass

    return {
        # Identifiers
        "arxiv_id": paper_id,
        "arxiv_id_base": base_id,
        "version": version,

        # Core metadata
        "title": title,
        "abstract": abstract,
        "abstract_word_count": len(abstract.split()),
        "title_word_count": len(title.split()),

        # Authors
        "authors": "; ".join(authors),
        "author_list": authors,  # Keep as list for processing
        "author_count": len(authors),
        "first_author": authors[0] if authors else "",
        "last_author": authors[-1] if authors else "",
        "author_affiliations": "; ".join(author_affiliations),

        # Dates
        "published_date": pub_date,
        "updated_date": updated_date,
        "paper_age_days": paper_age_days,
        "year": pub_date[:4] if pub_date else "",
        "month": pub_date[:7] if pub_date else "",

        # Categories
        "primary_category": primary_category,
        "all_categories": ", ".join(categories),
        "category_list": categories,
        "category_count": len(categories),
        "is_cross_listed": len(categories) > 1,

        # Parsed comment info
        "arxiv_comment": arxiv_comment,
        "page_count": page_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "accepted_venue": accepted_venue,

        # Publication info
        "journal_reference": journal_reference,
        "is_published": bool(journal_reference),
        "doi": doi,

        # URLs
        "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",

        # Placeholders for enrichment
        "citations": None,
        "influential_citations": None,
        "reference_count": None,
        "fields_of_study": "",
        "publication_venue": "",
        "venue_type": "",
        "is_open_access": None,
        "tldr": "",
        "s2_paper_id": "",

        # Author metrics (to be filled)
        "first_author_h_index": None,
        "first_author_citation_count": None,
        "first_author_paper_count": None,
        "max_author_h_index": None,
        "avg_author_h_index": None,

        # OpenAlex data
        "institutions": "",
        "institution_countries": "",
        "funders": "",
        "concepts": "",
        "cited_by_count_openalex": None,

        # Papers With Code
        "has_code": False,
        "github_stars": None,
        "github_url": "",
        "datasets_used": "",
        "tasks": "",
        "methods": "",

        # Computed metrics
        "citations_per_month": None,
        "citations_per_year": None,
        "citation_velocity_rank": None,
        "interdisciplinary_score": None,
    }


# =============================================================================
# SEMANTIC SCHOLAR ENRICHMENT
# =============================================================================

def enrich_with_semantic_scholar(papers: List[Dict], rate_limit: float = 1.0,
                                 fetch_authors: bool = True) -> List[Dict]:
    """Enrich papers with Semantic Scholar data including author metrics."""

    if not papers:
        return papers

    # Batch paper data
    batch_size = 100
    total = len(papers)

    paper_fields = [
        "citationCount", "influentialCitationCount", "referenceCount",
        "fieldsOfStudy", "s2FieldsOfStudy", "publicationVenue",
        "openAccessPdf", "tldr", "publicationTypes", "authors",
        "references", "citations.paperId", "citations.citationCount"
    ]

    for i in range(0, total, batch_size):
        batch = papers[i:i + batch_size]
        batch_ids = [f"ARXIV:{p['arxiv_id_base']}" for p in batch]

        print(f"      S2 enriching {i+1}-{min(i+batch_size, total)}/{total}...")

        try:
            RATE_LIMITER.wait("s2", rate_limit)

            request_data = json.dumps({"ids": batch_ids}).encode("utf-8")
            url = f"{S2_BATCH_API}?fields={','.join(paper_fields)}"

            req = urllib.request.Request(
                url, data=request_data,
                headers={"User-Agent": "research-miner/3.0", "Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                results = json.loads(response.read().decode("utf-8"))

            for j, result in enumerate(results):
                paper = papers[i + j]

                if result is None:
                    paper["citations"] = 0
                    paper["influential_citations"] = 0
                    paper["reference_count"] = 0
                    continue

                # Basic citation data
                paper["citations"] = result.get("citationCount", 0)
                paper["influential_citations"] = result.get("influentialCitationCount", 0)
                paper["reference_count"] = result.get("referenceCount", 0)
                paper["s2_paper_id"] = result.get("paperId", "")

                # Fields of study
                fos = result.get("fieldsOfStudy") or []
                paper["fields_of_study"] = ", ".join(fos)

                s2_fos = result.get("s2FieldsOfStudy") or []
                paper["s2_fields_of_study"] = ", ".join([f.get("category", "") for f in s2_fos])

                # Venue
                venue = result.get("publicationVenue") or {}
                paper["publication_venue"] = venue.get("name", "")
                paper["venue_type"] = venue.get("type", "")

                # Open access
                oa_pdf = result.get("openAccessPdf") or {}
                paper["is_open_access"] = oa_pdf.get("url") is not None
                paper["open_access_pdf_url"] = oa_pdf.get("url", "")

                # TL;DR
                tldr = result.get("tldr") or {}
                paper["tldr"] = tldr.get("text", "")

                # Publication types
                pub_types = result.get("publicationTypes") or []
                paper["publication_types"] = ", ".join(pub_types)

                # Citation velocity
                if paper["paper_age_days"] and paper["paper_age_days"] > 0 and paper["citations"]:
                    months = max(paper["paper_age_days"] / 30.44, 1)
                    paper["citations_per_month"] = round(paper["citations"] / months, 3)
                    paper["citations_per_year"] = round(paper["citations"] / (months / 12), 3)

                # Reference analysis
                refs = result.get("references") or []
                paper["reference_arxiv_count"] = sum(1 for r in refs if r and "arxiv" in str(r.get("externalIds", {})).lower())

                # Citation sources
                citing = result.get("citations") or []
                paper["high_impact_citations"] = sum(1 for c in citing if c and (c.get("citationCount") or 0) > 50)

                # Store author IDs for later lookup
                s2_authors = result.get("authors") or []
                paper["s2_author_ids"] = [a.get("authorId") for a in s2_authors if a.get("authorId")]

        except Exception as e:
            print(f"      S2 batch error: {e}")
            for j in range(len(batch)):
                papers[i + j]["citations"] = 0
                papers[i + j]["influential_citations"] = 0

    # Fetch author metrics if requested
    if fetch_authors:
        enrich_author_metrics(papers, rate_limit)

    return papers


def enrich_author_metrics(papers: List[Dict], rate_limit: float = 1.0):
    """Fetch author h-index and metrics for papers."""

    # Collect unique author IDs
    author_ids = set()
    for paper in papers:
        author_ids.update(paper.get("s2_author_ids", [])[:5])  # Top 5 authors per paper

    author_ids = [aid for aid in author_ids if aid]

    if not author_ids:
        return

    print(f"      Fetching metrics for {len(author_ids)} unique authors...")

    # Fetch author data in batches
    author_data = {}
    batch_size = 100

    for i in range(0, len(author_ids), batch_size):
        batch = author_ids[i:i + batch_size]

        try:
            RATE_LIMITER.wait("s2_author", rate_limit)

            request_data = json.dumps({"ids": batch}).encode("utf-8")
            url = f"{S2_GRAPH_API}/author/batch?fields=hIndex,citationCount,paperCount,name"

            req = urllib.request.Request(
                url, data=request_data,
                headers={"User-Agent": "research-miner/3.0", "Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                results = json.loads(response.read().decode("utf-8"))

            for j, result in enumerate(results):
                if result:
                    author_data[batch[j]] = {
                        "h_index": result.get("hIndex", 0),
                        "citation_count": result.get("citationCount", 0),
                        "paper_count": result.get("paperCount", 0),
                        "name": result.get("name", "")
                    }

        except Exception as e:
            print(f"      Author batch error: {e}")

    # Map back to papers
    for paper in papers:
        s2_ids = paper.get("s2_author_ids", [])
        if not s2_ids:
            continue

        h_indices = []
        for aid in s2_ids:
            if aid in author_data:
                h_indices.append(author_data[aid]["h_index"] or 0)

        if h_indices:
            paper["max_author_h_index"] = max(h_indices)
            paper["avg_author_h_index"] = round(sum(h_indices) / len(h_indices), 1)

        # First author
        if s2_ids and s2_ids[0] in author_data:
            first = author_data[s2_ids[0]]
            paper["first_author_h_index"] = first["h_index"]
            paper["first_author_citation_count"] = first["citation_count"]
            paper["first_author_paper_count"] = first["paper_count"]


# =============================================================================
# OPENALEX ENRICHMENT
# =============================================================================

def enrich_with_openalex(papers: List[Dict], rate_limit: float = 0.2) -> List[Dict]:
    """Enrich papers with OpenAlex institutional and concept data."""

    if not papers:
        return papers

    # OpenAlex allows filtering by DOI or arXiv ID
    for idx, paper in enumerate(papers):
        if idx % 50 == 0:
            print(f"      OpenAlex enriching {idx+1}/{len(papers)}...")

        try:
            # Try DOI first, then arXiv ID
            if paper.get("doi"):
                url = f"{OPENALEX_API}/works/doi:{paper['doi']}"
            else:
                url = f"{OPENALEX_API}/works?filter=ids.openalex:arxiv:{paper['arxiv_id_base']}"

            RATE_LIMITER.wait("openalex", rate_limit)

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "research-miner/3.0 (mailto:research@example.com)"}
            )

            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Handle search results vs direct lookup
            if "results" in data:
                if data["results"]:
                    data = data["results"][0]
                else:
                    continue

            # Institutions
            authorships = data.get("authorships") or []
            institutions = []
            countries = []
            for authorship in authorships[:10]:
                for inst in (authorship.get("institutions") or []):
                    if inst.get("display_name"):
                        institutions.append(inst["display_name"])
                    if inst.get("country_code"):
                        countries.append(inst["country_code"])

            paper["institutions"] = "; ".join(list(dict.fromkeys(institutions))[:10])
            paper["institution_countries"] = ", ".join(list(dict.fromkeys(countries)))
            paper["institution_count"] = len(set(institutions))
            paper["country_count"] = len(set(countries))

            # Funders
            grants = data.get("grants") or []
            funders = [g.get("funder_display_name", "") for g in grants if g.get("funder_display_name")]
            paper["funders"] = "; ".join(list(dict.fromkeys(funders)))
            paper["is_funded"] = len(funders) > 0

            # Concepts (topics)
            concepts = data.get("concepts") or []
            concept_names = [c.get("display_name", "") for c in concepts[:10] if c.get("score", 0) > 0.3]
            paper["concepts"] = ", ".join(concept_names)

            # Citation count from OpenAlex
            paper["cited_by_count_openalex"] = data.get("cited_by_count", 0)

            # Citation trend
            counts_by_year = data.get("counts_by_year") or []
            if counts_by_year:
                recent_years = sorted(counts_by_year, key=lambda x: x.get("year", 0), reverse=True)[:3]
                paper["citations_last_3_years"] = sum(y.get("cited_by_count", 0) for y in recent_years)

            # Type
            paper["work_type"] = data.get("type", "")

        except urllib.error.HTTPError:
            continue
        except Exception as e:
            continue

    return papers


# =============================================================================
# PAPERS WITH CODE ENRICHMENT
# =============================================================================

def enrich_with_papers_with_code(papers: List[Dict], rate_limit: float = 0.5) -> List[Dict]:
    """Enrich papers with code/dataset information from Papers With Code."""

    if not papers:
        return papers

    for idx, paper in enumerate(papers):
        if idx % 50 == 0:
            print(f"      Papers With Code enriching {idx+1}/{len(papers)}...")

        try:
            arxiv_id = paper["arxiv_id_base"]
            url = f"{PAPERS_WITH_CODE_API}/papers/?arxiv_id={arxiv_id}"

            RATE_LIMITER.wait("pwc", rate_limit)

            req = urllib.request.Request(url, headers={"User-Agent": "research-miner/3.0"})

            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = data.get("results", [])
            if not results:
                continue

            pwc_paper = results[0]
            paper["pwc_id"] = pwc_paper.get("id", "")
            paper["pwc_url"] = pwc_paper.get("url_abs", "")

            # Get repository info
            if pwc_paper.get("id"):
                repo_url = f"{PAPERS_WITH_CODE_API}/papers/{pwc_paper['id']}/repositories/"

                RATE_LIMITER.wait("pwc", rate_limit)
                req = urllib.request.Request(repo_url, headers={"User-Agent": "research-miner/3.0"})

                with urllib.request.urlopen(req, timeout=15) as response:
                    repo_data = json.loads(response.read().decode("utf-8"))

                repos = repo_data.get("results", [])
                if repos:
                    paper["has_code"] = True
                    # Get most starred repo
                    repos_sorted = sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)
                    best_repo = repos_sorted[0]
                    paper["github_url"] = best_repo.get("url", "")
                    paper["github_stars"] = best_repo.get("stars", 0)
                    paper["repo_count"] = len(repos)
                    paper["total_github_stars"] = sum(r.get("stars", 0) for r in repos)

            # Tasks
            tasks = pwc_paper.get("tasks", [])
            paper["tasks"] = ", ".join([t.get("name", "") for t in tasks[:5]])
            paper["task_count"] = len(tasks)

            # Methods
            methods = pwc_paper.get("methods", [])
            paper["methods"] = ", ".join([m.get("name", "") for m in methods[:5]])

        except Exception:
            continue

    return papers


# =============================================================================
# COMPUTED METRICS & ANALYSIS
# =============================================================================

def compute_derived_metrics(papers: List[Dict]) -> List[Dict]:
    """Compute derived metrics for analysis."""

    if not papers:
        return papers

    # Citation velocity ranking
    papers_with_velocity = [(i, p.get("citations_per_month") or 0) for i, p in enumerate(papers)]
    papers_with_velocity.sort(key=lambda x: x[1], reverse=True)

    for rank, (idx, _) in enumerate(papers_with_velocity, 1):
        papers[idx]["citation_velocity_rank"] = rank
        papers[idx]["citation_velocity_percentile"] = round(100 * (1 - rank / len(papers)), 1)

    # Interdisciplinary score
    for paper in papers:
        categories = paper.get("category_list", [])
        if len(categories) > 1:
            # Count unique top-level categories
            top_level = set(c.split(".")[0] for c in categories)
            paper["interdisciplinary_score"] = len(top_level)
        else:
            paper["interdisciplinary_score"] = 0

    # Is survey/review paper
    for paper in papers:
        title_lower = paper.get("title", "").lower()
        abstract_lower = paper.get("abstract", "").lower()

        survey_indicators = ["survey", "review", "overview", "tutorial", "comprehensive study"]
        paper["is_survey"] = any(ind in title_lower or ind in abstract_lower[:200] for ind in survey_indicators)

    # Quality indicators
    for paper in papers:
        # High-quality venue
        venue = (paper.get("publication_venue") or "").lower()
        top_venues = ["neurips", "icml", "iclr", "cvpr", "iccv", "eccv", "acl", "emnlp", "naacl",
                      "aaai", "ijcai", "kdd", "www", "sigir", "chi", "uist", "nature", "science"]
        paper["is_top_venue"] = any(tv in venue for tv in top_venues)

        # Has strong signals
        paper["quality_signals"] = sum([
            paper.get("is_published", False),
            paper.get("has_code", False),
            paper.get("is_open_access", False),
            (paper.get("influential_citations") or 0) > 0,
            (paper.get("max_author_h_index") or 0) > 20,
            paper.get("is_top_venue", False)
        ])

    return papers


def compute_topic_trends(papers: List[Dict]) -> Dict:
    """Compute topic trends over time."""

    trends = {
        "category_by_month": defaultdict(lambda: defaultdict(int)),
        "field_by_month": defaultdict(lambda: defaultdict(int)),
        "concept_by_month": defaultdict(lambda: defaultdict(int)),
        "category_growth": {},
        "emerging_topics": [],
        "declining_topics": [],
    }

    # Count by month
    for paper in papers:
        month = paper.get("month", "")
        if not month:
            continue

        # Category counts
        for cat in paper.get("category_list", []):
            trends["category_by_month"][cat][month] += 1

        # Field counts
        for field in (paper.get("fields_of_study") or "").split(", "):
            if field.strip():
                trends["field_by_month"][field.strip()][month] += 1

        # Concept counts
        for concept in (paper.get("concepts") or "").split(", "):
            if concept.strip():
                trends["concept_by_month"][concept.strip()][month] += 1

    # Calculate growth rates
    for cat, months in trends["category_by_month"].items():
        sorted_months = sorted(months.keys())
        if len(sorted_months) >= 2:
            early = sum(months[m] for m in sorted_months[:len(sorted_months)//2])
            late = sum(months[m] for m in sorted_months[len(sorted_months)//2:])

            if early > 0:
                growth = (late - early) / early * 100
                trends["category_growth"][cat] = round(growth, 1)

    # Identify emerging and declining
    sorted_growth = sorted(trends["category_growth"].items(), key=lambda x: x[1], reverse=True)
    trends["emerging_topics"] = sorted_growth[:10]
    trends["declining_topics"] = sorted_growth[-10:]

    # Convert defaultdicts
    trends["category_by_month"] = {k: dict(v) for k, v in trends["category_by_month"].items()}
    trends["field_by_month"] = {k: dict(v) for k, v in trends["field_by_month"].items()}
    trends["concept_by_month"] = {k: dict(v) for k, v in trends["concept_by_month"].items()}

    return trends


def identify_research_gaps(papers: List[Dict]) -> Dict:
    """Identify potential research gaps and opportunities."""

    gaps = {
        "underexplored_intersections": [],
        "high_velocity_low_volume": [],
        "high_author_interest_low_output": [],
        "needs_code": [],
        "needs_survey": [],
    }

    # Category co-occurrence
    category_pairs = defaultdict(int)
    category_counts = Counter()

    for paper in papers:
        cats = paper.get("category_list", [])
        for cat in cats:
            category_counts[cat] += 1

        for i, cat1 in enumerate(cats):
            for cat2 in cats[i+1:]:
                pair = tuple(sorted([cat1, cat2]))
                category_pairs[pair] += 1

    # Find underexplored intersections
    for (cat1, cat2), count in category_pairs.items():
        expected = (category_counts[cat1] * category_counts[cat2]) / len(papers)
        if expected > 10 and count < expected * 0.3:
            gaps["underexplored_intersections"].append({
                "categories": [cat1, cat2],
                "actual": count,
                "expected": round(expected, 1),
                "ratio": round(count / expected, 2)
            })

    gaps["underexplored_intersections"].sort(key=lambda x: x["ratio"])
    gaps["underexplored_intersections"] = gaps["underexplored_intersections"][:20]

    # High velocity, low volume (emerging hot topics)
    category_velocity = defaultdict(list)
    for paper in papers:
        cat = paper.get("primary_category", "")
        velocity = paper.get("citations_per_month") or 0
        if velocity > 0:
            category_velocity[cat].append(velocity)

    for cat, velocities in category_velocity.items():
        avg_velocity = sum(velocities) / len(velocities)
        if len(velocities) < 50 and avg_velocity > 1:  # Low volume, high velocity
            gaps["high_velocity_low_volume"].append({
                "category": cat,
                "paper_count": len(velocities),
                "avg_citations_per_month": round(avg_velocity, 2)
            })

    gaps["high_velocity_low_volume"].sort(key=lambda x: x["avg_citations_per_month"], reverse=True)

    # Topics needing surveys
    category_has_survey = defaultdict(bool)
    for paper in papers:
        if paper.get("is_survey"):
            for cat in paper.get("category_list", []):
                category_has_survey[cat] = True

    for cat, count in category_counts.items():
        if count > 100 and not category_has_survey[cat]:
            gaps["needs_survey"].append({"category": cat, "paper_count": count})

    gaps["needs_survey"].sort(key=lambda x: x["paper_count"], reverse=True)

    # High-cited papers needing code
    for paper in papers:
        if (paper.get("citations") or 0) > 50 and not paper.get("has_code"):
            gaps["needs_code"].append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"][:80],
                "citations": paper["citations"],
                "category": paper["primary_category"]
            })

    gaps["needs_code"].sort(key=lambda x: x["citations"], reverse=True)
    gaps["needs_code"] = gaps["needs_code"][:30]

    return gaps


def compute_author_network(papers: List[Dict]) -> Dict:
    """Compute author collaboration network statistics."""

    author_stats = defaultdict(lambda: {
        "paper_count": 0,
        "total_citations": 0,
        "categories": set(),
        "collaborators": set(),
        "first_author_count": 0,
        "institutions": set()
    })

    for paper in papers:
        authors = paper.get("author_list", [])
        citations = paper.get("citations") or 0
        categories = paper.get("category_list", [])
        institutions = (paper.get("institutions") or "").split("; ")

        for i, author in enumerate(authors):
            author_stats[author]["paper_count"] += 1
            author_stats[author]["total_citations"] += citations
            author_stats[author]["categories"].update(categories)
            author_stats[author]["collaborators"].update(a for a in authors if a != author)

            if i == 0:
                author_stats[author]["first_author_count"] += 1

            if institutions:
                author_stats[author]["institutions"].update(inst for inst in institutions if inst)

    # Convert to serializable format
    top_authors = []
    for author, stats in author_stats.items():
        top_authors.append({
            "name": author,
            "paper_count": stats["paper_count"],
            "first_author_count": stats["first_author_count"],
            "total_citations": stats["total_citations"],
            "avg_citations": round(stats["total_citations"] / stats["paper_count"], 1),
            "category_count": len(stats["categories"]),
            "collaborator_count": len(stats["collaborators"]),
            "top_categories": list(stats["categories"])[:5],
            "institutions": list(stats["institutions"])[:3]
        })

    top_authors.sort(key=lambda x: x["total_citations"], reverse=True)

    return {
        "top_authors_by_citations": top_authors[:100],
        "top_authors_by_papers": sorted(top_authors, key=lambda x: x["paper_count"], reverse=True)[:100],
        "most_collaborative": sorted(top_authors, key=lambda x: x["collaborator_count"], reverse=True)[:50],
        "total_unique_authors": len(author_stats)
    }


# =============================================================================
# OUTPUT
# =============================================================================

def save_dataset(papers: List[Dict], output_dir: str):
    """Save the complete dataset."""

    os.makedirs(output_dir, exist_ok=True)

    if not papers:
        print("No papers to save!")
        return

    # Clean papers for CSV (remove list fields)
    csv_papers = []
    for p in papers:
        csv_paper = {k: v for k, v in p.items()
                     if not isinstance(v, (list, set)) and k not in ["s2_author_ids"]}
        csv_papers.append(csv_paper)

    # Field order for readability
    priority_fields = [
        "arxiv_id", "title", "authors", "author_count", "published_date", "year",
        "primary_category", "all_categories", "category_count", "is_cross_listed",
        "citations", "influential_citations", "citations_per_month", "citation_velocity_percentile",
        "reference_count", "max_author_h_index", "avg_author_h_index",
        "fields_of_study", "concepts", "publication_venue", "venue_type", "is_top_venue",
        "has_code", "github_stars", "github_url",
        "institutions", "institution_countries", "is_funded", "funders",
        "is_open_access", "is_published", "is_survey", "quality_signals",
        "abstract", "tldr", "arxiv_comment", "accepted_venue",
        "page_count", "figure_count",
        "doi", "arxiv_url", "pdf_url"
    ]

    all_fields = set()
    for p in csv_papers:
        all_fields.update(p.keys())

    fieldnames = [f for f in priority_fields if f in all_fields]
    extra = sorted(all_fields - set(fieldnames))
    fieldnames.extend(extra)

    # Main CSV
    csv_path = os.path.join(output_dir, "arxiv_comprehensive.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_papers)
    print(f"Saved: {csv_path} ({len(papers)} papers)")

    # JSON for full data
    json_path = os.path.join(output_dir, "arxiv_comprehensive.json")
    with open(json_path, "w", encoding="utf-8") as f:
        # Convert sets to lists for JSON
        json_papers = []
        for p in papers:
            jp = {}
            for k, v in p.items():
                if isinstance(v, set):
                    jp[k] = list(v)
                else:
                    jp[k] = v
            json_papers.append(jp)
        json.dump(json_papers, f, indent=2)
    print(f"Saved: {json_path}")


def save_analysis(papers: List[Dict], output_dir: str):
    """Save analysis results."""

    os.makedirs(output_dir, exist_ok=True)

    print("\nComputing analysis...")

    # Topic trends
    trends = compute_topic_trends(papers)
    trends_path = os.path.join(output_dir, "topic_trends.json")
    with open(trends_path, "w", encoding="utf-8") as f:
        json.dump(trends, f, indent=2)
    print(f"Saved: {trends_path}")

    # Research gaps
    gaps = identify_research_gaps(papers)
    gaps_path = os.path.join(output_dir, "research_gaps.json")
    with open(gaps_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)
    print(f"Saved: {gaps_path}")

    # Author network
    authors = compute_author_network(papers)
    authors_path = os.path.join(output_dir, "author_analysis.json")
    with open(authors_path, "w", encoding="utf-8") as f:
        json.dump(authors, f, indent=2)
    print(f"Saved: {authors_path}")

    # Summary statistics
    stats = compute_summary_stats(papers)
    stats_path = os.path.join(output_dir, "summary_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")

    # Top papers for different criteria
    save_top_papers(papers, output_dir)


def compute_summary_stats(papers: List[Dict]) -> Dict:
    """Compute summary statistics."""

    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_papers": len(papers),
        "date_range": {},
        "totals": {},
        "averages": {},
        "distributions": {},
        "coverage": {},
    }

    dates = [p["published_date"] for p in papers if p.get("published_date")]
    if dates:
        stats["date_range"] = {"min": min(dates), "max": max(dates)}

    # Totals
    stats["totals"] = {
        "citations": sum(p.get("citations") or 0 for p in papers),
        "influential_citations": sum(p.get("influential_citations") or 0 for p in papers),
        "references": sum(p.get("reference_count") or 0 for p in papers),
        "papers_with_code": sum(1 for p in papers if p.get("has_code")),
        "papers_open_access": sum(1 for p in papers if p.get("is_open_access")),
        "papers_published": sum(1 for p in papers if p.get("is_published")),
        "papers_funded": sum(1 for p in papers if p.get("is_funded")),
        "surveys": sum(1 for p in papers if p.get("is_survey")),
        "unique_authors": len(set(a for p in papers for a in p.get("author_list", []))),
        "unique_institutions": len(set(i for p in papers for i in (p.get("institutions") or "").split("; ") if i)),
    }

    # Averages
    citations = [p.get("citations") or 0 for p in papers]
    stats["averages"] = {
        "citations": round(sum(citations) / len(citations), 2) if citations else 0,
        "author_count": round(sum(p.get("author_count") or 0 for p in papers) / len(papers), 1),
        "category_count": round(sum(p.get("category_count") or 0 for p in papers) / len(papers), 1),
        "abstract_words": round(sum(p.get("abstract_word_count") or 0 for p in papers) / len(papers), 0),
    }

    # Distributions
    stats["distributions"] = {
        "by_category": dict(Counter(p.get("primary_category") for p in papers)),
        "by_year": dict(Counter(p.get("year") for p in papers if p.get("year"))),
        "by_month": dict(Counter(p.get("month") for p in papers if p.get("month"))),
        "by_venue_type": dict(Counter(p.get("venue_type") for p in papers if p.get("venue_type"))),
    }

    # Coverage
    stats["coverage"] = {
        "has_citations": sum(1 for p in papers if p.get("citations") is not None) / len(papers) * 100,
        "has_tldr": sum(1 for p in papers if p.get("tldr")) / len(papers) * 100,
        "has_venue": sum(1 for p in papers if p.get("publication_venue")) / len(papers) * 100,
        "has_concepts": sum(1 for p in papers if p.get("concepts")) / len(papers) * 100,
        "has_institutions": sum(1 for p in papers if p.get("institutions")) / len(papers) * 100,
        "has_github": sum(1 for p in papers if p.get("has_code")) / len(papers) * 100,
    }

    return stats


def save_top_papers(papers: List[Dict], output_dir: str):
    """Save various top paper lists."""

    def paper_summary(p):
        return {
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "authors": p["authors"][:100],
            "published_date": p["published_date"],
            "category": p["primary_category"],
            "citations": p.get("citations"),
            "influential_citations": p.get("influential_citations"),
            "citations_per_month": p.get("citations_per_month"),
            "venue": p.get("publication_venue"),
            "has_code": p.get("has_code"),
            "github_stars": p.get("github_stars"),
            "arxiv_url": p["arxiv_url"]
        }

    top_lists = {
        "most_cited": sorted(papers, key=lambda x: x.get("citations") or 0, reverse=True)[:100],
        "most_influential": sorted(papers, key=lambda x: x.get("influential_citations") or 0, reverse=True)[:100],
        "highest_velocity": sorted(papers, key=lambda x: x.get("citations_per_month") or 0, reverse=True)[:100],
        "most_starred_code": sorted([p for p in papers if p.get("has_code")],
                                    key=lambda x: x.get("github_stars") or 0, reverse=True)[:100],
        "top_venue_papers": sorted([p for p in papers if p.get("is_top_venue")],
                                   key=lambda x: x.get("citations") or 0, reverse=True)[:100],
        "interdisciplinary": sorted(papers, key=lambda x: x.get("interdisciplinary_score") or 0, reverse=True)[:100],
    }

    for name, paper_list in top_lists.items():
        path = os.path.join(output_dir, f"top_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([paper_summary(p) for p in paper_list], f, indent=2)

    print(f"Saved: top paper lists (6 files)")


def print_summary(papers: List[Dict]):
    """Print summary to console."""

    stats = compute_summary_stats(papers)

    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal papers: {stats['total_papers']:,}")
    print(f"Date range: {stats['date_range'].get('min', 'N/A')} to {stats['date_range'].get('max', 'N/A')}")

    print(f"\n--- Totals ---")
    for k, v in stats["totals"].items():
        print(f"  {k}: {v:,}")

    print(f"\n--- Data Coverage ---")
    for k, v in stats["coverage"].items():
        print(f"  {k}: {v:.1f}%")

    print(f"\n--- Top 10 Categories ---")
    cat_dist = sorted(stats["distributions"]["by_category"].items(), key=lambda x: x[1], reverse=True)
    for cat, count in cat_dist[:10]:
        print(f"  {cat}: {count:,}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive arXiv research mining")

    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (default: all CS)")
    parser.add_argument("--all-categories", action="store_true",
                        help="Include all categories (CS, ML, math, physics, etc.)")
    parser.add_argument("--max-results", type=int, default=500,
                        help="Max papers per category (default: 500)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 12 months ago)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output-dir", type=str, default="arxiv_research_dataset",
                        help="Output directory")

    # Data source toggles
    parser.add_argument("--skip-s2", action="store_true",
                        help="Skip Semantic Scholar enrichment")
    parser.add_argument("--skip-openalex", action="store_true",
                        help="Skip OpenAlex enrichment")
    parser.add_argument("--skip-pwc", action="store_true",
                        help="Skip Papers With Code enrichment")
    parser.add_argument("--skip-authors", action="store_true",
                        help="Skip author metrics lookup")

    # Rate limits
    parser.add_argument("--arxiv-rate", type=float, default=3.0,
                        help="arXiv rate limit seconds (default: 3.0)")
    parser.add_argument("--s2-rate", type=float, default=1.0,
                        help="Semantic Scholar rate limit (default: 1.0)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Date range
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else end_date - timedelta(days=365)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # Categories
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    elif args.all_categories:
        categories = ALL_CATEGORIES
    else:
        categories = CS_CATEGORIES

    print("=" * 70)
    print("COMPREHENSIVE ARXIV RESEARCH MINER")
    print("=" * 70)
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Categories: {len(categories)}")
    print(f"Max per category: {args.max_results}")
    print(f"Output: {args.output_dir}")
    print(f"\nData sources:")
    print(f"  arXiv: enabled")
    print(f"  Semantic Scholar: {'disabled' if args.skip_s2 else 'enabled'}")
    print(f"  OpenAlex: {'disabled' if args.skip_openalex else 'enabled'}")
    print(f"  Papers With Code: {'disabled' if args.skip_pwc else 'enabled'}")
    print(f"  Author metrics: {'disabled' if args.skip_authors else 'enabled'}")
    print("=" * 70)

    # Collect papers
    all_papers = []

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] {category}")

        print("    Fetching from arXiv...")
        papers = fetch_arxiv_papers(
            category=category,
            start_date=start_str,
            end_date=end_str,
            max_results=args.max_results,
            rate_limit=args.arxiv_rate
        )
        print(f"    Found {len(papers)} papers")

        if papers and not args.skip_s2:
            print("    Enriching with Semantic Scholar...")
            papers = enrich_with_semantic_scholar(
                papers,
                rate_limit=args.s2_rate,
                fetch_authors=not args.skip_authors
            )

        all_papers.extend(papers)

    # Deduplicate by arxiv_id
    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique_papers.append(p)

    print(f"\nTotal unique papers: {len(unique_papers)}")

    # Additional enrichment (can take a while)
    if not args.skip_openalex and unique_papers:
        print("\nEnriching with OpenAlex...")
        unique_papers = enrich_with_openalex(unique_papers)

    if not args.skip_pwc and unique_papers:
        print("\nEnriching with Papers With Code...")
        unique_papers = enrich_with_papers_with_code(unique_papers)

    # Compute derived metrics
    print("\nComputing derived metrics...")
    unique_papers = compute_derived_metrics(unique_papers)

    # Save everything
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    save_dataset(unique_papers, args.output_dir)
    save_analysis(unique_papers, args.output_dir)

    print_summary(unique_papers)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"Dataset saved to: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
