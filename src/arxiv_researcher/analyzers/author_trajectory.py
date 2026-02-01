"""
Author Career Trajectory Analysis Module
=========================================
Track and analyze how researchers evolve over time:
- Career trajectory analysis
- Rising star identification
- Topic migration tracking
- Collaboration evolution
- Author momentum scoring
- Author clustering
"""

import re
import time
import json
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import math

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 10.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0

    def wait(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()


class AuthorCache:
    """Cache for author data to avoid redundant API calls."""

    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".arxiv_research_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.memory_cache: Dict[str, Dict] = {}

    def _get_cache_key(self, author_id: str) -> str:
        """Generate a cache key for an author."""
        return hashlib.md5(author_id.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached author."""
        return self.cache_dir / f"author_{cache_key}.json"

    def get(self, author_id: str) -> Optional[Dict]:
        """Get cached author data if available and not expired."""
        # Check memory cache first
        if author_id in self.memory_cache:
            cached = self.memory_cache[author_id]
            if time.time() - cached.get("_cached_at", 0) < self.ttl_seconds:
                return cached.get("data")

        # Check disk cache
        cache_key = self._get_cache_key(author_id)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                if time.time() - cached.get("_cached_at", 0) < self.ttl_seconds:
                    # Store in memory cache for faster access
                    self.memory_cache[author_id] = cached
                    return cached.get("data")
            except (json.JSONDecodeError, IOError):
                pass

        return None

    def set(self, author_id: str, data: Dict):
        """Cache author data."""
        cached = {
            "data": data,
            "_cached_at": time.time(),
            "_author_id": author_id,
        }

        # Store in memory
        self.memory_cache[author_id] = cached

        # Store on disk
        cache_key = self._get_cache_key(author_id)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump(cached, f)
        except IOError:
            pass  # Silently fail disk cache writes

    def clear(self):
        """Clear all cached data."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("author_*.json"):
            try:
                cache_file.unlink()
            except IOError:
                pass


class SemanticScholarClient:
    """Client for Semantic Scholar API with rate limiting and caching."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        calls_per_second: float = 10.0,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
    ):
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")

        self.api_key = api_key
        self.rate_limiter = RateLimiter(calls_per_second)
        self.cache = AuthorCache(cache_dir, cache_ttl_hours)
        self.session = requests.Session()

        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Any] = None,
    ) -> Optional[Dict]:
        """Make a rate-limited API request."""
        self.rate_limiter.wait()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params, json=json_data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry once
                time.sleep(1.0)
                self.rate_limiter.wait()
                response = self.session.request(
                    method, url, params=params, json=json_data, timeout=30
                )
                if response.status_code == 200:
                    return response.json()
            elif response.status_code == 404:
                return None

            # Log error but don't raise
            print(f"    API error {response.status_code}: {endpoint}")
            return None

        except requests.RequestException as e:
            print(f"    Request error: {e}")
            return None

    def get_author(
        self,
        author_id: str,
        fields: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Optional[Dict]:
        """
        Get author data from Semantic Scholar.

        Args:
            author_id: Semantic Scholar author ID or name
            fields: Fields to retrieve
            use_cache: Whether to use cached data

        Returns:
            Author data dict or None if not found
        """
        if use_cache:
            cached = self.cache.get(author_id)
            if cached:
                return cached

        if fields is None:
            fields = [
                "authorId", "name", "aliases", "affiliations",
                "paperCount", "citationCount", "hIndex",
                "papers", "papers.year", "papers.citationCount",
                "papers.title", "papers.venue", "papers.authors",
                "papers.fieldsOfStudy",
            ]

        params = {"fields": ",".join(fields)}
        result = self._make_request("GET", f"author/{author_id}", params=params)

        if result and use_cache:
            self.cache.set(author_id, result)

        return result

    def get_authors_batch(
        self,
        author_ids: List[str],
        fields: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Dict]:
        """
        Get multiple authors in batch.

        Args:
            author_ids: List of Semantic Scholar author IDs
            fields: Fields to retrieve
            use_cache: Whether to use cached data

        Returns:
            Dict mapping author IDs to their data
        """
        results = {}
        ids_to_fetch = []

        # Check cache first
        if use_cache:
            for author_id in author_ids:
                cached = self.cache.get(author_id)
                if cached:
                    results[author_id] = cached
                else:
                    ids_to_fetch.append(author_id)
        else:
            ids_to_fetch = list(author_ids)

        if not ids_to_fetch:
            return results

        if fields is None:
            fields = [
                "authorId", "name", "paperCount", "citationCount", "hIndex",
            ]

        # Batch API endpoint - max 1000 per request
        for i in range(0, len(ids_to_fetch), 500):
            batch = ids_to_fetch[i:i + 500]
            params = {"fields": ",".join(fields)}

            response = self._make_request(
                "POST",
                "author/batch",
                params=params,
                json_data={"ids": batch},
            )

            if response:
                for author_data in response:
                    if author_data and author_data.get("authorId"):
                        aid = author_data["authorId"]
                        results[aid] = author_data
                        if use_cache:
                            self.cache.set(aid, author_data)

        return results

    def search_author(self, name: str, limit: int = 5) -> List[Dict]:
        """
        Search for authors by name.

        Args:
            name: Author name to search
            limit: Maximum results to return

        Returns:
            List of matching author records
        """
        params = {
            "query": name,
            "limit": limit,
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
        }

        result = self._make_request("GET", "author/search", params=params)

        if result and "data" in result:
            return result["data"]
        return []


class AuthorTrajectoryAnalyzer:
    """
    Analyze researcher career trajectories and evolution patterns.

    This analyzer tracks how researchers evolve over time, including:
    - Publication patterns and productivity
    - Citation impact growth
    - Topic/field migrations
    - Collaboration network evolution
    - Career momentum indicators
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        calls_per_second: float = 10.0,
    ):
        """
        Initialize the analyzer.

        Args:
            api_key: Semantic Scholar API key (optional, increases rate limits)
            cache_dir: Directory for caching author data
            calls_per_second: API rate limit
        """
        self.client = None
        if HAS_REQUESTS:
            try:
                self.client = SemanticScholarClient(
                    api_key=api_key,
                    calls_per_second=calls_per_second,
                    cache_dir=cache_dir,
                )
            except ImportError:
                pass

        # Internal data structures
        self.author_papers: Dict[str, List[Dict]] = defaultdict(list)
        self.coauthor_graph: Dict[str, Set[str]] = defaultdict(set)
        self.author_categories: Dict[str, Counter] = defaultdict(Counter)
        self.author_metrics: Dict[str, Dict] = {}

    def _resolve_author_id(self, author_name_or_id: str) -> Optional[str]:
        """Resolve an author name to a Semantic Scholar ID."""
        if not self.client:
            return None

        # If it looks like an S2 ID (numeric), use directly
        if author_name_or_id.isdigit():
            return author_name_or_id

        # Otherwise search by name
        results = self.client.search_author(author_name_or_id, limit=1)
        if results:
            return results[0].get("authorId")

        return None

    def analyze_author(self, author_name_or_id: str) -> Dict:
        """
        Perform a comprehensive analysis of an author's career trajectory.

        Args:
            author_name_or_id: Author name or Semantic Scholar ID

        Returns:
            Dict containing full profile analysis including:
            - Basic info (name, affiliations, h-index)
            - Publication timeline
            - Citation growth patterns
            - Topic evolution
            - Collaboration patterns
            - Career momentum
        """
        if not self.client:
            return {"error": "Semantic Scholar client not available (install requests library)"}

        # Resolve author ID
        author_id = self._resolve_author_id(author_name_or_id)
        if not author_id:
            return {"error": f"Could not find author: {author_name_or_id}"}

        # Fetch author data
        author_data = self.client.get_author(author_id)
        if not author_data:
            return {"error": f"Could not retrieve data for author: {author_name_or_id}"}

        papers = author_data.get("papers", [])

        # Build analysis
        analysis = {
            "author_id": author_data.get("authorId"),
            "name": author_data.get("name"),
            "aliases": author_data.get("aliases", []),
            "affiliations": author_data.get("affiliations", []),
            "h_index": author_data.get("hIndex"),
            "total_papers": author_data.get("paperCount"),
            "total_citations": author_data.get("citationCount"),
        }

        # Publication timeline
        analysis["publication_timeline"] = self._analyze_publication_timeline(papers)

        # Citation growth
        analysis["citation_growth"] = self._analyze_citation_growth(papers)

        # Topic evolution
        analysis["topic_evolution"] = self._analyze_topic_evolution(papers)

        # Collaboration patterns
        analysis["collaboration_patterns"] = self._analyze_collaboration_patterns_single(papers)

        # Career momentum
        analysis["momentum"] = self.compute_author_momentum_from_papers(papers)

        # Career stage estimation
        analysis["career_stage"] = self._estimate_career_stage(papers, analysis)

        return analysis

    def _analyze_publication_timeline(self, papers: List[Dict]) -> Dict:
        """Analyze publication patterns over time."""
        papers_by_year = defaultdict(list)

        for paper in papers:
            year = paper.get("year")
            if year:
                papers_by_year[year].append(paper)

        if not papers_by_year:
            return {"error": "No papers with year information"}

        sorted_years = sorted(papers_by_year.keys())

        # Compute metrics
        timeline = {}
        for year in sorted_years:
            year_papers = papers_by_year[year]
            citations = [p.get("citationCount", 0) or 0 for p in year_papers]

            timeline[year] = {
                "paper_count": len(year_papers),
                "total_citations": sum(citations),
                "avg_citations": round(sum(citations) / len(citations), 2) if citations else 0,
                "max_citations": max(citations) if citations else 0,
            }

        # Compute trends
        paper_counts = [timeline[y]["paper_count"] for y in sorted_years]
        trend = self._compute_trend(paper_counts)

        return {
            "by_year": timeline,
            "first_publication_year": sorted_years[0],
            "career_length_years": sorted_years[-1] - sorted_years[0] + 1,
            "total_active_years": len(sorted_years),
            "productivity_trend": trend,
            "peak_year": max(sorted_years, key=lambda y: timeline[y]["paper_count"]),
            "avg_papers_per_year": round(sum(paper_counts) / len(paper_counts), 2),
        }

    def _analyze_citation_growth(self, papers: List[Dict]) -> Dict:
        """Analyze citation accumulation patterns."""
        papers_by_year = defaultdict(list)

        for paper in papers:
            year = paper.get("year")
            if year:
                papers_by_year[year].append(paper)

        if not papers_by_year:
            return {}

        sorted_years = sorted(papers_by_year.keys())

        # Cumulative citations over time
        cumulative = {}
        running_total = 0
        running_papers = 0

        for year in sorted_years:
            year_papers = papers_by_year[year]
            year_citations = sum(p.get("citationCount", 0) or 0 for p in year_papers)
            running_total += year_citations
            running_papers += len(year_papers)

            cumulative[year] = {
                "cumulative_citations": running_total,
                "cumulative_papers": running_papers,
                "citations_per_paper": round(running_total / running_papers, 2) if running_papers else 0,
            }

        # h-index growth approximation (based on papers we have)
        all_citations = sorted(
            [p.get("citationCount", 0) or 0 for p in papers],
            reverse=True
        )
        h_index = sum(1 for i, c in enumerate(all_citations) if c >= i + 1)

        # Citation velocity (recent vs. overall)
        recent_years = sorted_years[-3:] if len(sorted_years) >= 3 else sorted_years
        recent_papers = []
        for y in recent_years:
            recent_papers.extend(papers_by_year[y])

        recent_citations = sum(p.get("citationCount", 0) or 0 for p in recent_papers)
        recent_avg = round(recent_citations / len(recent_papers), 2) if recent_papers else 0

        all_citations_sum = sum(p.get("citationCount", 0) or 0 for p in papers)
        overall_avg = round(all_citations_sum / len(papers), 2) if papers else 0

        return {
            "cumulative_by_year": cumulative,
            "computed_h_index": h_index,
            "recent_avg_citations": recent_avg,
            "overall_avg_citations": overall_avg,
            "citation_trend": "accelerating" if recent_avg > overall_avg * 1.2 else (
                "decelerating" if recent_avg < overall_avg * 0.8 else "stable"
            ),
        }

    def _analyze_topic_evolution(self, papers: List[Dict]) -> Dict:
        """Analyze how an author's research topics have evolved."""
        topics_by_year = defaultdict(Counter)

        for paper in papers:
            year = paper.get("year")
            fields = paper.get("fieldsOfStudy", []) or []

            if year and fields:
                for field in fields:
                    topics_by_year[year][field] += 1

        if not topics_by_year:
            return {"error": "No topic information available"}

        sorted_years = sorted(topics_by_year.keys())

        # Track topic prevalence over time
        all_topics = set()
        for year_topics in topics_by_year.values():
            all_topics.update(year_topics.keys())

        topic_timeline = {}
        for topic in all_topics:
            topic_timeline[topic] = {
                year: topics_by_year[year].get(topic, 0)
                for year in sorted_years
            }

        # Identify primary topics (overall)
        overall_topics = Counter()
        for year_topics in topics_by_year.values():
            overall_topics.update(year_topics)

        primary_topics = [t for t, _ in overall_topics.most_common(5)]

        # Identify topic migrations
        early_years = sorted_years[:len(sorted_years)//3] if len(sorted_years) >= 3 else sorted_years[:1]
        late_years = sorted_years[-len(sorted_years)//3:] if len(sorted_years) >= 3 else sorted_years[-1:]

        early_topics = Counter()
        late_topics = Counter()

        for y in early_years:
            early_topics.update(topics_by_year[y])
        for y in late_years:
            late_topics.update(topics_by_year[y])

        # Topics that emerged or faded
        emerged = []
        faded = []

        for topic in all_topics:
            early_count = early_topics.get(topic, 0)
            late_count = late_topics.get(topic, 0)

            if late_count > early_count * 2 and late_count >= 2:
                emerged.append({"topic": topic, "early": early_count, "late": late_count})
            elif early_count > late_count * 2 and early_count >= 2:
                faded.append({"topic": topic, "early": early_count, "late": late_count})

        return {
            "topic_timeline": topic_timeline,
            "primary_topics": primary_topics,
            "emerged_topics": emerged,
            "faded_topics": faded,
            "topic_diversity": len(all_topics),
            "early_period": f"{early_years[0]}-{early_years[-1]}" if early_years else "N/A",
            "late_period": f"{late_years[0]}-{late_years[-1]}" if late_years else "N/A",
        }

    def _analyze_collaboration_patterns_single(self, papers: List[Dict]) -> Dict:
        """Analyze collaboration patterns for a single author."""
        coauthors = Counter()
        coauthors_by_year = defaultdict(set)
        solo_papers = 0

        for paper in papers:
            year = paper.get("year")
            authors = paper.get("authors", []) or []

            if len(authors) <= 1:
                solo_papers += 1
                continue

            for author in authors:
                author_id = author.get("authorId")
                author_name = author.get("name")
                if author_id:
                    coauthors[author_id] += 1
                    if year:
                        coauthors_by_year[year].add(author_id)

        # Collaboration breadth over time
        sorted_years = sorted(coauthors_by_year.keys())
        cumulative_coauthors = set()
        breadth_over_time = {}

        for year in sorted_years:
            cumulative_coauthors.update(coauthors_by_year[year])
            breadth_over_time[year] = {
                "new_coauthors": len(coauthors_by_year[year]),
                "cumulative_coauthors": len(cumulative_coauthors),
            }

        # Most frequent collaborators
        top_collaborators = coauthors.most_common(10)

        return {
            "total_unique_coauthors": len(coauthors),
            "solo_paper_count": solo_papers,
            "solo_paper_rate": round(solo_papers / len(papers) * 100, 1) if papers else 0,
            "top_collaborators": [
                {"author_id": aid, "paper_count": count}
                for aid, count in top_collaborators
            ],
            "collaboration_breadth_over_time": breadth_over_time,
            "avg_coauthors_per_paper": round(
                sum(c for c in coauthors.values()) / len(papers), 2
            ) if papers else 0,
        }

    def _estimate_career_stage(self, papers: List[Dict], analysis: Dict) -> Dict:
        """Estimate the author's career stage."""
        timeline = analysis.get("publication_timeline", {})
        career_length = timeline.get("career_length_years", 0)
        h_index = analysis.get("h_index") or 0
        total_papers = analysis.get("total_papers", 0)

        # Heuristic career stage estimation
        if career_length <= 5:
            stage = "early_career"
            description = "Early-career researcher (0-5 years)"
        elif career_length <= 12:
            stage = "mid_career"
            description = "Mid-career researcher (5-12 years)"
        elif career_length <= 20:
            stage = "senior"
            description = "Senior researcher (12-20 years)"
        else:
            stage = "established"
            description = "Established/Distinguished researcher (20+ years)"

        # Adjust based on h-index
        if h_index >= 50:
            impact_level = "high_impact"
        elif h_index >= 20:
            impact_level = "significant_impact"
        elif h_index >= 10:
            impact_level = "moderate_impact"
        else:
            impact_level = "emerging_impact"

        return {
            "stage": stage,
            "description": description,
            "career_length_years": career_length,
            "impact_level": impact_level,
            "h_index": h_index,
            "total_papers": total_papers,
        }

    def compute_author_momentum_from_papers(self, papers: List[Dict]) -> Dict:
        """
        Compute career momentum from paper data.

        Momentum indicates whether an author's impact is growing or declining.
        """
        papers_by_year = defaultdict(list)

        for paper in papers:
            year = paper.get("year")
            if year:
                papers_by_year[year].append(paper)

        if len(papers_by_year) < 3:
            return {"momentum": 0, "trend": "insufficient_data"}

        sorted_years = sorted(papers_by_year.keys())

        # Compare recent period to earlier period
        mid_point = len(sorted_years) // 2
        early_years = sorted_years[:mid_point]
        recent_years = sorted_years[mid_point:]

        def compute_period_metrics(years):
            paper_count = sum(len(papers_by_year[y]) for y in years)
            total_citations = sum(
                sum(p.get("citationCount", 0) or 0 for p in papers_by_year[y])
                for y in years
            )
            return paper_count, total_citations

        early_papers, early_citations = compute_period_metrics(early_years)
        recent_papers, recent_citations = compute_period_metrics(recent_years)

        # Normalize by period length
        early_years_count = len(early_years)
        recent_years_count = len(recent_years)

        early_papers_rate = early_papers / early_years_count if early_years_count else 0
        recent_papers_rate = recent_papers / recent_years_count if recent_years_count else 0

        early_citations_rate = early_citations / early_years_count if early_years_count else 0
        recent_citations_rate = recent_citations / recent_years_count if recent_years_count else 0

        # Compute momentum scores
        productivity_momentum = (
            (recent_papers_rate - early_papers_rate) / max(early_papers_rate, 1)
        )
        citation_momentum = (
            (recent_citations_rate - early_citations_rate) / max(early_citations_rate, 1)
        )

        # Combined momentum (weighted average)
        combined_momentum = 0.4 * productivity_momentum + 0.6 * citation_momentum

        if combined_momentum > 0.3:
            trend = "accelerating"
        elif combined_momentum < -0.3:
            trend = "decelerating"
        else:
            trend = "stable"

        return {
            "combined_momentum": round(combined_momentum, 3),
            "productivity_momentum": round(productivity_momentum, 3),
            "citation_momentum": round(citation_momentum, 3),
            "trend": trend,
            "early_period": f"{early_years[0]}-{early_years[-1]}",
            "recent_period": f"{recent_years[0]}-{recent_years[-1]}",
            "early_papers_per_year": round(early_papers_rate, 2),
            "recent_papers_per_year": round(recent_papers_rate, 2),
        }

    def find_rising_stars(self, papers: List[Dict], top_n: int = 50) -> List[Dict]:
        """
        Identify junior researchers with accelerating impact.

        Rising stars are defined as:
        - Relatively early career (fewer total papers)
        - High recent citation velocity
        - Growing publication rate

        Args:
            papers: List of paper dicts with author information
            top_n: Number of rising stars to return

        Returns:
            List of rising star candidates with metrics
        """
        # Build author profiles from papers
        author_data = defaultdict(lambda: {
            "papers": [],
            "name": "",
            "total_citations": 0,
            "paper_years": [],
        })

        for paper in papers:
            authors = paper.get("author_list", [])
            year = paper.get("year")
            citations = paper.get("citations") or 0

            for author in authors:
                author_data[author]["papers"].append(paper)
                author_data[author]["name"] = author
                author_data[author]["total_citations"] += citations
                if year:
                    author_data[author]["paper_years"].append(year)

        current_year = datetime.now().year
        rising_stars = []

        for author, data in author_data.items():
            paper_count = len(data["papers"])
            years = data["paper_years"]

            # Filter: must have at least 3 papers but not too many (junior)
            if paper_count < 3 or paper_count > 50:
                continue

            # Filter: career must be relatively short
            if years:
                career_start = min(years)
                career_length = current_year - career_start + 1
                if career_length > 8:  # More than 8 years = not junior
                    continue
            else:
                continue

            # Compute recent vs overall citation rate
            recent_papers = [
                p for p in data["papers"]
                if (p.get("year") or 0) >= current_year - 2
            ]

            if not recent_papers:
                continue

            recent_citations = sum(p.get("citations") or 0 for p in recent_papers)
            recent_avg = recent_citations / len(recent_papers)

            overall_avg = data["total_citations"] / paper_count

            # Compute citation velocity
            recent_velocity = sum(
                p.get("citations_per_month") or 0 for p in recent_papers
            )

            # Rising star score
            # Higher if: high recent velocity, improving citation avg, short career
            velocity_score = min(recent_velocity / 5, 2)  # Cap at 2
            improvement_ratio = recent_avg / max(overall_avg, 1)
            recency_bonus = 1 + (8 - career_length) * 0.1  # Bonus for being junior

            star_score = (
                velocity_score * 2 +
                improvement_ratio * 1.5 +
                recency_bonus +
                math.log(1 + data["total_citations"]) * 0.5
            )

            rising_stars.append({
                "name": author,
                "star_score": round(star_score, 2),
                "paper_count": paper_count,
                "total_citations": data["total_citations"],
                "recent_paper_count": len(recent_papers),
                "recent_citations": recent_citations,
                "recent_avg_citations": round(recent_avg, 1),
                "overall_avg_citations": round(overall_avg, 1),
                "career_start": career_start,
                "career_length_years": career_length,
                "improvement_ratio": round(improvement_ratio, 2),
            })

        rising_stars.sort(key=lambda x: x["star_score"], reverse=True)
        return rising_stars[:top_n]

    def track_topic_migration(self, author_name_or_id: str) -> Dict:
        """
        Track what research fields an author has moved between.

        Args:
            author_name_or_id: Author name or Semantic Scholar ID

        Returns:
            Dict describing topic transitions over time
        """
        analysis = self.analyze_author(author_name_or_id)

        if "error" in analysis:
            return analysis

        topic_evolution = analysis.get("topic_evolution", {})

        # Compute explicit transitions
        topic_timeline = topic_evolution.get("topic_timeline", {})

        transitions = []
        topics_list = list(topic_timeline.keys())

        for topic in topics_list:
            timeline = topic_timeline[topic]
            years = sorted([y for y, c in timeline.items() if c > 0])

            if len(years) >= 2:
                transitions.append({
                    "topic": topic,
                    "first_year": years[0],
                    "last_year": years[-1],
                    "active_years": len(years),
                    "is_current": years[-1] >= datetime.now().year - 2,
                })

        # Sort by recency
        transitions.sort(key=lambda x: x["last_year"], reverse=True)

        return {
            "author": analysis.get("name"),
            "author_id": analysis.get("author_id"),
            "primary_topics": topic_evolution.get("primary_topics", []),
            "emerged_topics": topic_evolution.get("emerged_topics", []),
            "faded_topics": topic_evolution.get("faded_topics", []),
            "topic_transitions": transitions,
            "topic_diversity_score": topic_evolution.get("topic_diversity", 0),
        }

    def find_collaboration_evolution(self, author_name_or_id: str) -> Dict:
        """
        Analyze how an author's collaboration network has changed over time.

        Args:
            author_name_or_id: Author name or Semantic Scholar ID

        Returns:
            Dict describing collaboration evolution
        """
        analysis = self.analyze_author(author_name_or_id)

        if "error" in analysis:
            return analysis

        collab = analysis.get("collaboration_patterns", {})

        return {
            "author": analysis.get("name"),
            "author_id": analysis.get("author_id"),
            "total_unique_coauthors": collab.get("total_unique_coauthors", 0),
            "solo_paper_rate": collab.get("solo_paper_rate", 0),
            "top_collaborators": collab.get("top_collaborators", []),
            "collaboration_growth": collab.get("collaboration_breadth_over_time", {}),
            "avg_team_size": collab.get("avg_coauthors_per_paper", 0),
        }

    def identify_prolific_authors(
        self,
        papers: List[Dict],
        top_n: int = 100,
        min_papers: int = 3,
    ) -> List[Dict]:
        """
        Identify the most productive authors in a paper dataset.

        Args:
            papers: List of paper dicts
            top_n: Number of authors to return
            min_papers: Minimum papers to be considered

        Returns:
            List of prolific authors with productivity metrics
        """
        author_stats = defaultdict(lambda: {
            "paper_count": 0,
            "total_citations": 0,
            "categories": Counter(),
            "years": [],
            "first_author_count": 0,
        })

        for paper in papers:
            authors = paper.get("author_list", [])
            citations = paper.get("citations") or 0
            category = paper.get("primary_category", "")
            year = paper.get("year")

            for i, author in enumerate(authors):
                author_stats[author]["paper_count"] += 1
                author_stats[author]["total_citations"] += citations
                author_stats[author]["categories"][category] += 1
                if year:
                    author_stats[author]["years"].append(year)
                if i == 0:
                    author_stats[author]["first_author_count"] += 1

        prolific = []

        for author, stats in author_stats.items():
            if stats["paper_count"] < min_papers:
                continue

            years = stats["years"]
            if years:
                career_span = max(years) - min(years) + 1
                papers_per_year = stats["paper_count"] / career_span
            else:
                career_span = 1
                papers_per_year = stats["paper_count"]

            prolific.append({
                "name": author,
                "paper_count": stats["paper_count"],
                "total_citations": stats["total_citations"],
                "first_author_count": stats["first_author_count"],
                "first_author_rate": round(
                    stats["first_author_count"] / stats["paper_count"] * 100, 1
                ),
                "primary_category": stats["categories"].most_common(1)[0][0] if stats["categories"] else "",
                "category_diversity": len(stats["categories"]),
                "career_span_years": career_span,
                "papers_per_year": round(papers_per_year, 2),
                "avg_citations": round(
                    stats["total_citations"] / stats["paper_count"], 1
                ),
            })

        prolific.sort(key=lambda x: x["paper_count"], reverse=True)
        return prolific[:top_n]

    def identify_influential_authors(
        self,
        papers: List[Dict],
        top_n: int = 100,
        min_papers: int = 2,
    ) -> List[Dict]:
        """
        Identify the most influential authors (high impact, not just volume).

        Uses a composite score considering:
        - Citation count
        - Citation velocity
        - h-index approximation
        - Network position (co-authorship breadth)

        Args:
            papers: List of paper dicts
            top_n: Number of authors to return
            min_papers: Minimum papers to be considered

        Returns:
            List of influential authors with impact metrics
        """
        author_stats = defaultdict(lambda: {
            "papers": [],
            "coauthors": set(),
            "total_citations": 0,
        })

        for paper in papers:
            authors = paper.get("author_list", [])
            citations = paper.get("citations") or 0

            for author in authors:
                author_stats[author]["papers"].append({
                    "citations": citations,
                    "velocity": paper.get("citations_per_month") or 0,
                })
                author_stats[author]["total_citations"] += citations

                for coauthor in authors:
                    if coauthor != author:
                        author_stats[author]["coauthors"].add(coauthor)

        influential = []

        for author, stats in author_stats.items():
            paper_count = len(stats["papers"])
            if paper_count < min_papers:
                continue

            # Compute approximate h-index
            citation_counts = sorted(
                [p["citations"] for p in stats["papers"]],
                reverse=True
            )
            h_index = sum(1 for i, c in enumerate(citation_counts) if c >= i + 1)

            # Compute average velocity
            avg_velocity = sum(p["velocity"] for p in stats["papers"]) / paper_count

            # Compute influence score
            influence_score = (
                math.log(1 + stats["total_citations"]) * 2 +
                h_index * 3 +
                avg_velocity * 10 +
                math.log(1 + len(stats["coauthors"])) * 0.5
            )

            influential.append({
                "name": author,
                "influence_score": round(influence_score, 2),
                "total_citations": stats["total_citations"],
                "paper_count": paper_count,
                "h_index_approx": h_index,
                "avg_citations": round(stats["total_citations"] / paper_count, 1),
                "avg_velocity": round(avg_velocity, 3),
                "coauthor_count": len(stats["coauthors"]),
                "citations_per_coauthor": round(
                    stats["total_citations"] / max(len(stats["coauthors"]), 1), 1
                ),
            })

        influential.sort(key=lambda x: x["influence_score"], reverse=True)
        return influential[:top_n]

    def find_potential_collaborators(
        self,
        your_interests: List[str],
        papers: List[Dict],
        top_n: int = 50,
        exclude_authors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Find authors working on topics you care about.

        Args:
            your_interests: List of keywords/topics of interest
            papers: List of paper dicts to search
            top_n: Number of collaborators to return
            exclude_authors: Authors to exclude (e.g., yourself)

        Returns:
            List of potential collaborators with relevance scores
        """
        exclude_set = set(exclude_authors or [])
        interest_patterns = [re.compile(rf"\b{i}\b", re.IGNORECASE) for i in your_interests]

        author_relevance = defaultdict(lambda: {
            "matching_papers": [],
            "interest_matches": Counter(),
            "total_citations": 0,
        })

        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text = f"{title} {abstract}"

            # Check which interests match
            matches = []
            for i, pattern in enumerate(interest_patterns):
                if pattern.search(text):
                    matches.append(your_interests[i])

            if not matches:
                continue

            authors = paper.get("author_list", [])
            citations = paper.get("citations") or 0

            for author in authors:
                if author in exclude_set:
                    continue

                author_relevance[author]["matching_papers"].append({
                    "title": title,
                    "arxiv_id": paper.get("arxiv_id"),
                    "matches": matches,
                    "citations": citations,
                })
                author_relevance[author]["interest_matches"].update(matches)
                author_relevance[author]["total_citations"] += citations

        collaborators = []

        for author, data in author_relevance.items():
            paper_count = len(data["matching_papers"])

            # Compute relevance score
            interest_coverage = len(data["interest_matches"]) / len(your_interests)
            paper_relevance = sum(
                len(p["matches"]) / len(your_interests)
                for p in data["matching_papers"]
            ) / paper_count

            relevance_score = (
                interest_coverage * 3 +
                paper_relevance * 2 +
                math.log(1 + data["total_citations"]) * 0.5 +
                math.log(1 + paper_count) * 0.5
            )

            collaborators.append({
                "name": author,
                "relevance_score": round(relevance_score, 2),
                "matching_paper_count": paper_count,
                "interests_covered": list(data["interest_matches"].keys()),
                "interest_coverage": round(interest_coverage * 100, 1),
                "total_citations_in_area": data["total_citations"],
                "top_papers": sorted(
                    data["matching_papers"],
                    key=lambda x: x["citations"],
                    reverse=True
                )[:3],
            })

        collaborators.sort(key=lambda x: x["relevance_score"], reverse=True)
        return collaborators[:top_n]

    def compute_author_momentum(self, author_name_or_id: str) -> Dict:
        """
        Compute whether an author's impact is growing or declining.

        Args:
            author_name_or_id: Author name or Semantic Scholar ID

        Returns:
            Dict with momentum metrics
        """
        analysis = self.analyze_author(author_name_or_id)

        if "error" in analysis:
            return analysis

        momentum = analysis.get("momentum", {})
        citation_growth = analysis.get("citation_growth", {})

        return {
            "author": analysis.get("name"),
            "author_id": analysis.get("author_id"),
            "combined_momentum": momentum.get("combined_momentum", 0),
            "productivity_momentum": momentum.get("productivity_momentum", 0),
            "citation_momentum": momentum.get("citation_momentum", 0),
            "trend": momentum.get("trend", "unknown"),
            "citation_trend": citation_growth.get("citation_trend", "unknown"),
            "h_index": analysis.get("h_index"),
            "career_stage": analysis.get("career_stage", {}).get("stage"),
        }

    def find_author_clusters(
        self,
        papers: List[Dict],
        min_cluster_size: int = 3,
        min_shared_papers: int = 2,
    ) -> Dict:
        """
        Find groups of authors that frequently collaborate.

        Uses a simple clustering approach based on co-authorship edges.

        Args:
            papers: List of paper dicts
            min_cluster_size: Minimum authors to form a cluster
            min_shared_papers: Minimum papers shared to consider connected

        Returns:
            Dict containing author clusters
        """
        # Build co-authorship graph with edge weights
        coauthor_counts = defaultdict(Counter)

        for paper in papers:
            authors = paper.get("author_list", [])

            for i, a1 in enumerate(authors):
                for a2 in authors[i + 1:]:
                    coauthor_counts[a1][a2] += 1
                    coauthor_counts[a2][a1] += 1

        # Build graph with only strong edges
        graph = defaultdict(set)

        for author, coauthors in coauthor_counts.items():
            for coauthor, count in coauthors.items():
                if count >= min_shared_papers:
                    graph[author].add(coauthor)
                    graph[coauthor].add(author)

        # Find connected components
        visited = set()
        clusters = []

        for author in graph:
            if author in visited:
                continue

            # BFS to find component
            component = set()
            queue = [author]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= min_cluster_size:
                clusters.append(component)

        # Analyze each cluster
        cluster_info = []

        for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
            # Compute cluster metrics
            cluster_papers = set()
            total_citations = 0

            for paper in papers:
                authors = set(paper.get("author_list", []))
                if authors & cluster:
                    cluster_papers.add(paper.get("arxiv_id"))
                    total_citations += paper.get("citations") or 0

            # Find most central authors (highest degree within cluster)
            internal_degrees = {
                a: len(graph[a] & cluster) for a in cluster
            }
            central_authors = sorted(
                internal_degrees.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            cluster_info.append({
                "cluster_id": i + 1,
                "size": len(cluster),
                "members": list(cluster)[:20],  # Limit for readability
                "paper_count": len(cluster_papers),
                "total_citations": total_citations,
                "central_authors": [
                    {"name": a, "internal_connections": d}
                    for a, d in central_authors
                ],
                "cohesion": round(
                    sum(internal_degrees.values()) / (len(cluster) * (len(cluster) - 1)),
                    3
                ) if len(cluster) > 1 else 1.0,
            })

        return {
            "total_clusters": len(cluster_info),
            "clusters": cluster_info[:20],  # Top 20 clusters
            "largest_cluster_size": cluster_info[0]["size"] if cluster_info else 0,
            "total_clustered_authors": sum(c["size"] for c in cluster_info),
        }

    def build_from_papers(self, papers: List[Dict]) -> Dict:
        """
        Build internal data structures from a list of papers.

        This enables local analysis without API calls.

        Args:
            papers: List of paper dicts

        Returns:
            Summary of built data
        """
        self.author_papers.clear()
        self.coauthor_graph.clear()
        self.author_categories.clear()

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            authors = paper.get("author_list", [])
            category = paper.get("primary_category", "")

            for author in authors:
                self.author_papers[author].append(paper)
                self.author_categories[author][category] += 1

                for coauthor in authors:
                    if coauthor != author:
                        self.coauthor_graph[author].add(coauthor)

        return {
            "authors_indexed": len(self.author_papers),
            "total_papers": len(papers),
            "avg_papers_per_author": round(
                sum(len(p) for p in self.author_papers.values()) / len(self.author_papers), 2
            ) if self.author_papers else 0,
        }

    def _compute_trend(self, values: List[float]) -> str:
        """Compute simple linear trend direction."""
        if len(values) < 2:
            return "insufficient_data"

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"
