"""
Citation Trajectory Analysis Module
====================================
Track citation patterns over time to identify:
- Momentum papers (accelerating citations)
- Sleeping beauties (late recognition)
- Declining papers (past peak)
- Breakout papers (top velocity for age)

Uses Semantic Scholar API for citation data by year.
"""

import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import math


class CitationTrajectoryAnalyzer:
    """
    Analyze citation trajectories over time using Semantic Scholar API.

    Tracks temporal citation patterns to identify papers with interesting
    dynamics: gaining momentum, sleeping beauties, declining, or breaking out.
    """

    # Semantic Scholar API endpoints
    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
    S2_PAPER_ENDPOINT = f"{S2_API_BASE}/paper"
    S2_BATCH_ENDPOINT = f"{S2_API_BASE}/paper/batch"

    # Rate limiting settings
    DEFAULT_RATE_LIMIT = 1.0  # seconds between requests
    BATCH_SIZE = 100  # max papers per batch request

    def __init__(self, rate_limit: float = DEFAULT_RATE_LIMIT, api_key: Optional[str] = None):
        """
        Initialize the citation trajectory analyzer.

        Args:
            rate_limit: Seconds to wait between API calls (default 1.0)
            api_key: Optional Semantic Scholar API key for higher rate limits
        """
        self.rate_limit = rate_limit
        self.api_key = api_key
        self._last_request_time = 0
        self._cache = {}  # Cache for citation history data

    def _rate_limit_wait(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including API key if available."""
        headers = {"User-Agent": "arxiv-research-intelligence/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _fetch_paper_with_citations(self, paper_id: str, id_type: str = "ARXIV") -> Optional[Dict]:
        """
        Fetch a single paper with citation data from Semantic Scholar.

        Args:
            paper_id: Paper identifier (arXiv ID, DOI, or S2 Paper ID)
            id_type: Type of ID ("ARXIV", "DOI", or "S2")

        Returns:
            Paper data with citations or None if not found
        """
        self._rate_limit_wait()

        # Format paper ID for API
        if id_type == "ARXIV":
            api_id = f"ARXIV:{paper_id.split('v')[0]}"  # Remove version
        elif id_type == "DOI":
            api_id = f"DOI:{paper_id}"
        else:
            api_id = paper_id

        # Request paper with citations and year data
        fields = "citationCount,citations.year,year,title,publicationDate"
        url = f"{self.S2_PAPER_ENDPOINT}/{api_id}?fields={fields}"

        try:
            req = urllib.request.Request(url, headers=self._get_headers())
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            elif e.code == 429:
                # Rate limited - wait and retry once
                print(f"    Rate limited, waiting 10 seconds...")
                time.sleep(10)
                try:
                    req = urllib.request.Request(url, headers=self._get_headers())
                    with urllib.request.urlopen(req, timeout=30) as response:
                        return json.loads(response.read().decode("utf-8"))
                except Exception:
                    return None
            return None
        except Exception as e:
            print(f"    Error fetching {paper_id}: {e}")
            return None

    def _fetch_batch_papers(self, paper_ids: List[str], id_type: str = "ARXIV") -> Dict[str, Dict]:
        """
        Fetch multiple papers in a single batch request.

        Args:
            paper_ids: List of paper identifiers
            id_type: Type of IDs

        Returns:
            Dict mapping paper_id to paper data
        """
        results = {}

        for i in range(0, len(paper_ids), self.BATCH_SIZE):
            batch = paper_ids[i:i + self.BATCH_SIZE]
            self._rate_limit_wait()

            # Format IDs
            if id_type == "ARXIV":
                formatted_ids = [f"ARXIV:{pid.split('v')[0]}" for pid in batch]
            elif id_type == "DOI":
                formatted_ids = [f"DOI:{pid}" for pid in batch]
            else:
                formatted_ids = batch

            url = f"{self.S2_BATCH_ENDPOINT}?fields=citationCount,citations.year,year,title,publicationDate,paperId"

            try:
                request_data = json.dumps({"ids": formatted_ids}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=request_data,
                    headers={**self._get_headers(), "Content-Type": "application/json"},
                    method="POST"
                )

                with urllib.request.urlopen(req, timeout=60) as response:
                    batch_results = json.loads(response.read().decode("utf-8"))

                for j, result in enumerate(batch_results):
                    if result:
                        original_id = batch[j]
                        results[original_id] = result

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"    Rate limited on batch, falling back to individual requests...")
                    for pid in batch:
                        result = self._fetch_paper_with_citations(pid, id_type)
                        if result:
                            results[pid] = result
                else:
                    print(f"    Batch request error: {e}")
            except Exception as e:
                print(f"    Batch request error: {e}")

        return results

    def _extract_citation_history(self, paper_data: Dict) -> Dict[int, int]:
        """
        Extract yearly citation counts from paper data.

        Args:
            paper_data: Paper data from Semantic Scholar

        Returns:
            Dict mapping year to citation count for that year
        """
        citations_by_year = defaultdict(int)

        citations = paper_data.get("citations", [])
        for citation in citations:
            if citation and "year" in citation and citation["year"]:
                year = citation["year"]
                citations_by_year[year] += 1

        return dict(citations_by_year)

    def fetch_citation_history(self, papers: List[Dict]) -> Dict[str, Dict]:
        """
        Get yearly citation counts for each paper.

        Args:
            papers: List of paper dictionaries with 'arxiv_id' field

        Returns:
            Dict mapping arxiv_id to citation history:
            {
                "arxiv_id": {
                    "total_citations": int,
                    "publication_year": int,
                    "citations_by_year": {year: count},
                    "cumulative_by_year": {year: cumulative_count},
                    "has_history": bool
                }
            }
        """
        print(f"    Fetching citation history for {len(papers)} papers...")

        # Extract arXiv IDs
        arxiv_ids = [p.get("arxiv_id", "") for p in papers if p.get("arxiv_id")]

        # Batch fetch from Semantic Scholar
        paper_data = self._fetch_batch_papers(arxiv_ids, id_type="ARXIV")

        results = {}

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id:
                continue

            s2_data = paper_data.get(arxiv_id)

            if s2_data:
                citations_by_year = self._extract_citation_history(s2_data)
                total_citations = s2_data.get("citationCount", 0)

                # Get publication year
                pub_year = s2_data.get("year")
                if not pub_year and s2_data.get("publicationDate"):
                    try:
                        pub_year = int(s2_data["publicationDate"][:4])
                    except (ValueError, TypeError):
                        pub_year = None

                # Fall back to paper's published_date if available
                if not pub_year and paper.get("published_date"):
                    try:
                        pub_year = int(paper["published_date"][:4])
                    except (ValueError, TypeError):
                        pub_year = None

                # Compute cumulative citations by year
                cumulative_by_year = {}
                if citations_by_year:
                    sorted_years = sorted(citations_by_year.keys())
                    cumulative = 0
                    for year in sorted_years:
                        cumulative += citations_by_year[year]
                        cumulative_by_year[year] = cumulative

                results[arxiv_id] = {
                    "total_citations": total_citations,
                    "publication_year": pub_year,
                    "citations_by_year": citations_by_year,
                    "cumulative_by_year": cumulative_by_year,
                    "has_history": len(citations_by_year) > 0,
                }
            else:
                # Paper not found in Semantic Scholar
                results[arxiv_id] = {
                    "total_citations": paper.get("citations", 0) or 0,
                    "publication_year": None,
                    "citations_by_year": {},
                    "cumulative_by_year": {},
                    "has_history": False,
                }

        print(f"    Retrieved citation history for {sum(1 for r in results.values() if r['has_history'])} papers")
        return results

    def compute_citation_velocity(self, papers: List[Dict],
                                   citation_history: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
        """
        Compute citation velocity and acceleration for each paper.

        Velocity = citations per time period (monthly, yearly)
        Acceleration = change in velocity over time

        Args:
            papers: List of paper dictionaries
            citation_history: Pre-fetched citation history (optional)

        Returns:
            Dict mapping arxiv_id to velocity metrics:
            {
                "arxiv_id": {
                    "citations_per_month": float,
                    "citations_per_year": float,
                    "recent_velocity": float,  # Last 12 months
                    "early_velocity": float,   # First 12 months
                    "acceleration": float,     # Change in velocity
                    "velocity_trend": str,     # "accelerating", "stable", "decelerating"
                    "years_since_publication": float
                }
            }
        """
        print(f"    Computing citation velocity for {len(papers)} papers...")

        if citation_history is None:
            citation_history = self.fetch_citation_history(papers)

        current_year = datetime.now().year
        results = {}

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id or arxiv_id not in citation_history:
                continue

            history = citation_history[arxiv_id]
            citations_by_year = history.get("citations_by_year", {})
            total_citations = history.get("total_citations", 0)
            pub_year = history.get("publication_year")

            # Calculate age in years
            if pub_year:
                years_since_publication = current_year - pub_year
            else:
                # Fall back to paper age in days if available
                age_days = paper.get("paper_age_days")
                if age_days:
                    years_since_publication = age_days / 365.0
                else:
                    years_since_publication = 1  # Default to 1 year

            years_since_publication = max(0.1, years_since_publication)  # Avoid division by zero

            # Overall velocity
            citations_per_year = total_citations / years_since_publication
            citations_per_month = citations_per_year / 12.0

            # Recent vs early velocity (for papers with history)
            recent_velocity = 0.0
            early_velocity = 0.0

            if citations_by_year and pub_year:
                # Recent: last 2 years
                recent_years = [current_year - 1, current_year]
                recent_citations = sum(citations_by_year.get(y, 0) for y in recent_years)
                recent_velocity = recent_citations / 2.0

                # Early: first 2 years after publication
                early_years = [pub_year, pub_year + 1]
                early_citations = sum(citations_by_year.get(y, 0) for y in early_years)
                early_velocity = early_citations / 2.0

            # Acceleration
            acceleration = recent_velocity - early_velocity

            # Determine trend
            if acceleration > early_velocity * 0.5 and acceleration > 1:
                velocity_trend = "accelerating"
            elif acceleration < -early_velocity * 0.5 and acceleration < -1:
                velocity_trend = "decelerating"
            else:
                velocity_trend = "stable"

            results[arxiv_id] = {
                "citations_per_month": round(citations_per_month, 3),
                "citations_per_year": round(citations_per_year, 2),
                "recent_velocity": round(recent_velocity, 2),
                "early_velocity": round(early_velocity, 2),
                "acceleration": round(acceleration, 2),
                "velocity_trend": velocity_trend,
                "years_since_publication": round(years_since_publication, 1),
            }

        return results

    def identify_momentum_papers(self, papers: List[Dict],
                                  citation_history: Optional[Dict[str, Dict]] = None,
                                  min_citations: int = 5,
                                  top_n: int = 50) -> List[Dict]:
        """
        Identify papers with accelerating citations (gaining momentum NOW).

        These are papers where recent citation rate exceeds early citation rate,
        indicating growing interest.

        Args:
            papers: List of paper dictionaries
            citation_history: Pre-fetched citation history (optional)
            min_citations: Minimum total citations to consider
            top_n: Number of top papers to return

        Returns:
            List of momentum papers sorted by acceleration
        """
        print(f"    Identifying momentum papers from {len(papers)} papers...")

        if citation_history is None:
            citation_history = self.fetch_citation_history(papers)

        velocity_data = self.compute_citation_velocity(papers, citation_history)

        momentum_papers = []

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id not in velocity_data or arxiv_id not in citation_history:
                continue

            history = citation_history[arxiv_id]
            velocity = velocity_data[arxiv_id]

            total_citations = history.get("total_citations", 0)

            # Filter by minimum citations
            if total_citations < min_citations:
                continue

            # Look for accelerating papers
            if velocity["velocity_trend"] == "accelerating" and velocity["acceleration"] > 0:
                # Compute momentum score
                # Higher = more recent velocity relative to early velocity
                if velocity["early_velocity"] > 0:
                    momentum_ratio = velocity["recent_velocity"] / velocity["early_velocity"]
                else:
                    momentum_ratio = velocity["recent_velocity"] + 1

                momentum_score = velocity["acceleration"] * math.log(1 + momentum_ratio)

                momentum_papers.append({
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                    "total_citations": total_citations,
                    "recent_velocity": velocity["recent_velocity"],
                    "early_velocity": velocity["early_velocity"],
                    "acceleration": velocity["acceleration"],
                    "momentum_score": round(momentum_score, 2),
                    "years_since_publication": velocity["years_since_publication"],
                    "category": paper.get("primary_category", ""),
                    "has_code": paper.get("has_code", False),
                })

        # Sort by momentum score
        momentum_papers.sort(key=lambda x: x["momentum_score"], reverse=True)

        print(f"    Found {len(momentum_papers)} momentum papers")
        return momentum_papers[:top_n]

    def identify_sleeping_beauties(self, papers: List[Dict],
                                    citation_history: Optional[Dict[str, Dict]] = None,
                                    min_age_years: int = 3,
                                    min_citations: int = 20,
                                    awakening_threshold: float = 3.0,
                                    top_n: int = 30) -> List[Dict]:
        """
        Identify sleeping beauties - papers that were ignored then suddenly took off.

        A sleeping beauty is characterized by:
        - Low early citations (sleep period)
        - Sudden increase in later citations (awakening)

        Args:
            papers: List of paper dictionaries
            citation_history: Pre-fetched citation history (optional)
            min_age_years: Minimum paper age in years
            min_citations: Minimum total citations
            awakening_threshold: Ratio of recent/early velocity to qualify as awakening
            top_n: Number of top papers to return

        Returns:
            List of sleeping beauty papers
        """
        print(f"    Identifying sleeping beauties from {len(papers)} papers...")

        if citation_history is None:
            citation_history = self.fetch_citation_history(papers)

        velocity_data = self.compute_citation_velocity(papers, citation_history)
        current_year = datetime.now().year

        sleeping_beauties = []

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id not in velocity_data or arxiv_id not in citation_history:
                continue

            history = citation_history[arxiv_id]
            velocity = velocity_data[arxiv_id]

            total_citations = history.get("total_citations", 0)
            pub_year = history.get("publication_year")
            citations_by_year = history.get("citations_by_year", {})

            # Apply filters
            if total_citations < min_citations:
                continue
            if not pub_year or (current_year - pub_year) < min_age_years:
                continue
            if not citations_by_year:
                continue

            # Calculate sleep and awakening periods
            years = sorted(citations_by_year.keys())
            if len(years) < 2:
                continue

            # Sleep period: first half of years
            mid_year = pub_year + (current_year - pub_year) // 2
            sleep_citations = sum(citations_by_year.get(y, 0) for y in years if y <= mid_year)
            awake_citations = sum(citations_by_year.get(y, 0) for y in years if y > mid_year)

            # Count years in each period
            sleep_years = len([y for y in years if y <= mid_year])
            awake_years = len([y for y in years if y > mid_year])

            if sleep_years == 0 or awake_years == 0:
                continue

            sleep_velocity = sleep_citations / sleep_years
            awake_velocity = awake_citations / awake_years

            # Check for awakening pattern
            if sleep_velocity > 0:
                awakening_ratio = awake_velocity / sleep_velocity
            else:
                awakening_ratio = awake_velocity + 1 if awake_velocity > 0 else 0

            if awakening_ratio >= awakening_threshold:
                # Calculate beauty score (how dramatic was the awakening)
                beauty_score = awakening_ratio * math.log(1 + total_citations)

                sleeping_beauties.append({
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                    "total_citations": total_citations,
                    "publication_year": pub_year,
                    "sleep_velocity": round(sleep_velocity, 2),
                    "awake_velocity": round(awake_velocity, 2),
                    "awakening_ratio": round(awakening_ratio, 2),
                    "beauty_score": round(beauty_score, 2),
                    "years_since_publication": velocity["years_since_publication"],
                    "category": paper.get("primary_category", ""),
                })

        # Sort by beauty score
        sleeping_beauties.sort(key=lambda x: x["beauty_score"], reverse=True)

        print(f"    Found {len(sleeping_beauties)} sleeping beauties")
        return sleeping_beauties[:top_n]

    def identify_declining_papers(self, papers: List[Dict],
                                   citation_history: Optional[Dict[str, Dict]] = None,
                                   min_peak_citations: int = 50,
                                   decline_threshold: float = 0.5,
                                   top_n: int = 30) -> List[Dict]:
        """
        Identify papers that peaked and are now declining in citations.

        Args:
            papers: List of paper dictionaries
            citation_history: Pre-fetched citation history (optional)
            min_peak_citations: Minimum citations at peak year
            decline_threshold: Ratio of recent to peak velocity below which = declining
            top_n: Number of top papers to return

        Returns:
            List of declining papers sorted by decline magnitude
        """
        print(f"    Identifying declining papers from {len(papers)} papers...")

        if citation_history is None:
            citation_history = self.fetch_citation_history(papers)

        velocity_data = self.compute_citation_velocity(papers, citation_history)
        current_year = datetime.now().year

        declining_papers = []

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id not in velocity_data or arxiv_id not in citation_history:
                continue

            history = citation_history[arxiv_id]
            velocity = velocity_data[arxiv_id]

            citations_by_year = history.get("citations_by_year", {})

            if not citations_by_year or len(citations_by_year) < 3:
                continue

            # Find peak year
            peak_year = max(citations_by_year.keys(), key=lambda y: citations_by_year[y])
            peak_citations = citations_by_year[peak_year]

            if peak_citations < min_peak_citations:
                continue

            # Compare recent citations to peak
            recent_years = [current_year - 1, current_year]
            recent_citations = sum(citations_by_year.get(y, 0) for y in recent_years)
            recent_velocity = recent_citations / 2.0

            # Peak velocity (average of peak year and adjacent years)
            peak_adjacent = [peak_year - 1, peak_year, peak_year + 1]
            peak_window_citations = sum(citations_by_year.get(y, 0) for y in peak_adjacent)
            peak_velocity = peak_window_citations / 3.0

            if peak_velocity > 0:
                decline_ratio = recent_velocity / peak_velocity
            else:
                continue

            # Check if declining
            if decline_ratio < decline_threshold and peak_year < current_year - 1:
                decline_magnitude = peak_velocity - recent_velocity

                declining_papers.append({
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                    "total_citations": history.get("total_citations", 0),
                    "peak_year": peak_year,
                    "peak_citations": peak_citations,
                    "peak_velocity": round(peak_velocity, 2),
                    "recent_velocity": round(recent_velocity, 2),
                    "decline_ratio": round(decline_ratio, 2),
                    "decline_magnitude": round(decline_magnitude, 2),
                    "years_since_peak": current_year - peak_year,
                    "category": paper.get("primary_category", ""),
                })

        # Sort by decline magnitude
        declining_papers.sort(key=lambda x: x["decline_magnitude"], reverse=True)

        print(f"    Found {len(declining_papers)} declining papers")
        return declining_papers[:top_n]

    def predict_future_citations(self, paper: Dict,
                                  citation_history: Optional[Dict[str, Dict]] = None,
                                  years_ahead: int = 2) -> Dict:
        """
        Simple trend extrapolation to predict future citations.

        Uses linear regression on recent citation velocity to estimate
        future citation accumulation.

        Args:
            paper: Paper dictionary with 'arxiv_id' field
            citation_history: Pre-fetched citation history (optional)
            years_ahead: Number of years to predict

        Returns:
            Prediction dictionary with estimated citations and confidence
        """
        arxiv_id = paper.get("arxiv_id", "")

        if citation_history is None:
            citation_history = self.fetch_citation_history([paper])

        if arxiv_id not in citation_history:
            return {"error": "No citation history available"}

        history = citation_history[arxiv_id]
        citations_by_year = history.get("citations_by_year", {})
        total_citations = history.get("total_citations", 0)

        if not citations_by_year or len(citations_by_year) < 2:
            # No history - use simple velocity extrapolation
            velocity_data = self.compute_citation_velocity([paper], citation_history)
            if arxiv_id in velocity_data:
                annual_velocity = velocity_data[arxiv_id]["citations_per_year"]
            else:
                annual_velocity = 0

            predictions = []
            current = total_citations
            for i in range(1, years_ahead + 1):
                current += annual_velocity
                predictions.append(round(current))

            return {
                "arxiv_id": arxiv_id,
                "current_citations": total_citations,
                "predicted_citations": predictions,
                "years_ahead": years_ahead,
                "method": "simple_velocity",
                "confidence": "low",
            }

        # Linear regression on yearly counts
        years = sorted(citations_by_year.keys())
        values = [citations_by_year[y] for y in years]

        trend = self._calculate_trend(values)

        # Extrapolate
        current_year = datetime.now().year
        last_value = values[-1]

        predictions = []
        for i in range(1, years_ahead + 1):
            predicted = max(0, last_value + trend["slope"] * i)
            predictions.append(round(predicted))

        # Cumulative prediction
        cumulative_predictions = []
        cumulative = total_citations
        for p in predictions:
            cumulative += p
            cumulative_predictions.append(round(cumulative))

        return {
            "arxiv_id": arxiv_id,
            "current_citations": total_citations,
            "predicted_annual_citations": predictions,
            "predicted_cumulative_citations": cumulative_predictions,
            "years_ahead": years_ahead,
            "trend_slope": trend["slope"],
            "trend_direction": trend["direction"],
            "method": "linear_regression",
            "confidence": "low" if trend["r_squared"] < 0.3 else "medium" if trend["r_squared"] < 0.7 else "high",
            "r_squared": trend["r_squared"],
        }

    def find_breakout_papers(self, papers: List[Dict],
                              citation_history: Optional[Dict[str, Dict]] = None,
                              top_percentile: float = 0.1,
                              min_citations: int = 10) -> List[Dict]:
        """
        Find papers in top 10% velocity for their age cohort.

        Args:
            papers: List of paper dictionaries
            citation_history: Pre-fetched citation history (optional)
            top_percentile: Percentile threshold (0.1 = top 10%)
            min_citations: Minimum citations to consider

        Returns:
            List of breakout papers
        """
        print(f"    Finding breakout papers from {len(papers)} papers...")

        if citation_history is None:
            citation_history = self.fetch_citation_history(papers)

        velocity_data = self.compute_citation_velocity(papers, citation_history)

        # Group papers by age cohort (year bins)
        cohorts = defaultdict(list)

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id not in velocity_data or arxiv_id not in citation_history:
                continue

            history = citation_history[arxiv_id]
            velocity = velocity_data[arxiv_id]

            if history.get("total_citations", 0) < min_citations:
                continue

            age_years = velocity.get("years_since_publication", 1)

            # Bin by age (0-1, 1-2, 2-3, 3-5, 5+)
            if age_years < 1:
                cohort = "0-1 years"
            elif age_years < 2:
                cohort = "1-2 years"
            elif age_years < 3:
                cohort = "2-3 years"
            elif age_years < 5:
                cohort = "3-5 years"
            else:
                cohort = "5+ years"

            cohorts[cohort].append({
                "paper": paper,
                "velocity": velocity,
                "history": history,
            })

        breakout_papers = []

        for cohort, cohort_papers in cohorts.items():
            if len(cohort_papers) < 5:
                continue

            # Sort by velocity
            sorted_cohort = sorted(
                cohort_papers,
                key=lambda x: x["velocity"]["citations_per_year"],
                reverse=True
            )

            # Take top percentile
            n_top = max(1, int(len(sorted_cohort) * top_percentile))

            for item in sorted_cohort[:n_top]:
                paper = item["paper"]
                velocity = item["velocity"]
                history = item["history"]

                # Calculate percentile rank within cohort
                velocities = [x["velocity"]["citations_per_year"] for x in sorted_cohort]
                rank = velocities.index(velocity["citations_per_year"]) + 1
                percentile = round((1 - rank / len(velocities)) * 100, 1)

                breakout_papers.append({
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "title": paper.get("title", ""),
                    "total_citations": history.get("total_citations", 0),
                    "citations_per_year": velocity["citations_per_year"],
                    "citations_per_month": velocity["citations_per_month"],
                    "age_cohort": cohort,
                    "cohort_percentile": percentile,
                    "cohort_size": len(sorted_cohort),
                    "years_since_publication": velocity["years_since_publication"],
                    "category": paper.get("primary_category", ""),
                    "has_code": paper.get("has_code", False),
                })

        # Sort by velocity
        breakout_papers.sort(key=lambda x: x["citations_per_year"], reverse=True)

        print(f"    Found {len(breakout_papers)} breakout papers")
        return breakout_papers

    def compute_trajectory_metrics(self, paper: Dict,
                                    citation_history: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        Compute comprehensive trajectory metrics for a single paper.

        Args:
            paper: Paper dictionary
            citation_history: Pre-fetched citation history (optional)

        Returns:
            Dictionary with all trajectory metrics
        """
        arxiv_id = paper.get("arxiv_id", "")

        if citation_history is None:
            citation_history = self.fetch_citation_history([paper])

        if arxiv_id not in citation_history:
            return {"error": "Paper not found"}

        history = citation_history[arxiv_id]
        citations_by_year = history.get("citations_by_year", {})
        total_citations = history.get("total_citations", 0)
        pub_year = history.get("publication_year")
        current_year = datetime.now().year

        # Basic velocity
        velocity_data = self.compute_citation_velocity([paper], citation_history)
        velocity = velocity_data.get(arxiv_id, {})

        # Peak year
        if citations_by_year:
            peak_year = max(citations_by_year.keys(), key=lambda y: citations_by_year[y])
            peak_citations = citations_by_year[peak_year]
        else:
            peak_year = None
            peak_citations = 0

        # Time to N citations milestones
        time_to_milestones = {}
        cumulative = history.get("cumulative_by_year", {})
        for milestone in [10, 50, 100, 500, 1000]:
            if total_citations >= milestone:
                for year in sorted(cumulative.keys()):
                    if cumulative[year] >= milestone:
                        if pub_year:
                            time_to_milestones[milestone] = year - pub_year
                        break

        return {
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "total_citations": total_citations,
            "publication_year": pub_year,
            "citations_per_month": velocity.get("citations_per_month", 0),
            "citations_per_year": velocity.get("citations_per_year", 0),
            "recent_velocity": velocity.get("recent_velocity", 0),
            "early_velocity": velocity.get("early_velocity", 0),
            "acceleration": velocity.get("acceleration", 0),
            "velocity_trend": velocity.get("velocity_trend", "unknown"),
            "peak_year": peak_year,
            "peak_citations": peak_citations,
            "years_since_publication": velocity.get("years_since_publication", 0),
            "years_since_peak": (current_year - peak_year) if peak_year else None,
            "time_to_milestones": time_to_milestones,
            "citations_by_year": citations_by_year,
        }

    def visualize_trajectories(self, papers: List[Dict],
                                output_path: str,
                                citation_history: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        Generate trajectory data for plotting/visualization.

        Args:
            papers: List of paper dictionaries
            output_path: Path to save JSON data file
            citation_history: Pre-fetched citation history (optional)

        Returns:
            Dictionary with visualization data
        """
        print(f"    Generating trajectory visualization data for {len(papers)} papers...")

        if citation_history is None:
            citation_history = self.fetch_citation_history(papers)

        velocity_data = self.compute_citation_velocity(papers, citation_history)

        # Prepare visualization data
        trajectories = []

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            if arxiv_id not in citation_history:
                continue

            history = citation_history[arxiv_id]
            velocity = velocity_data.get(arxiv_id, {})

            citations_by_year = history.get("citations_by_year", {})
            cumulative_by_year = history.get("cumulative_by_year", {})

            if not citations_by_year:
                continue

            # Build time series
            years = sorted(citations_by_year.keys())

            trajectories.append({
                "arxiv_id": arxiv_id,
                "title": paper.get("title", "")[:100],
                "category": paper.get("primary_category", ""),
                "publication_year": history.get("publication_year"),
                "total_citations": history.get("total_citations", 0),
                "velocity_trend": velocity.get("velocity_trend", "unknown"),
                "years": years,
                "annual_citations": [citations_by_year[y] for y in years],
                "cumulative_citations": [cumulative_by_year.get(y, 0) for y in years],
            })

        # Aggregate statistics
        current_year = datetime.now().year
        all_years = sorted(set(
            y for t in trajectories for y in t["years"]
        ))

        yearly_totals = {y: 0 for y in all_years}
        for t in trajectories:
            for i, year in enumerate(t["years"]):
                yearly_totals[year] += t["annual_citations"][i]

        visualization_data = {
            "trajectories": trajectories,
            "summary": {
                "total_papers": len(trajectories),
                "years_covered": all_years,
                "yearly_citation_totals": yearly_totals,
            },
            "generated_at": datetime.now().isoformat(),
        }

        # Save to file
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(visualization_data, f, indent=2)

        print(f"    Saved visualization data to {output_path}")

        return visualization_data

    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate linear trend using least squares."""
        n = len(values)
        if n < 2:
            return {"direction": "stable", "slope": 0, "r_squared": 0}

        # Simple linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # R-squared
        y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Direction
        if slope > 0.5 and r_squared > 0.2:
            direction = "increasing"
        elif slope < -0.5 and r_squared > 0.2:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope": round(slope, 4),
            "r_squared": round(max(0, r_squared), 4),
        }

    def analyze_trajectories(self, papers: List[Dict]) -> Dict:
        """
        Run comprehensive trajectory analysis on a set of papers.

        Args:
            papers: List of paper dictionaries with 'arxiv_id' field

        Returns:
            Dictionary with all trajectory analysis results
        """
        print(f"    Running trajectory analysis for {len(papers)} papers...")

        # Fetch citation history once
        citation_history = self.fetch_citation_history(papers)

        # Compute all analyses
        velocity_data = self.compute_citation_velocity(papers, citation_history)
        momentum = self.identify_momentum_papers(papers, citation_history)
        sleeping = self.identify_sleeping_beauties(papers, citation_history)
        declining = self.identify_declining_papers(papers, citation_history)
        breakout = self.find_breakout_papers(papers, citation_history)

        # Summary statistics
        papers_with_history = sum(1 for h in citation_history.values() if h.get("has_history"))

        accelerating = sum(1 for v in velocity_data.values() if v.get("velocity_trend") == "accelerating")
        stable = sum(1 for v in velocity_data.values() if v.get("velocity_trend") == "stable")
        decelerating = sum(1 for v in velocity_data.values() if v.get("velocity_trend") == "decelerating")

        return {
            "summary": {
                "total_papers_analyzed": len(papers),
                "papers_with_citation_history": papers_with_history,
                "velocity_distribution": {
                    "accelerating": accelerating,
                    "stable": stable,
                    "decelerating": decelerating,
                },
            },
            "momentum_papers": momentum,
            "sleeping_beauties": sleeping,
            "declining_papers": declining,
            "breakout_papers": breakout,
            "velocity_data": velocity_data,
            "citation_history": citation_history,
        }
