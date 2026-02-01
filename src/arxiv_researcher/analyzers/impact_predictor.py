"""
Impact Prediction Module
========================
Analyze what predicts paper success:
- Feature importance for citations
- Success factor identification
- Paper scoring
"""

from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import math


class ImpactPredictor:
    """Analyze and predict paper impact."""

    def __init__(self):
        self.feature_correlations = {}

    def analyze_success_factors(self, papers: List[Dict]) -> Dict:
        """Analyze what factors correlate with paper success."""
        print(f"    Analyzing success factors for {len(papers)} papers...")

        # Filter to papers with citations data
        papers_with_citations = [p for p in papers if p.get("citations") is not None]

        if len(papers_with_citations) < 50:
            return {"error": "Not enough papers with citation data"}

        return {
            "feature_analysis": self._analyze_features(papers_with_citations),
            "category_success": self._analyze_category_success(papers_with_citations),
            "method_success": self._analyze_method_success(papers_with_citations),
            "author_success": self._analyze_author_success(papers_with_citations),
            "structural_success": self._analyze_structural_factors(papers_with_citations),
            "venue_success": self._analyze_venue_success(papers_with_citations),
            "success_formula": self._derive_success_formula(papers_with_citations),
        }

    def _analyze_features(self, papers: List[Dict]) -> Dict:
        """Analyze correlation of features with citations."""
        features = {}

        # Binary features
        binary_features = [
            ("has_code", "has_code"),
            ("is_open_access", "is_open_access"),
            ("is_published", "is_published"),
            ("is_cross_listed", "is_cross_listed"),
            ("is_top_venue", "is_top_venue"),
            ("claims_sota", "claims_sota"),
            ("has_theoretical_contribution", "has_theoretical_contribution"),
        ]

        for name, field in binary_features:
            with_feature = [p["citations"] for p in papers if p.get(field)]
            without_feature = [p["citations"] for p in papers if not p.get(field)]

            if with_feature and without_feature:
                avg_with = sum(with_feature) / len(with_feature)
                avg_without = sum(without_feature) / len(without_feature)
                lift = avg_with / max(avg_without, 0.1)

                features[name] = {
                    "papers_with_feature": len(with_feature),
                    "papers_without_feature": len(without_feature),
                    "avg_citations_with": round(avg_with, 2),
                    "avg_citations_without": round(avg_without, 2),
                    "citation_lift": round(lift, 2),
                    "impact": "positive" if lift > 1.2 else "negative" if lift < 0.8 else "neutral",
                }

        # Numeric features (bucketize and compare)
        numeric_features = [
            ("author_count", "author_count"),
            ("category_count", "category_count"),
            ("abstract_word_count", "abstract_word_count"),
            ("page_count", "page_count"),
            ("max_author_h_index", "max_author_h_index"),
        ]

        for name, field in numeric_features:
            values = [(p.get(field) or 0, p["citations"]) for p in papers if p.get(field)]
            if len(values) < 20:
                continue

            # Sort by feature value
            values.sort(key=lambda x: x[0])
            n = len(values)

            # Compare quartiles
            q1 = values[:n//4]
            q4 = values[3*n//4:]

            avg_q1 = sum(v[1] for v in q1) / len(q1) if q1 else 0
            avg_q4 = sum(v[1] for v in q4) / len(q4) if q4 else 0

            features[name] = {
                "low_value_avg_citations": round(avg_q1, 2),
                "high_value_avg_citations": round(avg_q4, 2),
                "high_vs_low_ratio": round(avg_q4 / max(avg_q1, 0.1), 2),
                "correlation": self._simple_correlation(values),
            }

        return features

    def _analyze_category_success(self, papers: List[Dict]) -> Dict:
        """Analyze success by category."""
        by_category = defaultdict(list)

        for paper in papers:
            cat = paper.get("primary_category", "")
            by_category[cat].append(paper["citations"])

        category_stats = []
        for cat, citations in by_category.items():
            if len(citations) >= 10:
                category_stats.append({
                    "category": cat,
                    "paper_count": len(citations),
                    "avg_citations": round(sum(citations) / len(citations), 2),
                    "max_citations": max(citations),
                    "median_citations": sorted(citations)[len(citations)//2],
                    "top_10_pct_threshold": sorted(citations, reverse=True)[len(citations)//10] if len(citations) >= 10 else 0,
                })

        category_stats.sort(key=lambda x: x["avg_citations"], reverse=True)

        return {
            "by_category": category_stats,
            "highest_avg": category_stats[:10] if category_stats else [],
            "most_competitive": sorted(category_stats, key=lambda x: x["top_10_pct_threshold"], reverse=True)[:10],
        }

    def _analyze_method_success(self, papers: List[Dict]) -> Dict:
        """Analyze which methods correlate with higher citations."""
        method_citations = defaultdict(list)

        for paper in papers:
            methods = paper.get("methods_detected", [])
            for method in methods:
                method_citations[method].append(paper["citations"])

        method_stats = []
        for method, citations in method_citations.items():
            if len(citations) >= 10:
                method_stats.append({
                    "method": method,
                    "paper_count": len(citations),
                    "avg_citations": round(sum(citations) / len(citations), 2),
                    "max_citations": max(citations),
                })

        method_stats.sort(key=lambda x: x["avg_citations"], reverse=True)

        return {
            "by_method": method_stats,
            "highest_impact_methods": method_stats[:20] if method_stats else [],
        }

    def _analyze_author_success(self, papers: List[Dict]) -> Dict:
        """Analyze how author metrics correlate with paper success."""
        # Group by author h-index buckets
        h_index_buckets = {
            "0-5": [],
            "6-15": [],
            "16-30": [],
            "31-50": [],
            "51+": [],
        }

        for paper in papers:
            h_index = paper.get("max_author_h_index") or 0
            citations = paper["citations"]

            if h_index <= 5:
                h_index_buckets["0-5"].append(citations)
            elif h_index <= 15:
                h_index_buckets["6-15"].append(citations)
            elif h_index <= 30:
                h_index_buckets["16-30"].append(citations)
            elif h_index <= 50:
                h_index_buckets["31-50"].append(citations)
            else:
                h_index_buckets["51+"].append(citations)

        bucket_stats = {}
        for bucket, citations in h_index_buckets.items():
            if citations:
                bucket_stats[bucket] = {
                    "paper_count": len(citations),
                    "avg_citations": round(sum(citations) / len(citations), 2),
                }

        # Team size analysis
        team_size_citations = defaultdict(list)
        for paper in papers:
            team_size = paper.get("author_count") or 1
            if team_size == 1:
                bucket = "solo"
            elif team_size <= 3:
                bucket = "small (2-3)"
            elif team_size <= 6:
                bucket = "medium (4-6)"
            else:
                bucket = "large (7+)"
            team_size_citations[bucket].append(paper["citations"])

        team_stats = {
            bucket: {
                "paper_count": len(cites),
                "avg_citations": round(sum(cites) / len(cites), 2),
            }
            for bucket, cites in team_size_citations.items()
            if cites
        }

        return {
            "by_author_h_index": bucket_stats,
            "by_team_size": team_stats,
            "insight": self._derive_author_insight(bucket_stats, team_stats),
        }

    def _analyze_structural_factors(self, papers: List[Dict]) -> Dict:
        """Analyze how paper structure correlates with success."""
        # Abstract length
        length_buckets = {
            "short (<100 words)": [],
            "medium (100-200)": [],
            "long (200-300)": [],
            "very long (300+)": [],
        }

        for paper in papers:
            word_count = paper.get("abstract_word_count") or 0
            citations = paper["citations"]

            if word_count < 100:
                length_buckets["short (<100 words)"].append(citations)
            elif word_count < 200:
                length_buckets["medium (100-200)"].append(citations)
            elif word_count < 300:
                length_buckets["long (200-300)"].append(citations)
            else:
                length_buckets["very long (300+)"].append(citations)

        length_stats = {
            bucket: {
                "paper_count": len(cites),
                "avg_citations": round(sum(cites) / len(cites), 2) if cites else 0,
            }
            for bucket, cites in length_buckets.items()
        }

        # Title characteristics
        title_stats = {
            "has_colon": {"with": [], "without": []},
            "has_question": {"with": [], "without": []},
            "short_title": {"with": [], "without": []},
        }

        for paper in papers:
            title = paper.get("title", "")
            citations = paper["citations"]

            if ":" in title:
                title_stats["has_colon"]["with"].append(citations)
            else:
                title_stats["has_colon"]["without"].append(citations)

            if "?" in title:
                title_stats["has_question"]["with"].append(citations)
            else:
                title_stats["has_question"]["without"].append(citations)

            if len(title.split()) <= 8:
                title_stats["short_title"]["with"].append(citations)
            else:
                title_stats["short_title"]["without"].append(citations)

        title_analysis = {}
        for feature, groups in title_stats.items():
            if groups["with"] and groups["without"]:
                avg_with = sum(groups["with"]) / len(groups["with"])
                avg_without = sum(groups["without"]) / len(groups["without"])
                title_analysis[feature] = {
                    "avg_with": round(avg_with, 2),
                    "avg_without": round(avg_without, 2),
                    "lift": round(avg_with / max(avg_without, 0.1), 2),
                }

        return {
            "by_abstract_length": length_stats,
            "title_characteristics": title_analysis,
        }

    def _analyze_venue_success(self, papers: List[Dict]) -> Dict:
        """Analyze how venue correlates with success."""
        venue_citations = defaultdict(list)

        for paper in papers:
            venue = paper.get("publication_venue", "")
            if venue:
                venue_citations[venue].append(paper["citations"])

        venue_stats = []
        for venue, citations in venue_citations.items():
            if len(citations) >= 5:
                venue_stats.append({
                    "venue": venue,
                    "paper_count": len(citations),
                    "avg_citations": round(sum(citations) / len(citations), 2),
                    "total_citations": sum(citations),
                })

        venue_stats.sort(key=lambda x: x["avg_citations"], reverse=True)

        return {
            "by_venue": venue_stats[:30],
            "top_venues": venue_stats[:10],
        }

    def _derive_success_formula(self, papers: List[Dict]) -> Dict:
        """Derive a simple success prediction formula."""
        # Compute average citation impact of each factor
        factors = {}

        # Has code
        with_code = [p["citations"] for p in papers if p.get("has_code")]
        without_code = [p["citations"] for p in papers if not p.get("has_code")]
        if with_code and without_code:
            factors["has_code"] = round(
                (sum(with_code)/len(with_code)) / max(sum(without_code)/len(without_code), 0.1), 2
            )

        # Is open access
        with_oa = [p["citations"] for p in papers if p.get("is_open_access")]
        without_oa = [p["citations"] for p in papers if not p.get("is_open_access")]
        if with_oa and without_oa:
            factors["is_open_access"] = round(
                (sum(with_oa)/len(with_oa)) / max(sum(without_oa)/len(without_oa), 0.1), 2
            )

        # Is top venue
        with_top = [p["citations"] for p in papers if p.get("is_top_venue")]
        without_top = [p["citations"] for p in papers if not p.get("is_top_venue")]
        if with_top and without_top:
            factors["is_top_venue"] = round(
                (sum(with_top)/len(with_top)) / max(sum(without_top)/len(without_top), 0.1), 2
            )

        # High author h-index
        high_h = [p["citations"] for p in papers if (p.get("max_author_h_index") or 0) > 30]
        low_h = [p["citations"] for p in papers if (p.get("max_author_h_index") or 0) <= 30]
        if high_h and low_h:
            factors["senior_authors"] = round(
                (sum(high_h)/len(high_h)) / max(sum(low_h)/len(low_h), 0.1), 2
            )

        # Sort by impact
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)

        return {
            "factor_multipliers": dict(sorted_factors),
            "top_factors": [f[0] for f in sorted_factors[:5]],
            "interpretation": self._interpret_formula(sorted_factors),
        }

    def _simple_correlation(self, pairs: List[Tuple[float, float]]) -> str:
        """Compute simple correlation direction."""
        if len(pairs) < 10:
            return "insufficient_data"

        n = len(pairs)
        x_mean = sum(p[0] for p in pairs) / n
        y_mean = sum(p[1] for p in pairs) / n

        numerator = sum((p[0] - x_mean) * (p[1] - y_mean) for p in pairs)
        denom_x = sum((p[0] - x_mean) ** 2 for p in pairs) ** 0.5
        denom_y = sum((p[1] - y_mean) ** 2 for p in pairs) ** 0.5

        if denom_x * denom_y == 0:
            return "no_correlation"

        r = numerator / (denom_x * denom_y)

        if r > 0.3:
            return "positive"
        elif r < -0.3:
            return "negative"
        else:
            return "weak"

    def _derive_author_insight(self, h_index_stats: Dict, team_stats: Dict) -> str:
        """Derive insight about author factors."""
        insights = []

        if h_index_stats.get("51+", {}).get("avg_citations", 0) > 2 * h_index_stats.get("0-5", {}).get("avg_citations", 1):
            insights.append("Papers with senior authors (h-index 50+) get significantly more citations")

        if team_stats.get("medium (4-6)", {}).get("avg_citations", 0) > team_stats.get("solo", {}).get("avg_citations", 1):
            insights.append("Medium-sized teams (4-6) outperform solo authors")

        return "; ".join(insights) if insights else "No strong author patterns detected"

    def _interpret_formula(self, factors: List[Tuple[str, float]]) -> str:
        """Interpret the success formula."""
        if not factors:
            return "Insufficient data for interpretation"

        top_factor = factors[0]
        interpretations = {
            "has_code": "Having code available is the strongest citation predictor",
            "is_top_venue": "Publishing at top venues correlates most with citations",
            "is_open_access": "Open access papers get significantly more citations",
            "senior_authors": "Having senior co-authors (high h-index) drives citations",
        }

        return interpretations.get(
            top_factor[0],
            f"{top_factor[0]} is the strongest factor ({top_factor[1]}x citations)"
        )

    def score_paper(self, paper: Dict, success_factors: Dict) -> Dict:
        """Score a paper based on success factors."""
        multipliers = success_factors.get("success_formula", {}).get("factor_multipliers", {})

        base_score = 1.0

        if paper.get("has_code") and "has_code" in multipliers:
            base_score *= multipliers["has_code"]

        if paper.get("is_open_access") and "is_open_access" in multipliers:
            base_score *= multipliers["is_open_access"]

        if paper.get("is_top_venue") and "is_top_venue" in multipliers:
            base_score *= multipliers["is_top_venue"]

        if (paper.get("max_author_h_index") or 0) > 30 and "senior_authors" in multipliers:
            base_score *= multipliers["senior_authors"]

        return {
            "predicted_impact_multiplier": round(base_score, 2),
            "factors_present": [
                f for f in ["has_code", "is_open_access", "is_top_venue"]
                if paper.get(f)
            ],
        }
