"""
Trend Analysis Module
=====================
Detect and forecast research trends:
- Time series analysis of topics
- Trend detection (emerging, stable, declining)
- Momentum scoring
- Saturation analysis
- Forecasting
"""

from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import math
from datetime import datetime


class TrendAnalyzer:
    """Analyze research trends over time."""

    def __init__(self):
        self.time_series = {}

    def analyze_trends(self, papers: List[Dict]) -> Dict:
        """Run comprehensive trend analysis."""
        print(f"    Analyzing trends for {len(papers)} papers...")

        # Build time series
        self._build_time_series(papers)

        return {
            "category_trends": self.analyze_category_trends(papers),
            "method_trends": self.analyze_method_trends(papers),
            "topic_trends": self.analyze_topic_trends(papers),
            "velocity_analysis": self.analyze_citation_velocity(papers),
            "saturation_analysis": self.analyze_saturation(papers),
            "hot_papers": self.find_hot_papers(papers),
            "sleeping_beauties": self.find_sleeping_beauties(papers),
        }

    def _build_time_series(self, papers: List[Dict]):
        """Build time series data structures."""
        self.papers_by_month = defaultdict(list)
        self.papers_by_category_month = defaultdict(lambda: defaultdict(list))

        for paper in papers:
            month = paper.get("month", "")
            if month:
                self.papers_by_month[month].append(paper)

                category = paper.get("primary_category", "")
                self.papers_by_category_month[category][month].append(paper)

    def analyze_category_trends(self, papers: List[Dict]) -> Dict:
        """Analyze trends by arXiv category."""
        trends = {}

        for category, months_data in self.papers_by_category_month.items():
            sorted_months = sorted(months_data.keys())
            if len(sorted_months) < 2:
                continue

            counts = [len(months_data[m]) for m in sorted_months]
            citations = [
                sum(p.get("citations") or 0 for p in months_data[m])
                for m in sorted_months
            ]

            # Calculate trend metrics
            trend_info = self._calculate_trend(counts)
            citation_trend = self._calculate_trend(citations)

            # Momentum (recent vs early)
            if len(counts) >= 4:
                recent = sum(counts[-len(counts)//4:])
                early = sum(counts[:len(counts)//4])
                momentum = (recent - early) / max(early, 1)
            else:
                momentum = 0

            trends[category] = {
                "total_papers": sum(counts),
                "months_active": len(sorted_months),
                "avg_papers_per_month": round(sum(counts) / len(counts), 1),
                "trend_direction": trend_info["direction"],
                "trend_slope": trend_info["slope"],
                "trend_r_squared": trend_info["r_squared"],
                "momentum": round(momentum, 2),
                "citation_trend_direction": citation_trend["direction"],
                "total_citations": sum(citations),
                "time_series": {m: len(months_data[m]) for m in sorted_months},
            }

        # Classify into emerging, stable, declining
        for cat, data in trends.items():
            if data["momentum"] > 0.5 and data["trend_slope"] > 0:
                data["status"] = "emerging"
            elif data["momentum"] < -0.3 and data["trend_slope"] < 0:
                data["status"] = "declining"
            else:
                data["status"] = "stable"

        return trends

    def analyze_method_trends(self, papers: List[Dict]) -> Dict:
        """Analyze trends in methods/techniques usage."""
        method_by_month = defaultdict(lambda: defaultdict(int))

        for paper in papers:
            month = paper.get("month", "")
            methods = paper.get("methods_detected", [])
            if month:
                for method in methods:
                    method_by_month[method][month] += 1

        trends = {}
        for method, months_data in method_by_month.items():
            sorted_months = sorted(months_data.keys())
            if len(sorted_months) < 2:
                continue

            counts = [months_data[m] for m in sorted_months]
            trend_info = self._calculate_trend(counts)

            # Recent momentum
            if len(counts) >= 4:
                recent = sum(counts[-len(counts)//4:])
                early = sum(counts[:len(counts)//4])
                momentum = (recent - early) / max(early, 1)
            else:
                momentum = 0

            trends[method] = {
                "total_mentions": sum(counts),
                "trend_direction": trend_info["direction"],
                "trend_slope": trend_info["slope"],
                "momentum": round(momentum, 2),
                "months_present": len(sorted_months),
            }

        # Sort by momentum
        sorted_trends = sorted(
            trends.items(),
            key=lambda x: x[1]["momentum"],
            reverse=True
        )

        return {
            "all_methods": trends,
            "emerging_methods": [
                {"method": m, **d} for m, d in sorted_trends[:20]
                if d["momentum"] > 0.3
            ],
            "declining_methods": [
                {"method": m, **d} for m, d in sorted_trends[-20:]
                if d["momentum"] < -0.2
            ],
        }

    def analyze_topic_trends(self, papers: List[Dict]) -> Dict:
        """Analyze trends in extracted topics/keywords."""
        keyword_by_month = defaultdict(lambda: defaultdict(int))

        for paper in papers:
            month = paper.get("month", "")
            keywords = paper.get("keywords", [])
            if month:
                for kw in keywords[:5]:  # Top 5 keywords per paper
                    keyword_by_month[kw][month] += 1

        trends = {}
        for keyword, months_data in keyword_by_month.items():
            sorted_months = sorted(months_data.keys())
            if len(sorted_months) < 2 or sum(months_data.values()) < 5:
                continue

            counts = [months_data[m] for m in sorted_months]
            trend_info = self._calculate_trend(counts)

            trends[keyword] = {
                "total_mentions": sum(counts),
                "trend_slope": trend_info["slope"],
                "first_seen": sorted_months[0],
                "last_seen": sorted_months[-1],
            }

        # Find emerging keywords (first appeared recently, growing)
        sorted_months = sorted(self.papers_by_month.keys())
        recent_cutoff = sorted_months[-len(sorted_months)//4] if len(sorted_months) >= 4 else sorted_months[0]

        emerging_topics = [
            {"keyword": kw, **data}
            for kw, data in trends.items()
            if data["first_seen"] >= recent_cutoff and data["trend_slope"] > 0
        ]
        emerging_topics.sort(key=lambda x: x["total_mentions"], reverse=True)

        return {
            "all_topics": trends,
            "emerging_topics": emerging_topics[:30],
        }

    def analyze_citation_velocity(self, papers: List[Dict]) -> Dict:
        """Analyze citation velocity patterns."""
        velocity_by_category = defaultdict(list)
        velocity_by_month = defaultdict(list)

        for paper in papers:
            velocity = paper.get("citations_per_month") or 0
            category = paper.get("primary_category", "")
            month = paper.get("month", "")

            if velocity > 0:
                velocity_by_category[category].append(velocity)
                if month:
                    velocity_by_month[month].append(velocity)

        # Category velocity stats
        category_velocity = {}
        for cat, velocities in velocity_by_category.items():
            if len(velocities) >= 5:
                category_velocity[cat] = {
                    "avg_velocity": round(sum(velocities) / len(velocities), 3),
                    "max_velocity": round(max(velocities), 3),
                    "papers_with_velocity": len(velocities),
                    "high_velocity_count": sum(1 for v in velocities if v > 1),
                }

        # Velocity over time
        sorted_months = sorted(velocity_by_month.keys())
        velocity_trend = {
            m: round(sum(velocity_by_month[m]) / len(velocity_by_month[m]), 3)
            for m in sorted_months
            if velocity_by_month[m]
        }

        return {
            "by_category": category_velocity,
            "over_time": velocity_trend,
            "hottest_categories": sorted(
                category_velocity.items(),
                key=lambda x: x[1]["avg_velocity"],
                reverse=True
            )[:10],
        }

    def analyze_saturation(self, papers: List[Dict]) -> Dict:
        """Analyze topic/category saturation (too crowded vs. opportunity)."""
        saturation = {}

        for category, months_data in self.papers_by_category_month.items():
            sorted_months = sorted(months_data.keys())
            if len(sorted_months) < 3:
                continue

            # Count papers
            counts = [len(months_data[m]) for m in sorted_months]
            total = sum(counts)

            # Citation metrics
            all_citations = []
            for m in sorted_months:
                for p in months_data[m]:
                    cites = p.get("citations") or 0
                    all_citations.append(cites)

            avg_citations = sum(all_citations) / len(all_citations) if all_citations else 0
            citation_gini = self._calculate_gini(all_citations)

            # Growth rate
            trend = self._calculate_trend(counts)

            # Saturation indicators:
            # - High paper count + low avg citations = saturated
            # - High gini (few papers get most citations) = winner-take-all
            # - Declining growth = mature
            saturation_score = (
                (total / 100) * 0.3 +  # More papers = more saturated
                (1 - avg_citations / 100) * 0.3 +  # Lower citations = saturated
                citation_gini * 0.2 +  # High inequality = saturated
                (-trend["slope"] if trend["slope"] < 0 else 0) * 0.2  # Declining = saturated
            )

            saturation[category] = {
                "total_papers": total,
                "avg_citations": round(avg_citations, 1),
                "citation_gini": round(citation_gini, 3),
                "growth_slope": trend["slope"],
                "saturation_score": round(saturation_score, 3),
                "assessment": (
                    "saturated" if saturation_score > 0.6 else
                    "competitive" if saturation_score > 0.4 else
                    "opportunity"
                ),
            }

        return saturation

    def find_hot_papers(self, papers: List[Dict], top_n: int = 50) -> List[Dict]:
        """Find papers with unusually high citation velocity."""
        hot = []

        for paper in papers:
            velocity = paper.get("citations_per_month") or 0
            citations = paper.get("citations") or 0
            age_days = paper.get("paper_age_days") or 365

            # Hot = high velocity, especially if young
            if velocity > 0.5 or (velocity > 0.2 and age_days < 180):
                hot.append({
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "citations": citations,
                    "citations_per_month": velocity,
                    "age_days": age_days,
                    "category": paper.get("primary_category"),
                    "has_code": paper.get("has_code", False),
                    "venue": paper.get("publication_venue", ""),
                })

        hot.sort(key=lambda x: x["citations_per_month"], reverse=True)
        return hot[:top_n]

    def find_sleeping_beauties(self, papers: List[Dict], top_n: int = 30) -> List[Dict]:
        """Find papers that got recognition late (sleeping beauties)."""
        beauties = []

        for paper in papers:
            citations = paper.get("citations") or 0
            age_days = paper.get("paper_age_days") or 0
            velocity = paper.get("citations_per_month") or 0

            # Sleeping beauty: old paper with recent high velocity
            # (would need citation history to do this properly)
            if age_days > 365 and citations > 20 and velocity > 0.5:
                beauties.append({
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "citations": citations,
                    "citations_per_month": velocity,
                    "age_days": age_days,
                    "category": paper.get("primary_category"),
                })

        beauties.sort(key=lambda x: x["citations_per_month"], reverse=True)
        return beauties[:top_n]

    def forecast_categories(self, papers: List[Dict], periods_ahead: int = 3) -> Dict:
        """Simple forecasting for category volumes."""
        forecasts = {}

        for category, months_data in self.papers_by_category_month.items():
            sorted_months = sorted(months_data.keys())
            if len(sorted_months) < 6:
                continue

            counts = [len(months_data[m]) for m in sorted_months]

            # Simple linear forecast
            trend = self._calculate_trend(counts)
            last_value = counts[-1]

            predictions = []
            for i in range(1, periods_ahead + 1):
                pred = last_value + trend["slope"] * i
                predictions.append(max(0, round(pred)))

            forecasts[category] = {
                "historical": {m: len(months_data[m]) for m in sorted_months},
                "trend_slope": trend["slope"],
                "forecast_next_periods": predictions,
                "confidence": "low" if trend["r_squared"] < 0.3 else "medium" if trend["r_squared"] < 0.7 else "high",
            }

        return forecasts

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
        if slope > 0.1 and r_squared > 0.2:
            direction = "increasing"
        elif slope < -0.1 and r_squared > 0.2:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "slope": round(slope, 4),
            "r_squared": round(max(0, r_squared), 4),
        }

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient (inequality measure)."""
        if not values or len(values) < 2:
            return 0

        values = sorted(values)
        n = len(values)
        total = sum(values)

        if total == 0:
            return 0

        cumsum = 0
        gini_sum = 0
        for i, v in enumerate(values):
            cumsum += v
            gini_sum += cumsum

        gini = (2 * gini_sum) / (n * total) - (n + 1) / n
        return max(0, min(1, gini))
