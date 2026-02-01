"""
Research Gap Analysis Module
============================
Identify research opportunities:
- Underexplored category intersections
- Method-domain gaps
- High-demand low-supply topics
- Reproducibility gaps
- Survey opportunities
"""

from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import math


class GapAnalyzer:
    """Identify research gaps and opportunities."""

    def __init__(self):
        pass

    def analyze_gaps(self, papers: List[Dict]) -> Dict:
        """Run comprehensive gap analysis."""
        print(f"    Analyzing research gaps for {len(papers)} papers...")

        return {
            "category_intersections": self.find_underexplored_intersections(papers),
            "method_domain_gaps": self.find_method_domain_gaps(papers),
            "reproducibility_gaps": self.find_reproducibility_gaps(papers),
            "survey_opportunities": self.find_survey_opportunities(papers),
            "dataset_gaps": self.find_dataset_gaps(papers),
            "application_gaps": self.find_application_gaps(papers),
            "opportunity_scores": self.compute_opportunity_scores(papers),
        }

    def find_underexplored_intersections(self, papers: List[Dict]) -> Dict:
        """Find category pairs that are underexplored relative to their components."""
        # Count individual categories and pairs
        category_counts = Counter()
        pair_counts = Counter()

        for paper in papers:
            categories = paper.get("category_list", [])
            for cat in categories:
                category_counts[cat] += 1

            # Count pairs
            for i, cat1 in enumerate(categories):
                for cat2 in categories[i+1:]:
                    pair = tuple(sorted([cat1, cat2]))
                    pair_counts[pair] += 1

        total_papers = len(papers)
        underexplored = []

        # Find pairs with fewer papers than expected
        for (cat1, cat2), actual in pair_counts.items():
            # Expected under independence
            expected = (category_counts[cat1] * category_counts[cat2]) / total_papers

            if expected > 5:  # Only consider meaningful intersections
                ratio = actual / expected

                if ratio < 0.5:  # Less than half expected
                    # Check if both categories are active
                    if category_counts[cat1] > 20 and category_counts[cat2] > 20:
                        underexplored.append({
                            "categories": [cat1, cat2],
                            "actual_papers": actual,
                            "expected_papers": round(expected, 1),
                            "ratio": round(ratio, 3),
                            "cat1_papers": category_counts[cat1],
                            "cat2_papers": category_counts[cat2],
                            "opportunity_score": round((1 - ratio) * min(expected, 50), 1),
                        })

        underexplored.sort(key=lambda x: x["opportunity_score"], reverse=True)

        # Also find unexplored top-level category combinations
        top_level_pairs = Counter()
        top_level_counts = Counter()

        for paper in papers:
            categories = paper.get("category_list", [])
            top_levels = list(set(c.split(".")[0] for c in categories))

            for tl in top_levels:
                top_level_counts[tl] += 1

            for i, tl1 in enumerate(top_levels):
                for tl2 in top_levels[i+1:]:
                    pair = tuple(sorted([tl1, tl2]))
                    top_level_pairs[pair] += 1

        cross_discipline = []
        for (tl1, tl2), count in top_level_pairs.items():
            expected = (top_level_counts[tl1] * top_level_counts[tl2]) / total_papers
            if expected > 10:
                ratio = count / expected
                if ratio < 0.3:
                    cross_discipline.append({
                        "disciplines": [tl1, tl2],
                        "actual": count,
                        "expected": round(expected, 1),
                        "ratio": round(ratio, 3),
                    })

        cross_discipline.sort(key=lambda x: x["expected"] - x["actual"], reverse=True)

        return {
            "underexplored_pairs": underexplored[:30],
            "cross_discipline_gaps": cross_discipline[:20],
        }

    def find_method_domain_gaps(self, papers: List[Dict]) -> Dict:
        """Find methods not yet applied to certain domains."""
        method_by_category = defaultdict(Counter)
        category_totals = Counter()
        method_totals = Counter()

        for paper in papers:
            category = paper.get("primary_category", "")
            methods = paper.get("methods_detected", [])

            category_totals[category] += 1
            for method in methods:
                method_by_category[method][category] += 1
                method_totals[method] += 1

        gaps = []

        # Find methods popular overall but rare in specific categories
        for method, categories in method_by_category.items():
            if method_totals[method] < 20:  # Method must be somewhat popular
                continue

            method_rate = method_totals[method] / len(papers)

            for category, total in category_totals.items():
                if total < 30:  # Category must have enough papers
                    continue

                expected = total * method_rate
                actual = categories.get(category, 0)

                if expected > 3 and actual < expected * 0.3:
                    gaps.append({
                        "method": method,
                        "category": category,
                        "actual_papers": actual,
                        "expected_papers": round(expected, 1),
                        "method_overall_papers": method_totals[method],
                        "category_total_papers": total,
                        "gap_score": round(expected - actual, 1),
                    })

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)

        return {
            "method_category_gaps": gaps[:50],
            "summary": {
                "total_methods_analyzed": len(method_by_category),
                "total_categories_analyzed": len(category_totals),
                "gaps_found": len(gaps),
            }
        }

    def find_reproducibility_gaps(self, papers: List[Dict]) -> Dict:
        """Find highly-cited papers lacking code or reproduction attempts."""
        # Papers with high citations but no code
        no_code_high_cited = []

        for paper in papers:
            citations = paper.get("citations") or 0
            has_code = paper.get("has_code", False)

            if citations >= 20 and not has_code:
                no_code_high_cited.append({
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "citations": citations,
                    "category": paper.get("primary_category"),
                    "methods": paper.get("methods_detected", [])[:3],
                    "has_code": False,
                    "reproduction_opportunity_score": min(citations / 10, 10),
                })

        no_code_high_cited.sort(key=lambda x: x["citations"], reverse=True)

        # Categories with low code availability
        code_by_category = defaultdict(lambda: {"with_code": 0, "without_code": 0})

        for paper in papers:
            category = paper.get("primary_category", "")
            if paper.get("has_code"):
                code_by_category[category]["with_code"] += 1
            else:
                code_by_category[category]["without_code"] += 1

        category_code_rates = []
        for cat, counts in code_by_category.items():
            total = counts["with_code"] + counts["without_code"]
            if total >= 20:
                rate = counts["with_code"] / total
                category_code_rates.append({
                    "category": cat,
                    "papers_with_code": counts["with_code"],
                    "papers_without_code": counts["without_code"],
                    "code_availability_rate": round(rate * 100, 1),
                })

        category_code_rates.sort(key=lambda x: x["code_availability_rate"])

        return {
            "high_cited_no_code": no_code_high_cited[:50],
            "code_availability_by_category": category_code_rates,
            "lowest_code_categories": category_code_rates[:10],
        }

    def find_survey_opportunities(self, papers: List[Dict]) -> Dict:
        """Find categories/topics that need surveys."""
        # Count papers and surveys by category
        category_stats = defaultdict(lambda: {"papers": 0, "surveys": 0, "recent_papers": 0})

        for paper in papers:
            category = paper.get("primary_category", "")
            is_survey = paper.get("is_survey", False)
            age_days = paper.get("paper_age_days") or 365

            category_stats[category]["papers"] += 1
            if is_survey:
                category_stats[category]["surveys"] += 1
            if age_days < 180:  # Recent papers
                category_stats[category]["recent_papers"] += 1

        opportunities = []

        for cat, stats in category_stats.items():
            if stats["papers"] >= 50:  # Enough papers to warrant a survey
                survey_ratio = stats["surveys"] / stats["papers"]
                recency_ratio = stats["recent_papers"] / stats["papers"]

                # Opportunity if: many papers, few surveys, lots of recent activity
                opportunity_score = (
                    stats["papers"] * 0.01 +
                    (1 - survey_ratio) * 5 +
                    recency_ratio * 3
                )

                if stats["surveys"] < 3 or survey_ratio < 0.02:
                    opportunities.append({
                        "category": cat,
                        "total_papers": stats["papers"],
                        "existing_surveys": stats["surveys"],
                        "survey_ratio": round(survey_ratio * 100, 2),
                        "recent_papers": stats["recent_papers"],
                        "opportunity_score": round(opportunity_score, 2),
                    })

        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)

        # Also check method-level survey opportunities
        method_stats = defaultdict(lambda: {"papers": 0, "surveys": 0})

        for paper in papers:
            methods = paper.get("methods_detected", [])
            is_survey = paper.get("is_survey", False)

            for method in methods:
                method_stats[method]["papers"] += 1
                if is_survey:
                    method_stats[method]["surveys"] += 1

        method_opportunities = []
        for method, stats in method_stats.items():
            if stats["papers"] >= 30 and stats["surveys"] < 2:
                method_opportunities.append({
                    "method": method,
                    "papers_using_method": stats["papers"],
                    "existing_surveys": stats["surveys"],
                })

        method_opportunities.sort(key=lambda x: x["papers_using_method"], reverse=True)

        return {
            "category_survey_opportunities": opportunities[:20],
            "method_survey_opportunities": method_opportunities[:20],
        }

    def find_dataset_gaps(self, papers: List[Dict]) -> Dict:
        """Find datasets not used in certain domains, and domains lacking benchmarks."""
        dataset_by_category = defaultdict(Counter)
        category_totals = Counter()

        for paper in papers:
            category = paper.get("primary_category", "")
            datasets = paper.get("datasets_mentioned", [])

            category_totals[category] += 1
            for ds in datasets:
                dataset_by_category[ds][category] += 1

        # Categories with few dataset mentions (need benchmarks)
        category_dataset_usage = []
        for cat, total in category_totals.items():
            if total >= 30:
                # Count papers that mention any dataset
                papers_with_datasets = sum(
                    1 for p in papers
                    if p.get("primary_category") == cat and p.get("datasets_mentioned")
                )
                rate = papers_with_datasets / total

                if rate < 0.3:
                    category_dataset_usage.append({
                        "category": cat,
                        "papers_with_datasets": papers_with_datasets,
                        "total_papers": total,
                        "dataset_usage_rate": round(rate * 100, 1),
                    })

        category_dataset_usage.sort(key=lambda x: x["dataset_usage_rate"])

        # Popular datasets not used in certain categories
        transfer_opportunities = []
        for dataset, categories in dataset_by_category.items():
            total_uses = sum(categories.values())
            if total_uses < 20:
                continue

            # Find categories where this dataset isn't used but could be
            for cat, cat_total in category_totals.items():
                if cat_total < 30:
                    continue

                uses_in_cat = categories.get(cat, 0)
                expected = cat_total * (total_uses / len(papers))

                if expected > 3 and uses_in_cat < expected * 0.2:
                    transfer_opportunities.append({
                        "dataset": dataset,
                        "category": cat,
                        "current_uses": uses_in_cat,
                        "expected_uses": round(expected, 1),
                        "dataset_total_uses": total_uses,
                    })

        transfer_opportunities.sort(key=lambda x: x["expected_uses"] - x["current_uses"], reverse=True)

        return {
            "categories_needing_benchmarks": category_dataset_usage[:15],
            "dataset_transfer_opportunities": transfer_opportunities[:30],
        }

    def find_application_gaps(self, papers: List[Dict]) -> Dict:
        """Find methods that could be applied to new domains."""
        # Check which methods have been applied to real-world problems
        method_applications = defaultdict(lambda: {"total": 0, "applied": 0})

        for paper in papers:
            methods = paper.get("methods_detected", [])
            is_application = "application" in (paper.get("contribution_types") or [])

            for method in methods:
                method_applications[method]["total"] += 1
                if is_application:
                    method_applications[method]["applied"] += 1

        application_gaps = []
        for method, stats in method_applications.items():
            if stats["total"] >= 20:
                rate = stats["applied"] / stats["total"]
                if rate < 0.15:  # Less than 15% application papers
                    application_gaps.append({
                        "method": method,
                        "total_papers": stats["total"],
                        "application_papers": stats["applied"],
                        "application_rate": round(rate * 100, 1),
                    })

        application_gaps.sort(key=lambda x: x["total_papers"], reverse=True)

        return {
            "methods_needing_applications": application_gaps[:30],
        }

    def compute_opportunity_scores(self, papers: List[Dict]) -> Dict:
        """Compute overall opportunity scores for different research directions."""
        opportunities = []

        # Category opportunities
        category_metrics = defaultdict(lambda: {
            "papers": 0, "citations": 0, "code_count": 0,
            "survey_count": 0, "recent_count": 0, "velocity_sum": 0
        })

        for paper in papers:
            cat = paper.get("primary_category", "")
            category_metrics[cat]["papers"] += 1
            category_metrics[cat]["citations"] += paper.get("citations") or 0
            if paper.get("has_code"):
                category_metrics[cat]["code_count"] += 1
            if paper.get("is_survey"):
                category_metrics[cat]["survey_count"] += 1
            if (paper.get("paper_age_days") or 365) < 180:
                category_metrics[cat]["recent_count"] += 1
            category_metrics[cat]["velocity_sum"] += paper.get("citations_per_month") or 0

        for cat, m in category_metrics.items():
            if m["papers"] < 20:
                continue

            avg_citations = m["citations"] / m["papers"]
            code_rate = m["code_count"] / m["papers"]
            survey_rate = m["survey_count"] / m["papers"]
            recency_rate = m["recent_count"] / m["papers"]
            avg_velocity = m["velocity_sum"] / m["papers"]

            # Opportunity score components:
            # - High velocity (hot topic) +
            # - High recency (growing) +
            # - Low survey rate (need review) +
            # - Moderate competition (not too crowded) +
            score = (
                min(avg_velocity * 2, 3) +  # Velocity bonus
                recency_rate * 2 +  # Growth bonus
                (1 - survey_rate) * 2 +  # Survey gap bonus
                (1 - min(m["papers"] / 500, 1)) * 1 +  # Not too crowded
                min(avg_citations / 20, 2)  # Citation potential
            )

            opportunities.append({
                "category": cat,
                "opportunity_score": round(score, 2),
                "paper_count": m["papers"],
                "avg_citations": round(avg_citations, 1),
                "avg_velocity": round(avg_velocity, 3),
                "code_availability": round(code_rate * 100, 1),
                "has_recent_survey": m["survey_count"] > 0,
                "growth_indicator": round(recency_rate * 100, 1),
                "recommendation": self._get_recommendation(m, avg_velocity, survey_rate),
            })

        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)

        return {
            "ranked_opportunities": opportunities[:30],
            "top_10_summary": [
                f"{o['category']}: {o['recommendation']}"
                for o in opportunities[:10]
            ],
        }

    def _get_recommendation(self, metrics: Dict, velocity: float, survey_rate: float) -> str:
        """Generate a recommendation based on metrics."""
        if velocity > 1 and survey_rate < 0.02:
            return "Hot topic needs survey"
        elif velocity > 0.5 and metrics["code_count"] / metrics["papers"] < 0.2:
            return "Growing area needs implementations"
        elif metrics["recent_count"] / metrics["papers"] > 0.5:
            return "Rapidly emerging field"
        elif metrics["papers"] < 100 and velocity > 0.3:
            return "Niche with high impact potential"
        else:
            return "Established area"
