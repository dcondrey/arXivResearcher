"""
Research Analyzer
=================

Comprehensive analysis of arXiv research papers.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import re


class ResearchAnalyzer:
    """Analyzer for research papers with multiple analysis modules."""

    # Common method patterns
    METHOD_PATTERNS = [
        r'\b(transformer|attention|bert|gpt|llm|language model)\b',
        r'\b(cnn|convolutional|resnet|vgg|inception)\b',
        r'\b(rnn|lstm|gru|recurrent)\b',
        r'\b(gan|generative adversarial|diffusion)\b',
        r'\b(reinforcement learning|rl|policy gradient|q-learning)\b',
        r'\b(graph neural|gnn|gcn|message passing)\b',
        r'\b(self-supervised|contrastive learning|siamese)\b',
        r'\b(fine-tuning|transfer learning|pre-training)\b',
        r'\b(neural network|deep learning|machine learning)\b',
        r'\b(monte carlo|mcmc|bayesian)\b',
    ]

    # Common dataset patterns
    DATASET_PATTERNS = [
        r'\b(imagenet|coco|cifar|mnist)\b',
        r'\b(squad|glue|superglue|mnli)\b',
        r'\b(wikitext|c4|pile|common crawl)\b',
        r'\b(openwebtext|bookcorpus|wikipedia)\b',
    ]

    def __init__(
        self,
        papers: List[Dict[str, Any]],
        output_dir: str = "analysis",
    ) -> None:
        """Initialize the analyzer.

        Args:
            papers: List of paper dictionaries.
            output_dir: Directory to save analysis results.
        """
        self.papers = papers
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze(
        self,
        advanced: bool = True,
        fulltext: bool = False,
    ) -> Dict[str, Any]:
        """Run all analysis modules.

        Args:
            advanced: Run advanced analysis (embeddings, etc.).
            fulltext: Analyze PDF full text.

        Returns:
            Dictionary of analysis results.
        """
        results = {}

        print(f"\nAnalyzing {len(self.papers):,} papers...")

        # Text analysis
        print("  Running text analysis...")
        results["text"] = self.analyze_text()

        # Trend analysis
        print("  Running trend analysis...")
        results["trends"] = self.analyze_trends()

        # Gap analysis
        print("  Running gap analysis...")
        results["gaps"] = self.find_gaps()

        # Network analysis
        print("  Running network analysis...")
        results["network"] = self.analyze_network()

        # Impact analysis
        print("  Running impact analysis...")
        results["impact"] = self.predict_impact()

        # Save results
        self._save_results(results)

        return results

    def analyze_text(self) -> Dict[str, Any]:
        """Analyze text content of papers."""
        # Extract keywords using simple TF
        word_counts: Counter = Counter()
        method_counts: Counter = Counter()
        dataset_counts: Counter = Counter()

        for paper in self.papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

            # Count words (simple tokenization)
            words = re.findall(r'\b[a-z]{4,}\b', text)
            word_counts.update(words)

            # Detect methods
            for pattern in self.METHOD_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                method_counts.update([m.lower() if isinstance(m, str) else m[0].lower() for m in matches])

            # Detect datasets
            for pattern in self.DATASET_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dataset_counts.update([m.lower() for m in matches])

        # Filter common words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'which', 'their',
            'more', 'will', 'also', 'such', 'than', 'into', 'these', 'other',
            'some', 'only', 'based', 'using', 'paper', 'method', 'approach',
            'results', 'propose', 'proposed', 'show', 'shown', 'model', 'models',
        }

        keywords = [
            (word, count)
            for word, count in word_counts.most_common(200)
            if word not in stop_words
        ][:50]

        return {
            "keywords": keywords,
            "methods": method_counts.most_common(20),
            "datasets": dataset_counts.most_common(20),
            "total_papers": len(self.papers),
        }

    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze research trends over time."""
        # Group papers by time period
        papers_by_period: Dict[str, List[Dict]] = defaultdict(list)

        for paper in self.papers:
            date_str = paper.get("published_date", "")
            if date_str:
                # Group by quarter
                try:
                    date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    period = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                    papers_by_period[period].append(paper)
                except ValueError:
                    pass

        # Calculate trends
        periods = sorted(papers_by_period.keys())
        trend_data = []

        for period in periods:
            papers = papers_by_period[period]
            categories = Counter()
            for p in papers:
                categories.update(p.get("categories", []))

            trend_data.append({
                "period": period,
                "paper_count": len(papers),
                "top_categories": categories.most_common(5),
            })

        # Identify emerging topics (simple heuristic)
        if len(periods) >= 4:
            recent = periods[-2:]
            earlier = periods[-4:-2]

            recent_keywords = Counter()
            earlier_keywords = Counter()

            for period in recent:
                for paper in papers_by_period[period]:
                    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
                    words = re.findall(r'\b[a-z]{4,}\b', text)
                    recent_keywords.update(words)

            for period in earlier:
                for paper in papers_by_period[period]:
                    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
                    words = re.findall(r'\b[a-z]{4,}\b', text)
                    earlier_keywords.update(words)

            # Find emerging terms
            emerging = []
            for word, count in recent_keywords.most_common(100):
                earlier_count = earlier_keywords.get(word, 0)
                if earlier_count > 0:
                    growth = (count - earlier_count) / earlier_count
                    if growth > 0.5:  # 50% growth
                        emerging.append((word, growth))
                elif count >= 10:
                    emerging.append((word, float('inf')))

            emerging.sort(key=lambda x: x[1], reverse=True)
        else:
            emerging = []

        return {
            "periods": trend_data,
            "emerging_topics": emerging[:20],
            "total_periods": len(periods),
        }

    def find_gaps(self) -> Dict[str, Any]:
        """Identify research gaps and opportunities."""
        # Count category intersections
        intersection_counts: Counter = Counter()
        category_counts: Counter = Counter()

        for paper in self.papers:
            cats = paper.get("categories", [])
            category_counts.update(cats)

            # Count intersections
            cats_sorted = sorted(cats)
            for i, cat1 in enumerate(cats_sorted):
                for cat2 in cats_sorted[i + 1:]:
                    intersection_counts[(cat1, cat2)] += 1

        # Find underexplored intersections
        gaps = []
        for (cat1, cat2), count in intersection_counts.items():
            expected = (category_counts[cat1] * category_counts[cat2]) / len(self.papers)
            if expected > 10 and count < expected * 0.5:
                gaps.append({
                    "categories": [cat1, cat2],
                    "actual": count,
                    "expected": expected,
                    "opportunity_score": (expected - count) / expected,
                })

        gaps.sort(key=lambda x: x["opportunity_score"], reverse=True)

        # Survey opportunities (categories with many papers but few surveys)
        survey_opps = []
        for cat, count in category_counts.most_common(20):
            survey_count = sum(
                1 for p in self.papers
                if cat in p.get("categories", [])
                and any(
                    word in p.get("title", "").lower()
                    for word in ["survey", "review", "overview", "tutorial"]
                )
            )
            if count > 50 and survey_count < count * 0.05:
                survey_opps.append({
                    "category": cat,
                    "paper_count": count,
                    "survey_count": survey_count,
                })

        return {
            "underexplored_intersections": gaps[:15],
            "survey_opportunities": survey_opps[:10],
        }

    def analyze_network(self) -> Dict[str, Any]:
        """Analyze author collaboration network."""
        # Build co-authorship counts
        coauthor_counts: Counter = Counter()
        author_paper_counts: Counter = Counter()

        for paper in self.papers:
            authors = paper.get("authors", [])
            author_paper_counts.update(authors)

            # Count co-authorships
            for i, a1 in enumerate(authors):
                for a2 in authors[i + 1:]:
                    pair = tuple(sorted([a1, a2]))
                    coauthor_counts[pair] += 1

        # Top authors
        top_authors = []
        for author, count in author_paper_counts.most_common(50):
            citations = sum(
                p.get("citation_count", 0)
                for p in self.papers
                if author in p.get("authors", [])
            )
            top_authors.append({
                "name": author,
                "papers": count,
                "citations": citations,
            })

        # Top collaborations
        top_collabs = [
            {"authors": list(pair), "papers": count}
            for pair, count in coauthor_counts.most_common(20)
        ]

        return {
            "top_authors": top_authors,
            "top_collaborations": top_collabs,
            "unique_authors": len(author_paper_counts),
        }

    def predict_impact(self) -> Dict[str, Any]:
        """Analyze factors correlated with paper impact."""
        # Simple feature analysis
        features = {
            "has_code": [],
            "author_count": [],
            "abstract_length": [],
            "has_figures": [],
        }
        citations = []

        for paper in self.papers:
            citation_count = paper.get("citation_count", 0)
            citations.append(citation_count)

            # Extract features
            features["author_count"].append(len(paper.get("authors", [])))
            features["abstract_length"].append(len(paper.get("abstract", "")))

            # Check for code indicators
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            features["has_code"].append(
                1 if any(w in text for w in ["github", "code available", "implementation"]) else 0
            )

        # Calculate correlations (simplified)
        factors = []

        for feature_name, values in features.items():
            if len(values) > 0:
                # Simple correlation approximation
                high_citation_avg = sum(
                    v for v, c in zip(values, citations) if c > 10
                ) / max(1, sum(1 for c in citations if c > 10))

                low_citation_avg = sum(
                    v for v, c in zip(values, citations) if c <= 10
                ) / max(1, sum(1 for c in citations if c <= 10))

                if low_citation_avg > 0:
                    impact = high_citation_avg / low_citation_avg
                    factors.append({
                        "factor": feature_name,
                        "impact_multiplier": round(impact, 2),
                    })

        factors.sort(key=lambda x: x["impact_multiplier"], reverse=True)

        # Hot papers (high citation velocity)
        hot_papers = []
        for paper in self.papers:
            citations = paper.get("citation_count", 0)
            date_str = paper.get("published_date", "")
            if date_str and citations > 0:
                try:
                    pub_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    months = max(1, (datetime.now() - pub_date).days / 30)
                    velocity = citations / months
                    if velocity > 1:
                        hot_papers.append({
                            "title": paper["title"][:80],
                            "arxiv_id": paper["arxiv_id"],
                            "citations": citations,
                            "velocity": round(velocity, 2),
                        })
                except ValueError:
                    pass

        hot_papers.sort(key=lambda x: x["velocity"], reverse=True)

        return {
            "success_factors": factors,
            "hot_papers": hot_papers[:20],
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to files."""
        analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        for name, data in results.items():
            filepath = os.path.join(analysis_dir, f"{name}_analysis.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

        print(f"\nAnalysis saved to {analysis_dir}/")
