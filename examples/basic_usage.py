#!/usr/bin/env python3
"""
Basic Usage Example
===================

This example demonstrates how to use arXiv Researcher to collect
and analyze papers from arXiv.
"""

from datetime import datetime, timedelta

# Import the main classes
from arxiv_researcher import ArxivCollector, ResearchAnalyzer


def main():
    """Run a basic collection and analysis."""

    # 1. Collect papers from specific categories
    print("=" * 60)
    print("arXiv Researcher - Basic Usage Example")
    print("=" * 60)

    collector = ArxivCollector(
        output_dir="example_data",
        skip_s2=False,  # Include Semantic Scholar data
    )

    # Collect recent AI and ML papers
    papers = collector.collect(
        categories=["cs.AI", "cs.LG"],
        max_results=100,  # Small sample for example
        start_date=datetime.now() - timedelta(days=90),  # Last 3 months
    )

    print(f"\nCollected {len(papers)} papers")

    # 2. Analyze the papers
    print("\n" + "=" * 60)
    print("Running Analysis")
    print("=" * 60)

    analyzer = ResearchAnalyzer(papers, output_dir="example_data")
    results = analyzer.analyze()

    # 3. Show some results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Top keywords
    if "text" in results:
        print("\nTop Keywords:")
        for word, count in results["text"]["keywords"][:10]:
            print(f"  - {word}: {count}")

    # Emerging topics
    if "trends" in results:
        print("\nEmerging Topics:")
        for topic, growth in results["trends"]["emerging_topics"][:5]:
            print(f"  - {topic}: +{growth:.0%}" if growth != float('inf') else f"  - {topic}: new")

    # Research gaps
    if "gaps" in results:
        print("\nResearch Opportunities:")
        for gap in results["gaps"]["underexplored_intersections"][:3]:
            cats = " + ".join(gap["categories"])
            print(f"  - {cats} (score: {gap['opportunity_score']:.2f})")

    # Hot papers
    if "impact" in results:
        print("\nHot Papers:")
        for paper in results["impact"]["hot_papers"][:3]:
            print(f"  - {paper['title']}")
            print(f"    {paper['velocity']} citations/month")

    print("\n" + "=" * 60)
    print(f"Full results saved to: example_data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
