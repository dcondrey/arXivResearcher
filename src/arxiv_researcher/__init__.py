"""
arXiv Researcher - Research Intelligence and Visualization Dashboard
====================================================================

A comprehensive tool for mining, analyzing, and visualizing arXiv research data.

Features:
- Multi-source data collection (arXiv, Semantic Scholar, OpenAlex, Papers With Code)
- Text analysis (keywords, methods, datasets extraction)
- Network analysis (co-authorship, citation patterns)
- Trend analysis (emerging topics, declining areas)
- Gap analysis (research opportunities, underexplored areas)
- Impact prediction (success factors)
- Interactive Streamlit dashboard

Usage:
    # Command line
    arxiv-researcher collect --categories cs.AI,cs.LG --max-results 500
    arxiv-researcher analyze --input data/papers.json
    arxiv-dashboard --data data/

    # Python API
    from arxiv_researcher import ArxivCollector, ResearchAnalyzer

    collector = ArxivCollector()
    papers = collector.collect(categories=['cs.AI'], max_results=100)

    analyzer = ResearchAnalyzer(papers)
    trends = analyzer.analyze_trends()
"""

__version__ = "1.0.0"
__author__ = "David Condrey"
__email__ = "david@writerslogic.com"
__license__ = "Apache-2.0"

from arxiv_researcher.collector import ArxivCollector
from arxiv_researcher.analyzer import ResearchAnalyzer

__all__ = [
    "__version__",
    "ArxivCollector",
    "ResearchAnalyzer",
]
