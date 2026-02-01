"""
Analysis Modules
================

Specialized analyzers for research paper analysis.

Core Analyzers (always available):
- TextAnalyzer: Keyword extraction, method/dataset detection
- NetworkAnalyzer: Co-authorship and citation networks
- TrendAnalyzer: Topic trends and forecasting
- GapAnalyzer: Research opportunities
- ImpactPredictor: Success factors

Advanced Analyzers (require optional dependencies):
- EmbeddingAnalyzer: Semantic clustering (requires sentence-transformers)
- CitationGraphAnalyzer: Full citation networks
- CitationTrajectoryAnalyzer: Citation patterns over time
- AuthorTrajectoryAnalyzer: Author career analysis
- BenchmarkTracker: SOTA tracking
- FullTextAnalyzer: PDF analysis (requires pymupdf)
"""

# Core analyzers
from arxiv_researcher.analyzers.text_analyzer import TextAnalyzer
from arxiv_researcher.analyzers.network_analyzer import NetworkAnalyzer
from arxiv_researcher.analyzers.trend_analyzer import TrendAnalyzer
from arxiv_researcher.analyzers.gap_analyzer import GapAnalyzer
from arxiv_researcher.analyzers.impact_predictor import ImpactPredictor

__all__ = [
    "TextAnalyzer",
    "NetworkAnalyzer",
    "TrendAnalyzer",
    "GapAnalyzer",
    "ImpactPredictor",
]

# Advanced analyzers - import with graceful fallback
try:
    from arxiv_researcher.analyzers.embedding_analyzer import EmbeddingAnalyzer
    __all__.append("EmbeddingAnalyzer")
except ImportError:
    EmbeddingAnalyzer = None

try:
    from arxiv_researcher.analyzers.citation_graph import CitationGraphAnalyzer
    __all__.append("CitationGraphAnalyzer")
except ImportError:
    CitationGraphAnalyzer = None

try:
    from arxiv_researcher.analyzers.citation_trajectory import CitationTrajectoryAnalyzer
    __all__.append("CitationTrajectoryAnalyzer")
except ImportError:
    CitationTrajectoryAnalyzer = None

try:
    from arxiv_researcher.analyzers.author_trajectory import AuthorTrajectoryAnalyzer
    __all__.append("AuthorTrajectoryAnalyzer")
except ImportError:
    AuthorTrajectoryAnalyzer = None

try:
    from arxiv_researcher.analyzers.benchmark_tracker import BenchmarkTracker
    __all__.append("BenchmarkTracker")
except ImportError:
    BenchmarkTracker = None

try:
    from arxiv_researcher.analyzers.fulltext_analyzer import FullTextAnalyzer
    __all__.append("FullTextAnalyzer")
except ImportError:
    FullTextAnalyzer = None
