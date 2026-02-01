<p align="center">
  <img src="docs/images/logo.svg" alt="arXiv Researcher" width="120">
</p>

<h1 align="center">arXiv Researcher</h1>

<p align="center">
  <strong>Research Intelligence and Visualization Dashboard for arXiv</strong>
</p>

<p align="center">
  <a href="https://github.com/dcondrey/arXivResearcher/actions/workflows/ci.yml">
    <img src="https://github.com/dcondrey/arXivResearcher/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/arxiv-researcher/">
    <img src="https://img.shields.io/pypi/v/arxiv-researcher.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/arxiv-researcher/">
    <img src="https://img.shields.io/pypi/pyversions/arxiv-researcher.svg" alt="Python versions">
  </a>
  <a href="https://github.com/dcondrey/arXivResearcher/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/dcondrey/arXivResearcher.svg" alt="License">
  </a>
  <a href="https://codecov.io/gh/dcondrey/arXivResearcher">
    <img src="https://codecov.io/gh/dcondrey/arXivResearcher/branch/main/graph/badge.svg" alt="codecov">
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

---

**arXiv Researcher** is a comprehensive tool for mining, analyzing, and visualizing research data from [arXiv](https://arxiv.org). It helps researchers identify trends, discover opportunities, and understand the research landscape across computer science, physics, mathematics, and more.

## Features

### Data Collection
- **Multi-source aggregation**: Collects data from arXiv, Semantic Scholar, OpenAlex, and Papers With Code
- **Rich metadata**: Full abstracts, author affiliations, citations, references, code repositories
- **Historical data**: Access papers from 1991 to present
- **Smart rate limiting**: Respectful API usage with exponential backoff and checkpointing

### Analysis Capabilities
- **Text Analysis**: Keyword extraction, method detection, dataset identification
- **Network Analysis**: Co-authorship networks, citation graphs, collaboration patterns
- **Trend Analysis**: Emerging topics, declining areas, research momentum
- **Gap Analysis**: Underexplored intersections, survey opportunities
- **Impact Prediction**: Success factors, citation trajectory forecasting
- **Semantic Embeddings**: Paper clustering, similarity search, topic modeling

### Visualization
- **Interactive Dashboard**: Streamlit-based web interface
- **Network Graphs**: PyVis collaboration and citation networks
- **Trend Charts**: Plotly time series and heatmaps
- **Export Options**: JSON, CSV, Excel, and PNG/SVG charts

## Installation

### Using pip (recommended)

```bash
pip install arxiv-researcher
```

### Using uv (fast)

```bash
uv pip install arxiv-researcher
```

### Using Homebrew (macOS)

```bash
brew tap dcondrey/tap
brew install arxiv-researcher
```

### Using Scoop (Windows)

```powershell
scoop bucket add arxiv-researcher https://github.com/dcondrey/scoop-arxiv-researcher
scoop install arxiv-researcher
```

### From source

```bash
git clone https://github.com/dcondrey/arXivResearcher.git
cd arXivResearcher
pip install -e ".[all]"
```

### Optional dependencies

```bash
# For the interactive dashboard
pip install arxiv-researcher[dashboard]

# For semantic embeddings and clustering
pip install arxiv-researcher[embeddings]

# For PDF full-text analysis
pip install arxiv-researcher[fulltext]

# Install everything
pip install arxiv-researcher[all]
```

## Quick Start

### Command Line

```bash
# Collect papers from specific categories
arxiv-researcher collect --categories cs.AI,cs.LG --max-results 500

# Collect with date range
arxiv-researcher collect --categories cs.CV --start-date 2023-01-01 --max-results 1000

# Run analysis on collected data
arxiv-researcher analyze --input arxiv_data/papers.json

# Generate a report
arxiv-researcher report --input arxiv_data --format markdown

# Launch the interactive dashboard
arxiv-dashboard --data arxiv_data
```

### Python API

```python
from arxiv_researcher import ArxivCollector, ResearchAnalyzer
from datetime import datetime, timedelta

# Collect papers
collector = ArxivCollector()
papers = collector.collect(
    categories=['cs.AI', 'cs.LG'],
    max_results=500,
    start_date=datetime.now() - timedelta(days=365)
)

# Analyze
analyzer = ResearchAnalyzer(papers)

# Get trending topics
trends = analyzer.analyze_trends()
print(f"Emerging topics: {trends['emerging'][:5]}")

# Find research gaps
gaps = analyzer.find_gaps()
print(f"Top opportunity: {gaps['underexplored'][0]}")

# Predict impact factors
factors = analyzer.predict_impact()
print(f"Key success factor: {factors['top_factors'][0]}")
```

### Interactive Dashboard

```bash
arxiv-dashboard --data arxiv_data
```

The dashboard provides:
- **Overview**: Dataset statistics and key metrics
- **Trends**: Topic emergence and decline over time
- **Opportunities**: Research gaps and survey opportunities
- **Paper Explorer**: Search and filter papers
- **Author Network**: Collaboration visualization
- **Success Factors**: What makes papers impactful

## Data Sources

| Source | Data Provided | Rate Limit |
|--------|--------------|------------|
| [arXiv](https://arxiv.org) | Papers, abstracts, categories, authors | 3s between requests |
| [Semantic Scholar](https://www.semanticscholar.org) | Citations, references, author metrics | 100 req/5min |
| [OpenAlex](https://openalex.org) | Institutions, funders, concepts | 10 req/s |
| [Papers With Code](https://paperswithcode.com) | Code repos, benchmarks, datasets | 1 req/s |

## Analysis Modules

| Module | Description |
|--------|-------------|
| **Text Analyzer** | TF-IDF keywords, 100+ method patterns, 50+ dataset patterns |
| **Network Analyzer** | Co-authorship graphs, PageRank, community detection |
| **Trend Analyzer** | Time series, momentum scoring, forecasting |
| **Gap Analyzer** | Intersection analysis, survey opportunities |
| **Impact Predictor** | Success factors, citation modeling |
| **Embedding Analyzer** | Semantic clustering, similarity search |
| **Citation Graph** | Full citation networks, idea lineage |
| **Author Trajectory** | Rising stars, topic migration |

## Output Example

```
RESEARCH INTELLIGENCE REPORT
============================

Dataset: 150,000 papers (2000-2024)

Top Research Opportunities:
  • cs.AI + cs.CL: NLP for AI systems (Score: 8.5)
  • cs.LG + cs.CV: Vision-language models (Score: 8.2)
  • cs.CR + cs.LG: ML security (Score: 7.9)

Emerging Topics:
  • Large language models (+450% growth)
  • Diffusion models (+380% growth)
  • Neural radiance fields (+290% growth)

Success Factors:
  • Open access: 7.3x more citations
  • Code available: 4.2x more citations
  • Senior co-author: 3.5x more citations
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/dcondrey/arXivResearcher.git
cd arXivResearcher
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/ tests/
```

## Roadmap

- [ ] Real-time paper monitoring with alerts
- [ ] LLM-powered research question generation
- [ ] Citation prediction models
- [ ] Integration with reference managers
- [ ] Collaborative filtering recommendations
- [ ] REST API for programmatic access

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{arxiv_researcher,
  author = {Condrey, David},
  title = {arXiv Researcher: Research Intelligence and Visualization Dashboard},
  url = {https://github.com/dcondrey/arXivResearcher},
  year = {2024}
}
```

## Acknowledgments

- [arXiv](https://arxiv.org) for open access to research
- [Semantic Scholar](https://www.semanticscholar.org) for citation data
- [OpenAlex](https://openalex.org) for institutional metadata
- [Papers With Code](https://paperswithcode.com) for code and benchmarks

---

<p align="center">
  Made with science by <a href="https://github.com/dcondrey">David Condrey</a>
</p>
