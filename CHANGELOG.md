# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release

## [1.0.0] - 2024-02-01

### Added

#### Data Collection
- Multi-source data aggregation from arXiv, Semantic Scholar, OpenAlex, and Papers With Code
- Smart rate limiting with exponential backoff
- Checkpoint-based collection with resume capability
- Date range filtering with 2-year chunk optimization
- Support for all arXiv categories (CS, physics, math, stats, etc.)

#### Analysis Modules
- **Text Analyzer**: TF-IDF keyword extraction, 100+ method patterns, 50+ dataset patterns
- **Network Analyzer**: Co-authorship graphs, PageRank influence scoring, community detection
- **Trend Analyzer**: Time series analysis, momentum scoring, topic forecasting
- **Gap Analyzer**: Intersection analysis, survey opportunity detection
- **Impact Predictor**: Success factor identification, citation modeling
- **Embedding Analyzer**: Semantic clustering with sentence-transformers, UMAP visualization
- **Citation Graph**: Full citation network analysis, bridge paper detection
- **Citation Trajectory**: Yearly citation tracking, sleeping beauty detection
- **Benchmark Tracker**: SOTA tracking across benchmarks
- **Author Trajectory**: Rising star detection, topic migration analysis
- **Full-text Analyzer**: PDF parsing, section extraction, limitation mining

#### Visualization
- Interactive Streamlit dashboard with 6 pages
- PyVis network visualizations
- Plotly charts and heatmaps
- Export to JSON, CSV, Excel

#### CLI
- `arxiv-researcher collect` - Data collection command
- `arxiv-researcher analyze` - Analysis command
- `arxiv-researcher report` - Report generation
- `arxiv-dashboard` - Launch interactive dashboard

#### Infrastructure
- GitHub Actions CI/CD pipeline
- Automated weekly data updates
- PyPI package distribution
- Homebrew formula for macOS
- Scoop manifest for Windows

### Dependencies
- Python 3.10+
- requests, pandas, numpy, networkx, matplotlib, seaborn
- Optional: streamlit, plotly, sentence-transformers, pymupdf

## [0.1.0] - 2024-01-15

### Added
- Initial development version
- Basic arXiv data collection
- Simple analysis pipeline

[Unreleased]: https://github.com/dcondrey/arXivResearcher/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dcondrey/arXivResearcher/releases/tag/v1.0.0
[0.1.0]: https://github.com/dcondrey/arXivResearcher/releases/tag/v0.1.0
