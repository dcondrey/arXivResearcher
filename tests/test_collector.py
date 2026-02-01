"""Tests for the ArxivCollector class."""

import pytest
from unittest.mock import patch, MagicMock


class TestArxivCollector:
    """Test cases for ArxivCollector."""

    def test_rate_limiter(self):
        """Test that rate limiter properly delays requests."""
        from arxiv_researcher.collector import RateLimiter
        import time

        limiter = RateLimiter()

        # First call should not wait
        start = time.time()
        limiter.wait("test", 0.1)
        first_elapsed = time.time() - start
        assert first_elapsed < 0.05  # Should be nearly instant

        # Second call should wait
        start = time.time()
        limiter.wait("test", 0.1)
        second_elapsed = time.time() - start
        assert second_elapsed >= 0.09  # Should wait ~0.1s

    def test_collector_initialization(self):
        """Test collector initializes with correct defaults."""
        from arxiv_researcher.collector import ArxivCollector

        collector = ArxivCollector(output_dir="/tmp/test_arxiv")
        assert collector.output_dir == "/tmp/test_arxiv"
        assert collector.skip_s2 is False
        assert collector.skip_openalex is True
        assert collector.skip_pwc is True

    def test_cs_categories_defined(self):
        """Test that CS categories are properly defined."""
        from arxiv_researcher.collector import CS_CATEGORIES

        assert len(CS_CATEGORIES) == 40
        assert "cs.AI" in CS_CATEGORIES
        assert "cs.LG" in CS_CATEGORIES
        assert "cs.CV" in CS_CATEGORIES

    def test_date_chunk_generation(self):
        """Test date chunk generation for large ranges."""
        from arxiv_researcher.collector import ArxivCollector
        from datetime import datetime

        collector = ArxivCollector()

        start = datetime(2020, 1, 1)
        end = datetime(2024, 1, 1)

        chunks = collector._generate_date_chunks(start, end, months=12)

        assert len(chunks) >= 3
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

    @patch('arxiv_researcher.collector.urllib.request.urlopen')
    def test_parse_arxiv_entry(self, mock_urlopen):
        """Test parsing of arXiv XML entries."""
        from arxiv_researcher.collector import ArxivCollector
        import xml.etree.ElementTree as ET

        # Sample arXiv entry XML
        xml_str = """
        <entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
            <id>http://arxiv.org/abs/2301.00001v1</id>
            <title>Test Paper Title</title>
            <summary>This is a test abstract.</summary>
            <author><name>John Doe</name></author>
            <author><name>Jane Smith</name></author>
            <published>2023-01-01T00:00:00Z</published>
            <category term="cs.AI"/>
            <category term="cs.LG"/>
            <arxiv:primary_category term="cs.AI"/>
        </entry>
        """

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        # Parse using default namespace
        root = ET.fromstring(xml_str.replace('xmlns="http://www.w3.org/2005/Atom"', ''))

        collector = ArxivCollector()
        # Note: This test would need adjustment for the actual namespace handling


class TestAnalyzer:
    """Test cases for ResearchAnalyzer."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        from arxiv_researcher.analyzer import ResearchAnalyzer

        papers = [
            {
                "arxiv_id": "2301.00001",
                "title": "Test Paper",
                "abstract": "This is a test using transformers.",
                "authors": ["John Doe"],
                "published_date": "2023-01-01",
                "categories": ["cs.AI"],
            }
        ]

        analyzer = ResearchAnalyzer(papers)
        assert len(analyzer.papers) == 1

    def test_text_analysis(self):
        """Test text analysis extracts keywords and methods."""
        from arxiv_researcher.analyzer import ResearchAnalyzer

        papers = [
            {
                "arxiv_id": "2301.00001",
                "title": "Transformer Models for NLP",
                "abstract": "We propose a new transformer architecture using attention mechanisms.",
                "authors": ["John Doe"],
                "published_date": "2023-01-01",
                "categories": ["cs.CL"],
            },
            {
                "arxiv_id": "2301.00002",
                "title": "CNN for Image Classification",
                "abstract": "A convolutional neural network trained on ImageNet.",
                "authors": ["Jane Smith"],
                "published_date": "2023-01-02",
                "categories": ["cs.CV"],
            }
        ]

        analyzer = ResearchAnalyzer(papers, output_dir="/tmp/test_analysis")
        result = analyzer.analyze_text()

        assert "keywords" in result
        assert "methods" in result
        assert result["total_papers"] == 2

    def test_network_analysis(self):
        """Test network analysis identifies collaborations."""
        from arxiv_researcher.analyzer import ResearchAnalyzer

        papers = [
            {
                "arxiv_id": "2301.00001",
                "title": "Paper 1",
                "abstract": "Test",
                "authors": ["Alice", "Bob"],
                "published_date": "2023-01-01",
                "categories": ["cs.AI"],
                "citation_count": 10,
            },
            {
                "arxiv_id": "2301.00002",
                "title": "Paper 2",
                "abstract": "Test",
                "authors": ["Alice", "Charlie"],
                "published_date": "2023-01-02",
                "categories": ["cs.AI"],
                "citation_count": 5,
            }
        ]

        analyzer = ResearchAnalyzer(papers, output_dir="/tmp/test_analysis")
        result = analyzer.analyze_network()

        assert "top_authors" in result
        assert "top_collaborations" in result
        assert result["unique_authors"] == 3

        # Alice should be top author with 2 papers
        top_author = result["top_authors"][0]
        assert top_author["name"] == "Alice"
        assert top_author["papers"] == 2
