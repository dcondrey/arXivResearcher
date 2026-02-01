"""
Report Generator
================

Generate research intelligence reports in various formats.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class ReportGenerator:
    """Generate analysis reports in various formats."""

    def __init__(self, data_dir: str) -> None:
        """Initialize the report generator.

        Args:
            data_dir: Directory containing analysis results.
        """
        self.data_dir = data_dir
        self.analysis_dir = os.path.join(data_dir, "analysis")

    def generate(self, format: str = "markdown") -> str:
        """Generate a report in the specified format.

        Args:
            format: Output format ('markdown', 'html', 'json').

        Returns:
            Report content as a string.
        """
        # Load analysis data
        data = self._load_analysis_data()

        if format == "markdown":
            return self._generate_markdown(data)
        elif format == "html":
            return self._generate_html(data)
        elif format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _load_analysis_data(self) -> Dict[str, Any]:
        """Load all analysis JSON files."""
        data = {}

        # Load papers count
        papers_file = os.path.join(self.data_dir, "papers.json")
        if os.path.exists(papers_file):
            with open(papers_file, "r") as f:
                papers = json.load(f)
                data["paper_count"] = len(papers)

        # Load analysis files
        if os.path.exists(self.analysis_dir):
            for filename in os.listdir(self.analysis_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.analysis_dir, filename)
                    with open(filepath, "r") as f:
                        key = filename.replace("_analysis.json", "").replace(".json", "")
                        data[key] = json.load(f)

        return data

    def _generate_markdown(self, data: Dict[str, Any]) -> str:
        """Generate markdown report."""
        lines = [
            "# arXiv Research Intelligence Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
        ]

        # Overview
        paper_count = data.get("paper_count", 0)
        lines.extend([
            "## Overview",
            "",
            f"- **Total Papers Analyzed**: {paper_count:,}",
            "",
        ])

        # Trends
        if "trends" in data:
            trends = data["trends"]
            lines.extend([
                "## Emerging Topics",
                "",
            ])
            for term, growth in trends.get("emerging_topics", [])[:10]:
                growth_str = f"+{growth:.0%}" if growth != float('inf') else "new"
                lines.append(f"- **{term}**: {growth_str}")
            lines.append("")

        # Research Gaps
        if "gaps" in data:
            gaps = data["gaps"]
            lines.extend([
                "## Research Opportunities",
                "",
                "### Underexplored Intersections",
                "",
            ])
            for gap in gaps.get("underexplored_intersections", [])[:5]:
                cats = " + ".join(gap["categories"])
                score = gap["opportunity_score"]
                lines.append(f"- **{cats}**: Opportunity score {score:.2f}")
            lines.append("")

            lines.extend([
                "### Survey Opportunities",
                "",
            ])
            for opp in gaps.get("survey_opportunities", [])[:5]:
                lines.append(f"- **{opp['category']}**: {opp['paper_count']} papers, {opp['survey_count']} surveys")
            lines.append("")

        # Impact Factors
        if "impact" in data:
            impact = data["impact"]
            lines.extend([
                "## Success Factors",
                "",
            ])
            for factor in impact.get("success_factors", [])[:5]:
                lines.append(f"- **{factor['factor']}**: {factor['impact_multiplier']}x impact")
            lines.append("")

            lines.extend([
                "## Hot Papers (High Citation Velocity)",
                "",
            ])
            for paper in impact.get("hot_papers", [])[:5]:
                lines.append(f"- [{paper['title']}](https://arxiv.org/abs/{paper['arxiv_id']}) - {paper['velocity']} citations/month")
            lines.append("")

        # Network
        if "network" in data:
            network = data["network"]
            lines.extend([
                "## Top Authors",
                "",
                "| Author | Papers | Citations |",
                "|--------|--------|-----------|",
            ])
            for author in network.get("top_authors", [])[:10]:
                lines.append(f"| {author['name']} | {author['papers']} | {author['citations']} |")
            lines.append("")

        lines.extend([
            "---",
            "",
            "*Report generated by arXiv Researcher*",
        ])

        return "\n".join(lines)

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        markdown = self._generate_markdown(data)

        # Simple markdown to HTML conversion
        html = markdown
        html = html.replace("# ", "<h1>").replace("\n\n", "</h1>\n\n")
        html = html.replace("## ", "<h2>").replace("\n\n", "</h2>\n\n")
        html = html.replace("### ", "<h3>").replace("\n\n", "</h3>\n\n")
        html = html.replace("**", "<strong>").replace("**", "</strong>")
        html = html.replace("- ", "<li>").replace("\n", "</li>\n")

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>arXiv Research Intelligence Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
