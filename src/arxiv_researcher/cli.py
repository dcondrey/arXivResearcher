#!/usr/bin/env python3
"""
arXiv Researcher CLI
====================

Command-line interface for the arXiv Research Intelligence System.

Usage:
    arxiv-researcher collect [options]
    arxiv-researcher analyze [options]
    arxiv-researcher report [options]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from arxiv_researcher import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="arxiv-researcher",
        description="arXiv Research Intelligence System - Mine, analyze, and visualize research data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick collection test
  arxiv-researcher collect --categories cs.AI,cs.LG --max-results 50

  # Full CS analysis
  arxiv-researcher collect --max-results 500 --output my_analysis

  # Analysis only (existing data)
  arxiv-researcher analyze --input my_analysis/papers.json

  # Generate report
  arxiv-researcher report --input my_analysis

  # Launch dashboard
  arxiv-dashboard --data my_analysis
        """
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Collect command
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect papers from arXiv and enrich with external data"
    )
    collect_parser.add_argument(
        "--categories", "-c",
        type=str,
        default=None,
        help="Comma-separated arXiv categories (e.g., cs.AI,cs.LG)"
    )
    collect_parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Use all CS, stats, math, and physics categories"
    )
    collect_parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=500,
        help="Maximum papers per category (default: 500)"
    )
    collect_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 12 months ago)"
    )
    collect_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    collect_parser.add_argument(
        "--output", "-o",
        type=str,
        default="arxiv_data",
        help="Output directory (default: arxiv_data)"
    )
    collect_parser.add_argument(
        "--skip-s2",
        action="store_true",
        help="Skip Semantic Scholar enrichment"
    )
    collect_parser.add_argument(
        "--skip-openalex",
        action="store_true",
        help="Skip OpenAlex enrichment"
    )
    collect_parser.add_argument(
        "--skip-pwc",
        action="store_true",
        help="Skip Papers With Code enrichment"
    )
    collect_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run analysis on collected papers"
    )
    analyze_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSON file or directory with papers"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    analyze_parser.add_argument(
        "--skip-advanced",
        action="store_true",
        help="Skip advanced analysis (embeddings, graphs)"
    )
    analyze_parser.add_argument(
        "--fulltext",
        action="store_true",
        help="Download and analyze PDFs"
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate analysis report"
    )
    report_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory with analysis results"
    )
    report_parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    report_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file (default: stdout)"
    )

    return parser


def cmd_collect(args: argparse.Namespace) -> int:
    """Execute the collect command."""
    from arxiv_researcher.collector import ArxivCollector

    print(f"arXiv Researcher v{__version__}")
    print("=" * 60)

    collector = ArxivCollector(
        output_dir=args.output,
        skip_s2=args.skip_s2,
        skip_openalex=getattr(args, 'skip_openalex', True),
        skip_pwc=getattr(args, 'skip_pwc', True),
    )

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    elif args.all_categories:
        categories = "all"

    # Parse dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Run collection
    papers = collector.collect(
        categories=categories,
        max_results=args.max_results,
        start_date=start_date,
        end_date=end_date,
        resume=args.resume,
    )

    print(f"\nCollected {len(papers):,} papers")
    print(f"Results saved to: {args.output}/")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the analyze command."""
    from arxiv_researcher.analyzer import ResearchAnalyzer

    print(f"arXiv Researcher v{__version__}")
    print("=" * 60)

    # Load papers
    input_path = args.input
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, "papers.json")

    print(f"Loading papers from: {input_path}")
    with open(input_path, "r") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers):,} papers")

    # Determine output directory
    output_dir = args.output
    if output_dir is None:
        if os.path.isdir(args.input):
            output_dir = args.input
        else:
            output_dir = os.path.dirname(args.input) or "."

    # Run analysis
    analyzer = ResearchAnalyzer(papers, output_dir=output_dir)
    results = analyzer.analyze(
        advanced=not args.skip_advanced,
        fulltext=args.fulltext,
    )

    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}/analysis/")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Execute the report command."""
    from arxiv_researcher.reporter import ReportGenerator

    print(f"arXiv Researcher v{__version__}")
    print("=" * 60)

    generator = ReportGenerator(args.input)
    report = generator.generate(format=args.format)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "collect":
            return cmd_collect(args)
        elif args.command == "analyze":
            return cmd_analyze(args)
        elif args.command == "report":
            return cmd_report(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
