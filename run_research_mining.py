#!/usr/bin/env python3
"""
arXiv Research Intelligence System
===================================
Comprehensive mining, analysis, and opportunity identification.

This script orchestrates:
1. Data collection from arXiv, Semantic Scholar, OpenAlex, Papers With Code
2. Text analysis (keywords, methods, datasets, contribution types)
3. Network analysis (co-authorship, citation patterns)
4. Trend analysis (emerging topics, declining areas)
5. Gap analysis (research opportunities, underexplored areas)
6. Impact prediction (what makes papers successful)

Usage:
    uv run python run_research_mining.py [options]

Examples:
    # Quick test
    uv run python run_research_mining.py --categories cs.AI,cs.LG --max-results 50

    # Full analysis
    uv run python run_research_mining.py --max-results 300

    # Analysis only (use existing data)
    uv run python run_research_mining.py --analysis-only --input arxiv_research_data
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arxiv_research.analyzers.text_analyzer import TextAnalyzer
from arxiv_research.analyzers.network_analyzer import NetworkAnalyzer
from arxiv_research.analyzers.trend_analyzer import TrendAnalyzer
from arxiv_research.analyzers.gap_analyzer import GapAnalyzer
from arxiv_research.analyzers.impact_predictor import ImpactPredictor

# New advanced analyzers
try:
    from arxiv_research.analyzers.embedding_analyzer import EmbeddingAnalyzer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Note: sentence-transformers not installed. Embedding analysis disabled.")

try:
    from arxiv_research.analyzers.citation_graph import CitationGraphAnalyzer
    HAS_CITATION_GRAPH = True
except ImportError:
    HAS_CITATION_GRAPH = False

try:
    from arxiv_research.analyzers.citation_trajectory import CitationTrajectoryAnalyzer
    HAS_TRAJECTORIES = True
except ImportError:
    HAS_TRAJECTORIES = False

try:
    from arxiv_research.analyzers.author_trajectory import AuthorTrajectoryAnalyzer
    HAS_AUTHOR_TRAJECTORIES = True
except ImportError:
    HAS_AUTHOR_TRAJECTORIES = False

try:
    from arxiv_research.analyzers.benchmark_tracker import BenchmarkTracker
    HAS_BENCHMARKS = True
except ImportError:
    HAS_BENCHMARKS = False

try:
    from arxiv_research.analyzers.fulltext_analyzer import FullTextAnalyzer
    HAS_FULLTEXT = True
except ImportError:
    HAS_FULLTEXT = False

# Import data collection from comprehensive script
from mine_arxiv_comprehensive import (
    fetch_arxiv_papers, enrich_with_semantic_scholar,
    enrich_with_openalex, enrich_with_papers_with_code,
    CS_CATEGORIES, ALL_CATEGORIES
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="arXiv Research Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:
    uv run python run_research_mining.py --categories cs.AI,cs.LG --max-results 50

  Full CS analysis:
    uv run python run_research_mining.py --max-results 300

  All categories:
    uv run python run_research_mining.py --all-categories --max-results 200

  Analysis only (existing data):
    uv run python run_research_mining.py --analysis-only --input arxiv_research_data
        """
    )

    # Data collection options
    parser.add_argument("--categories", type=str, default=None,
                       help="Comma-separated arXiv categories")
    parser.add_argument("--all-categories", action="store_true",
                       help="Use all categories (CS, stats, math, physics, etc.)")
    parser.add_argument("--max-results", type=int, default=500,
                       help="Max papers per category (default: 500)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date YYYY-MM-DD (default: 12 months ago)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date YYYY-MM-DD (default: today)")

    # Output options
    parser.add_argument("--output", type=str, default="arxiv_research_data",
                       help="Output directory (default: arxiv_research_data)")

    # Data source toggles
    parser.add_argument("--skip-s2", action="store_true",
                       help="Skip Semantic Scholar enrichment")
    parser.add_argument("--skip-openalex", action="store_true",
                       help="Skip OpenAlex enrichment")
    parser.add_argument("--skip-pwc", action="store_true",
                       help="Skip Papers With Code enrichment")

    # Analysis options
    parser.add_argument("--analysis-only", action="store_true",
                       help="Skip data collection, run analysis on existing data")
    parser.add_argument("--input", type=str, default=None,
                       help="Input directory for analysis-only mode")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis, only collect data")
    parser.add_argument("--skip-advanced", action="store_true",
                       help="Skip advanced analysis (embeddings, graphs, trajectories)")
    parser.add_argument("--fulltext", action="store_true",
                       help="Download and analyze PDFs (slow, requires disk space)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")

    return parser.parse_args()


def generate_date_chunks(start_date: datetime, end_date: datetime, months_per_chunk: int = 6) -> List[tuple]:
    """Generate date range chunks for large queries."""
    chunks = []
    current_start = start_date

    while current_start < end_date:
        current_end = current_start + timedelta(days=months_per_chunk * 30)
        if current_end > end_date:
            current_end = end_date
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return chunks


def collect_data(args) -> List[Dict]:
    """Collect data from all sources."""
    # Check for checkpoint resume
    checkpoint_dir = args.output if hasattr(args, 'output') else "arxiv_checkpoint"
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
    completed_categories = []
    all_papers = []

    if args.resume and os.path.exists(checkpoint_file):
        print(f"\n📂 Found checkpoint at {checkpoint_file}")
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            all_papers = checkpoint.get("papers", [])
            completed_categories = checkpoint.get("completed_categories", [])
            print(f"   Resuming with {len(all_papers)} papers from {len(completed_categories)} categories")
            print(f"   Last checkpoint: {checkpoint.get('timestamp', 'unknown')}")
        except Exception as e:
            print(f"   Warning: Could not load checkpoint: {e}")
            all_papers = []
            completed_categories = []

    # Date range
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else end_date - timedelta(days=365)

    # Categories
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    elif args.all_categories:
        categories = ALL_CATEGORIES
    else:
        categories = CS_CATEGORIES

    # Filter out completed categories if resuming
    if completed_categories:
        remaining_categories = [c for c in categories if c not in completed_categories]
        print(f"   Skipping {len(categories) - len(remaining_categories)} completed categories")
        categories = remaining_categories

    # For large date ranges (> 2 years), split into 2-year chunks
    total_days = (end_date - start_date).days
    use_chunks = total_days > 730  # More than 2 years

    if use_chunks:
        date_chunks = generate_date_chunks(start_date, end_date, months_per_chunk=24)
        print("\n" + "="*70)
        print("DATA COLLECTION (CHUNKED MODE)")
        print("="*70)
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Split into {len(date_chunks)} time chunks (2-year periods)")
        print(f"Categories: {len(categories)}")
        print(f"Max per category per chunk: {min(500, args.max_results)}")
        print("="*70)
    else:
        date_chunks = [(start_date, end_date)]
        print("\n" + "="*70)
        print("DATA COLLECTION")
        print("="*70)
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Categories: {len(categories)}")
        print(f"Max per category: {args.max_results}")
        print("="*70)

    # Note: all_papers is initialized at the start of function (may contain checkpoint data)

    # Calculate per-chunk limits
    if use_chunks:
        # Distribute max_results across chunks proportionally
        per_chunk_limit = min(500, max(100, args.max_results // len(date_chunks)))
    else:
        per_chunk_limit = args.max_results

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] {category}")
        category_papers = []

        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks):
            start_str = chunk_start.strftime("%Y%m%d")
            end_str = chunk_end.strftime("%Y%m%d")

            if use_chunks:
                print(f"  → Chunk {chunk_idx+1}/{len(date_chunks)}: {chunk_start.strftime('%Y-%m')} to {chunk_end.strftime('%Y-%m')}")
            else:
                print("  → Fetching from arXiv...")

            papers = fetch_arxiv_papers(
                category=category,
                start_date=start_str,
                end_date=end_str,
                max_results=per_chunk_limit,
                rate_limit=5.0  # More conservative rate limit for large collections
            )

            if use_chunks:
                print(f"      Found {len(papers)} papers in this period")

            category_papers.extend(papers)

            # Check if we've hit the overall limit for this category
            if len(category_papers) >= args.max_results:
                category_papers = category_papers[:args.max_results]
                break

        print(f"    Total for {category}: {len(category_papers)} papers")

        # Semantic Scholar
        if not args.skip_s2 and category_papers:
            print("  → Enriching with Semantic Scholar...")
            category_papers = enrich_with_semantic_scholar(category_papers, rate_limit=1.0, fetch_authors=True)

        all_papers.extend(category_papers)

        # Save checkpoint after each category
        checkpoint_dir = args.output if hasattr(args, 'output') else "arxiv_checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
        # Track all completed categories (previously completed + newly completed)
        all_completed = completed_categories + categories[:i]
        checkpoint_data = {
            "papers": all_papers,
            "completed_categories": all_completed,
            "timestamp": datetime.now().isoformat()
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
        print(f"    Checkpoint saved ({len(all_papers)} papers total, {len(all_completed)} categories done)")

    # Deduplicate
    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique_papers.append(p)

    print(f"\nTotal unique papers: {len(unique_papers)}")

    # OpenAlex (slower, run on unique set)
    if not args.skip_openalex and unique_papers:
        print("\n  → Enriching with OpenAlex (institutions, funders)...")
        unique_papers = enrich_with_openalex(unique_papers)

    # Papers With Code
    if not args.skip_pwc and unique_papers:
        print("\n  → Enriching with Papers With Code (code, benchmarks)...")
        unique_papers = enrich_with_papers_with_code(unique_papers)

    return unique_papers


def run_analysis(papers: List[Dict], output_dir: str, advanced: bool = True) -> Dict:
    """Run comprehensive analysis on papers."""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    results = {}
    step = 1
    total_steps = 5 + (5 if advanced else 0)  # Base + advanced analyzers

    # 1. Text Analysis
    print(f"\n[{step}/{total_steps}] Text Analysis...")
    text_analyzer = TextAnalyzer()
    papers, text_results = text_analyzer.analyze_all(papers)
    results["text_analysis"] = text_results
    step += 1

    # 2. Network Analysis
    print(f"\n[{step}/{total_steps}] Network Analysis...")
    network_analyzer = NetworkAnalyzer()
    network_results = network_analyzer.build_networks(papers)
    results["network_analysis"] = network_results

    # Export network for visualization
    network_analyzer.export_for_visualization(papers, os.path.join(output_dir, "networks"))

    # Find influential authors and bridge papers
    results["influential_authors"] = network_analyzer.find_influential_authors(papers)
    results["bridge_papers"] = network_analyzer.find_bridge_papers(papers)
    step += 1

    # 3. Trend Analysis
    print(f"\n[{step}/{total_steps}] Trend Analysis...")
    trend_analyzer = TrendAnalyzer()
    trend_results = trend_analyzer.analyze_trends(papers)
    results["trend_analysis"] = trend_results

    # Forecasts
    results["forecasts"] = trend_analyzer.forecast_categories(papers)
    step += 1

    # 4. Gap Analysis
    print(f"\n[{step}/{total_steps}] Gap Analysis...")
    gap_analyzer = GapAnalyzer()
    gap_results = gap_analyzer.analyze_gaps(papers)
    results["gap_analysis"] = gap_results
    step += 1

    # 5. Impact Prediction
    print(f"\n[{step}/{total_steps}] Impact Analysis...")
    impact_predictor = ImpactPredictor()
    impact_results = impact_predictor.analyze_success_factors(papers)
    results["impact_analysis"] = impact_results
    step += 1

    # === ADVANCED ANALYZERS ===
    if advanced:
        # 6. Semantic Embeddings
        if HAS_EMBEDDINGS:
            print(f"\n[{step}/{total_steps}] Semantic Embedding Analysis...")
            try:
                embedding_analyzer = EmbeddingAnalyzer()
                embedding_results = embedding_analyzer.analyze_papers(papers)
                results["embedding_analysis"] = embedding_results

                # Export visualization data
                embedding_analyzer.export_visualization_data(
                    papers, os.path.join(output_dir, "embeddings")
                )
            except Exception as e:
                print(f"    Embedding analysis failed: {e}")
        step += 1

        # 7. Citation Graph
        if HAS_CITATION_GRAPH:
            print(f"\n[{step}/{total_steps}] Citation Graph Analysis...")
            try:
                citation_graph = CitationGraphAnalyzer()
                graph_results = citation_graph.build_and_analyze(papers)
                results["citation_graph"] = graph_results

                # Export graph for visualization
                citation_graph.export_graph(os.path.join(output_dir, "graphs"))
            except Exception as e:
                print(f"    Citation graph analysis failed: {e}")
        step += 1

        # 8. Citation Trajectories
        if HAS_TRAJECTORIES:
            print(f"\n[{step}/{total_steps}] Citation Trajectory Analysis...")
            try:
                trajectory_analyzer = CitationTrajectoryAnalyzer()
                trajectory_results = trajectory_analyzer.analyze_trajectories(papers)
                results["citation_trajectories"] = trajectory_results
            except Exception as e:
                print(f"    Trajectory analysis failed: {e}")
        step += 1

        # 9. Author Trajectories
        if HAS_AUTHOR_TRAJECTORIES:
            print(f"\n[{step}/{total_steps}] Author Trajectory Analysis...")
            try:
                author_analyzer = AuthorTrajectoryAnalyzer()
                author_results = author_analyzer.analyze_authors(papers)
                results["author_trajectories"] = author_results
            except Exception as e:
                print(f"    Author trajectory analysis failed: {e}")
        step += 1

        # 10. Benchmark Tracking
        if HAS_BENCHMARKS:
            print(f"\n[{step}/{total_steps}] Benchmark Tracking...")
            try:
                benchmark_tracker = BenchmarkTracker()
                benchmark_results = benchmark_tracker.analyze_benchmarks(papers)
                results["benchmarks"] = benchmark_results
            except Exception as e:
                print(f"    Benchmark tracking failed: {e}")
        step += 1

    return results


def save_results(papers: List[Dict], analysis: Dict, output_dir: str):
    """Save all results."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Save papers as JSON (full data)
    papers_path = os.path.join(output_dir, "papers.json")
    with open(papers_path, "w", encoding="utf-8") as f:
        # Convert sets to lists for JSON
        json_papers = []
        for p in papers:
            jp = {}
            for k, v in p.items():
                if isinstance(v, set):
                    jp[k] = list(v)
                elif isinstance(v, list) and k == "author_list":
                    jp[k] = v  # Keep author list
                else:
                    jp[k] = v
            json_papers.append(jp)
        json.dump(json_papers, f, indent=2)
    print(f"  → Saved papers: {papers_path}")

    # Save papers as CSV (flattened)
    import csv
    csv_path = os.path.join(output_dir, "papers.csv")

    # Flatten for CSV
    csv_papers = []
    skip_fields = {"author_list", "category_list", "s2_author_ids"}
    for p in papers:
        csv_p = {k: v for k, v in p.items() if k not in skip_fields and not isinstance(v, (list, set, dict))}
        # Convert lists to strings
        for k, v in p.items():
            if isinstance(v, list) and k not in skip_fields:
                csv_p[k] = ", ".join(str(x) for x in v)
        csv_papers.append(csv_p)

    if csv_papers:
        fieldnames = list(csv_papers[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_papers)
    print(f"  → Saved CSV: {csv_path}")

    # Save analysis results
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    for name, data in analysis.items():
        path = os.path.join(analysis_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  → Saved analysis: {path}")

    # Generate summary report
    generate_summary_report(papers, analysis, output_dir)


def generate_summary_report(papers: List[Dict], analysis: Dict, output_dir: str):
    """Generate a human-readable summary report."""
    report_path = os.path.join(output_dir, "RESEARCH_INTELLIGENCE_REPORT.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# arXiv Research Intelligence Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Papers**: {len(papers):,}\n")

        # Date range
        dates = [p.get("published_date") for p in papers if p.get("published_date")]
        if dates:
            f.write(f"- **Date Range**: {min(dates)} to {max(dates)}\n")

        # Basic stats
        with_citations = [p for p in papers if p.get("citations") is not None]
        if with_citations:
            total_citations = sum(p["citations"] for p in with_citations)
            f.write(f"- **Total Citations**: {total_citations:,}\n")
            f.write(f"- **Average Citations**: {total_citations/len(with_citations):.1f}\n")

        with_code = sum(1 for p in papers if p.get("has_code"))
        f.write(f"- **Papers with Code**: {with_code:,} ({100*with_code/len(papers):.1f}%)\n")

        f.write("\n---\n\n")

        # Top Research Opportunities
        f.write("## 🎯 Top Research Opportunities\n\n")

        opportunities = analysis.get("gap_analysis", {}).get("opportunity_scores", {}).get("ranked_opportunities", [])
        if opportunities:
            f.write("| Rank | Category | Score | Papers | Avg Citations | Recommendation |\n")
            f.write("|------|----------|-------|--------|---------------|----------------|\n")
            for i, opp in enumerate(opportunities[:15], 1):
                f.write(f"| {i} | {opp['category']} | {opp['opportunity_score']} | {opp['paper_count']} | {opp['avg_citations']} | {opp['recommendation']} |\n")
            f.write("\n")

        # Emerging Topics
        f.write("## 📈 Emerging Topics\n\n")

        emerging = analysis.get("text_analysis", {}).get("emerging_terms", {}).get("emerging_terms", {})
        if emerging:
            f.write("| Term | Recent Count | Growth |\n")
            f.write("|------|--------------|--------|\n")
            for term, data in list(emerging.items())[:20]:
                growth = data.get("growth_rate", "new")
                f.write(f"| {term} | {data['recent_count']} | {growth}x |\n")
            f.write("\n")

        # Hot Papers
        f.write("## 🔥 Hot Papers (High Citation Velocity)\n\n")

        hot = analysis.get("trend_analysis", {}).get("hot_papers", [])
        if hot:
            for i, paper in enumerate(hot[:10], 1):
                f.write(f"{i}. **{paper['title'][:80]}{'...' if len(paper['title']) > 80 else ''}**\n")
                f.write(f"   - Citations/month: {paper['citations_per_month']:.2f} | Total: {paper['citations']}\n")
                f.write(f"   - Category: {paper['category']}\n\n")

        # Underexplored Intersections
        f.write("## 🔍 Underexplored Research Intersections\n\n")

        intersections = analysis.get("gap_analysis", {}).get("category_intersections", {}).get("underexplored_pairs", [])
        if intersections:
            f.write("| Categories | Actual Papers | Expected | Opportunity Score |\n")
            f.write("|------------|---------------|----------|-------------------|\n")
            for inter in intersections[:15]:
                cats = " + ".join(inter["categories"])
                f.write(f"| {cats} | {inter['actual_papers']} | {inter['expected_papers']} | {inter['opportunity_score']} |\n")
            f.write("\n")

        # Survey Opportunities
        f.write("## 📚 Survey Paper Opportunities\n\n")

        surveys = analysis.get("gap_analysis", {}).get("survey_opportunities", {}).get("category_survey_opportunities", [])
        if surveys:
            f.write("Categories with many papers but few/no surveys:\n\n")
            for s in surveys[:10]:
                f.write(f"- **{s['category']}**: {s['total_papers']} papers, {s['existing_surveys']} surveys\n")
            f.write("\n")

        # Success Factors
        f.write("## 🏆 What Makes Papers Successful\n\n")

        factors = analysis.get("impact_analysis", {}).get("success_formula", {}).get("factor_multipliers", {})
        if factors:
            f.write("| Factor | Citation Multiplier |\n")
            f.write("|--------|--------------------|\n")
            for factor, mult in factors.items():
                f.write(f"| {factor.replace('_', ' ').title()} | {mult}x |\n")
            f.write("\n")

        interpretation = analysis.get("impact_analysis", {}).get("success_formula", {}).get("interpretation", "")
        if interpretation:
            f.write(f"**Key Insight**: {interpretation}\n\n")

        # Top Authors
        f.write("## 👥 Most Influential Authors\n\n")

        authors = analysis.get("influential_authors", [])
        if authors:
            f.write("| Author | Papers | Citations | Avg Citations |\n")
            f.write("|--------|--------|-----------|---------------|\n")
            for a in authors[:15]:
                f.write(f"| {a['name'][:30]} | {a['paper_count']} | {a['total_citations']} | {a['avg_citations']} |\n")
            f.write("\n")

        # Methods Analysis
        f.write("## 🔧 Method Trends\n\n")

        methods = analysis.get("trend_analysis", {}).get("method_trends", {}).get("emerging_methods", [])
        if methods:
            f.write("**Emerging Methods:**\n")
            for m in methods[:10]:
                f.write(f"- {m['method'].replace('_', ' ').title()}: {m['total_mentions']} papers, momentum: {m['momentum']}\n")
            f.write("\n")

        f.write("\n---\n\n")
        f.write("*Report generated by arXiv Research Intelligence System*\n")

    print(f"  → Generated report: {report_path}")


def load_existing_data(input_dir: str) -> List[Dict]:
    """Load existing paper data for analysis."""
    papers_path = os.path.join(input_dir, "papers.json")
    if os.path.exists(papers_path):
        with open(papers_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"No papers.json found in {input_dir}")


def print_summary(papers: List[Dict], analysis: Dict):
    """Print summary to console."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n📊 Dataset: {len(papers):,} papers")

    # Top opportunities
    opportunities = analysis.get("gap_analysis", {}).get("opportunity_scores", {}).get("top_10_summary", [])
    if opportunities:
        print("\n🎯 Top Research Opportunities:")
        for opp in opportunities[:5]:
            print(f"   • {opp}")

    # Emerging terms
    emerging = analysis.get("text_analysis", {}).get("emerging_terms", {}).get("emerging_terms", {})
    if emerging:
        print("\n📈 Emerging Terms:")
        for term in list(emerging.keys())[:5]:
            print(f"   • {term}")

    # Hot papers
    hot = analysis.get("trend_analysis", {}).get("hot_papers", [])
    if hot:
        print("\n🔥 Hottest Papers:")
        for paper in hot[:3]:
            print(f"   • {paper['title'][:60]}... ({paper['citations_per_month']:.2f} cites/month)")

    # Success factors
    factors = analysis.get("impact_analysis", {}).get("success_formula", {}).get("top_factors", [])
    if factors:
        print("\n🏆 Top Success Factors:")
        for f in factors[:3]:
            print(f"   • {f.replace('_', ' ').title()}")


def main():
    args = parse_args()

    print("="*70)
    print("arXiv Research Intelligence System")
    print("="*70)

    # Collect or load data
    if args.analysis_only:
        input_dir = args.input or args.output
        print(f"\nLoading existing data from: {input_dir}")
        papers = load_existing_data(input_dir)
        print(f"Loaded {len(papers)} papers")
    else:
        papers = collect_data(args)

    if not papers:
        print("No papers to analyze!")
        return

    # Run analysis
    if not args.skip_analysis:
        analysis = run_analysis(papers, args.output, advanced=not args.skip_advanced)

        # Save everything
        save_results(papers, analysis, args.output)

        # Print summary
        print_summary(papers, analysis)
    else:
        # Just save data without analysis
        os.makedirs(args.output, exist_ok=True)
        papers_path = os.path.join(args.output, "papers.json")
        with open(papers_path, "w") as f:
            json.dump(papers, f, indent=2, default=str)
        print(f"\nSaved {len(papers)} papers to {papers_path}")

    print("\n" + "="*70)
    print(f"Complete! Results saved to: {args.output}/")
    print("="*70)


if __name__ == "__main__":
    main()
