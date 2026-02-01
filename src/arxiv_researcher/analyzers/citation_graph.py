"""
Citation Graph Mining Module
============================
Build and analyze citation networks using Semantic Scholar API:
- Fetch references and citations for papers
- Build networkx-compatible graph structure
- Find foundational papers (most cited within dataset)
- Find bridge papers (connect topic clusters)
- Detect citation cliques
- Compute PageRank centrality
- Trace idea lineage through citation chains
- Export graph for visualization
"""

import json
import os
import time
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None


class CitationGraphAnalyzer:
    """
    Analyze citation networks using Semantic Scholar API.

    Builds a directed graph where edges represent citations:
    - Edge A -> B means paper A cites paper B
    - References: papers that A cites (outgoing edges)
    - Citations: papers that cite A (incoming edges)
    """

    # Semantic Scholar batch API endpoint
    S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

    # Fields to fetch from S2 API
    S2_FIELDS = "paperId,externalIds,title,authors,year,citationCount,references,citations"

    # Rate limiting
    RATE_LIMIT_SECONDS = 1.0
    BATCH_SIZE = 100  # Max papers per batch request

    def __init__(self, cache_dir: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the citation graph analyzer.

        Args:
            cache_dir: Directory to cache API responses. If None, uses .citation_cache
            api_key: Semantic Scholar API key (optional, for higher rate limits)
        """
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx is required for CitationGraphAnalyzer. "
                "Install it with: pip install networkx"
            )
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required for CitationGraphAnalyzer. "
                "Install it with: pip install requests"
            )

        self.cache_dir = Path(cache_dir) if cache_dir else Path(".citation_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.environ.get("S2_API_KEY")

        # The citation graph
        self.graph: nx.DiGraph = nx.DiGraph()

        # Paper metadata cache
        self.paper_metadata: Dict[str, Dict] = {}

        # Mapping from arXiv ID to S2 paper ID
        self.arxiv_to_s2: Dict[str, str] = {}
        self.s2_to_arxiv: Dict[str, str] = {}

        # Track last API call time for rate limiting
        self._last_api_call = 0.0

        # Papers not found in S2
        self.not_found: Set[str] = set()

    def _get_cache_path(self, arxiv_id: str) -> Path:
        """Get cache file path for an arXiv ID."""
        # Use hash to avoid filesystem issues with special characters
        safe_id = arxiv_id.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_id}.json"

    def _load_from_cache(self, arxiv_id: str) -> Optional[Dict]:
        """Load paper data from cache if available."""
        cache_path = self._get_cache_path(arxiv_id)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_to_cache(self, arxiv_id: str, data: Dict) -> None:
        """Save paper data to cache."""
        cache_path = self._get_cache_path(arxiv_id)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not cache data for {arxiv_id}: {e}")

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.RATE_LIMIT_SECONDS:
            time.sleep(self.RATE_LIMIT_SECONDS - elapsed)
        self._last_api_call = time.time()

    def _fetch_papers_batch(self, arxiv_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch paper data from Semantic Scholar for a batch of arXiv IDs.

        Args:
            arxiv_ids: List of arXiv IDs to fetch

        Returns:
            Dict mapping arXiv ID to paper data
        """
        # Check cache first
        results = {}
        to_fetch = []

        for arxiv_id in arxiv_ids:
            cached = self._load_from_cache(arxiv_id)
            if cached is not None:
                results[arxiv_id] = cached
            elif arxiv_id in self.not_found:
                # Skip previously not-found papers
                results[arxiv_id] = {"not_found": True}
            else:
                to_fetch.append(arxiv_id)

        if not to_fetch:
            return results

        # Fetch from API in batches
        for i in range(0, len(to_fetch), self.BATCH_SIZE):
            batch = to_fetch[i:i + self.BATCH_SIZE]

            # Convert arXiv IDs to S2 format
            s2_ids = [f"ARXIV:{aid}" for aid in batch]

            self._rate_limit()

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["x-api-key"] = self.api_key

            try:
                response = requests.post(
                    self.S2_BATCH_URL,
                    params={"fields": self.S2_FIELDS},
                    headers=headers,
                    json={"ids": s2_ids},
                    timeout=30
                )

                if response.status_code == 200:
                    batch_results = response.json()

                    for arxiv_id, paper_data in zip(batch, batch_results):
                        if paper_data is None:
                            # Paper not found in S2
                            self.not_found.add(arxiv_id)
                            results[arxiv_id] = {"not_found": True}
                            self._save_to_cache(arxiv_id, {"not_found": True})
                        else:
                            results[arxiv_id] = paper_data
                            self._save_to_cache(arxiv_id, paper_data)

                            # Store ID mappings
                            s2_id = paper_data.get("paperId")
                            if s2_id:
                                self.arxiv_to_s2[arxiv_id] = s2_id
                                self.s2_to_arxiv[s2_id] = arxiv_id

                elif response.status_code == 429:
                    print(f"Rate limited by S2 API. Waiting 60 seconds...")
                    time.sleep(60)
                    # Retry this batch
                    return self._fetch_papers_batch(arxiv_ids)
                else:
                    print(f"S2 API error {response.status_code}: {response.text[:200]}")
                    # Mark all in batch as failed
                    for arxiv_id in batch:
                        results[arxiv_id] = {"error": response.status_code}

            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                for arxiv_id in batch:
                    results[arxiv_id] = {"error": str(e)}

        return results

    def build_citation_graph(self, papers: List[Dict]) -> nx.DiGraph:
        """
        Build a citation graph from a list of papers.

        The graph is directed: edge A -> B means A cites B.

        Args:
            papers: List of paper dicts with at least 'arxiv_id' field

        Returns:
            NetworkX DiGraph with citation relationships
        """
        print(f"    Building citation graph for {len(papers)} papers...")

        # Extract arXiv IDs
        arxiv_ids = [p.get("arxiv_id", "") for p in papers if p.get("arxiv_id")]
        arxiv_id_set = set(arxiv_ids)

        # Create nodes for all input papers first
        paper_lookup = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}

        for arxiv_id, paper in paper_lookup.items():
            self.graph.add_node(
                arxiv_id,
                title=paper.get("title", ""),
                year=paper.get("year"),
                category=paper.get("primary_category", ""),
                authors=paper.get("author_list", []),
                arxiv_id=arxiv_id,
            )

        # Fetch citation data from S2
        print(f"    Fetching citation data from Semantic Scholar...")
        s2_data = self._fetch_papers_batch(arxiv_ids)

        # Build edges
        edges_added = 0
        references_total = 0
        citations_total = 0

        for arxiv_id, data in s2_data.items():
            if data.get("not_found") or data.get("error"):
                continue

            # Store metadata
            self.paper_metadata[arxiv_id] = {
                "s2_id": data.get("paperId"),
                "s2_title": data.get("title"),
                "s2_year": data.get("year"),
                "s2_citation_count": data.get("citationCount", 0),
                "s2_authors": [a.get("name") for a in data.get("authors", [])],
            }

            # Add reference edges (papers this paper cites)
            references = data.get("references") or []
            for ref in references:
                if ref is None:
                    continue

                ref_id = ref.get("paperId")
                ref_arxiv = None

                # Try to find arXiv ID
                ext_ids = ref.get("externalIds") or {}
                if "ArXiv" in ext_ids:
                    ref_arxiv = ext_ids["ArXiv"]

                if ref_arxiv and ref_arxiv in arxiv_id_set:
                    # Reference is in our dataset
                    self.graph.add_edge(arxiv_id, ref_arxiv, type="cites")
                    edges_added += 1
                elif ref_id:
                    # Store S2 ID for external reference
                    self.s2_to_arxiv[ref_id] = ref_arxiv or ref_id

                references_total += 1

            # Add citation edges (papers that cite this paper)
            citations = data.get("citations") or []
            for cit in citations:
                if cit is None:
                    continue

                cit_id = cit.get("paperId")
                cit_arxiv = None

                ext_ids = cit.get("externalIds") or {}
                if "ArXiv" in ext_ids:
                    cit_arxiv = ext_ids["ArXiv"]

                if cit_arxiv and cit_arxiv in arxiv_id_set:
                    # Citing paper is in our dataset
                    self.graph.add_edge(cit_arxiv, arxiv_id, type="cites")
                    edges_added += 1

                citations_total += 1

        print(f"    Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"    Processed {references_total} references, {citations_total} citations")
        print(f"    Papers not found in S2: {len(self.not_found)}")

        return self.graph

    def find_foundational_papers(self, papers: List[Dict], top_n: int = 50) -> List[Dict]:
        """
        Find papers most cited WITHIN the dataset.

        These are the foundational works that many papers in the dataset build upon.

        Args:
            papers: List of paper dicts
            top_n: Number of top papers to return

        Returns:
            List of dicts with paper info and citation metrics
        """
        if self.graph.number_of_nodes() == 0:
            self.build_citation_graph(papers)

        paper_lookup = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}

        # In-degree = number of papers that cite this paper (within dataset)
        in_degrees = dict(self.graph.in_degree())

        foundational = []
        for arxiv_id, in_degree in sorted(in_degrees.items(), key=lambda x: x[1], reverse=True):
            if in_degree == 0:
                continue

            paper = paper_lookup.get(arxiv_id, {})
            s2_meta = self.paper_metadata.get(arxiv_id, {})

            foundational.append({
                "arxiv_id": arxiv_id,
                "title": paper.get("title", s2_meta.get("s2_title", "")),
                "year": paper.get("year", s2_meta.get("s2_year")),
                "category": paper.get("primary_category", ""),
                "authors": paper.get("author_list", s2_meta.get("s2_authors", []))[:5],
                "internal_citations": in_degree,
                "total_s2_citations": s2_meta.get("s2_citation_count", 0),
                "internal_citation_ratio": round(
                    in_degree / max(s2_meta.get("s2_citation_count", 1), 1), 3
                ),
            })

            if len(foundational) >= top_n:
                break

        return foundational

    def find_bridge_papers(self, papers: List[Dict], top_n: int = 50) -> List[Dict]:
        """
        Find papers that connect different topic clusters.

        Bridge papers are identified by high betweenness centrality in the citation graph.

        Args:
            papers: List of paper dicts
            top_n: Number of top papers to return

        Returns:
            List of dicts with paper info and bridge metrics
        """
        if self.graph.number_of_nodes() == 0:
            self.build_citation_graph(papers)

        if self.graph.number_of_nodes() < 3:
            return []

        paper_lookup = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}

        # Compute betweenness centrality
        # Use approximate algorithm for large graphs
        if self.graph.number_of_nodes() > 1000:
            betweenness = nx.betweenness_centrality(
                self.graph, k=min(500, self.graph.number_of_nodes())
            )
        else:
            betweenness = nx.betweenness_centrality(self.graph)

        # Also compute categories of neighbors to measure cross-topic bridging
        bridge_papers = []

        for arxiv_id, centrality in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
            if centrality == 0:
                continue

            paper = paper_lookup.get(arxiv_id, {})

            # Get categories of papers that cite and are cited by this paper
            predecessor_cats = set()
            successor_cats = set()

            for pred in self.graph.predecessors(arxiv_id):
                pred_paper = paper_lookup.get(pred, {})
                if pred_paper.get("primary_category"):
                    predecessor_cats.add(pred_paper["primary_category"].split(".")[0])

            for succ in self.graph.successors(arxiv_id):
                succ_paper = paper_lookup.get(succ, {})
                if succ_paper.get("primary_category"):
                    successor_cats.add(succ_paper["primary_category"].split(".")[0])

            # Category diversity score
            all_cats = predecessor_cats | successor_cats
            category_diversity = len(all_cats)

            bridge_papers.append({
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "category": paper.get("primary_category", ""),
                "betweenness_centrality": round(centrality, 6),
                "category_diversity": category_diversity,
                "connected_categories": list(all_cats)[:10],
                "in_degree": self.graph.in_degree(arxiv_id),
                "out_degree": self.graph.out_degree(arxiv_id),
                "bridge_score": round(centrality * (1 + 0.1 * category_diversity), 6),
            })

            if len(bridge_papers) >= top_n:
                break

        # Re-sort by bridge score
        bridge_papers.sort(key=lambda x: x["bridge_score"], reverse=True)

        return bridge_papers

    def find_citation_cliques(self, papers: List[Dict], min_size: int = 3) -> List[Dict]:
        """
        Find groups of papers that frequently cite each other.

        These are research clusters or communities that are tightly interconnected.

        Args:
            papers: List of paper dicts
            min_size: Minimum clique size

        Returns:
            List of cliques with member papers
        """
        if self.graph.number_of_nodes() == 0:
            self.build_citation_graph(papers)

        paper_lookup = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}

        # Convert to undirected for clique detection
        # (we want mutual citation relationships)
        undirected = self.graph.to_undirected()

        # Find cliques
        try:
            cliques = list(nx.find_cliques(undirected))
        except Exception as e:
            print(f"Warning: Clique detection failed: {e}")
            return []

        # Filter by size and format results
        clique_results = []

        for i, clique in enumerate(sorted(cliques, key=len, reverse=True)):
            if len(clique) < min_size:
                break

            members = []
            categories = Counter()
            years = []

            for arxiv_id in clique:
                paper = paper_lookup.get(arxiv_id, {})
                members.append({
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", "")[:100],
                    "year": paper.get("year"),
                })
                if paper.get("primary_category"):
                    categories[paper["primary_category"]] += 1
                if paper.get("year"):
                    years.append(paper["year"])

            clique_results.append({
                "clique_id": i + 1,
                "size": len(clique),
                "members": members,
                "dominant_category": categories.most_common(1)[0][0] if categories else None,
                "category_distribution": dict(categories.most_common(5)),
                "year_range": f"{min(years)}-{max(years)}" if years else None,
            })

            if len(clique_results) >= 50:  # Limit number of cliques returned
                break

        return clique_results

    def compute_pagerank(self, papers: List[Dict], top_n: int = 100) -> List[Dict]:
        """
        Compute PageRank centrality in the citation network.

        PageRank measures importance based on the structure of incoming citations,
        giving more weight to citations from important papers.

        Args:
            papers: List of paper dicts
            top_n: Number of top papers to return

        Returns:
            List of papers with PageRank scores
        """
        if self.graph.number_of_nodes() == 0:
            self.build_citation_graph(papers)

        if self.graph.number_of_nodes() == 0:
            return []

        paper_lookup = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}

        # Compute PageRank
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except Exception as e:
            print(f"Warning: PageRank computation failed: {e}")
            return []

        ranked_papers = []

        for arxiv_id, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
            paper = paper_lookup.get(arxiv_id, {})
            s2_meta = self.paper_metadata.get(arxiv_id, {})

            ranked_papers.append({
                "arxiv_id": arxiv_id,
                "title": paper.get("title", s2_meta.get("s2_title", "")),
                "year": paper.get("year", s2_meta.get("s2_year")),
                "category": paper.get("primary_category", ""),
                "pagerank": round(score, 8),
                "in_degree": self.graph.in_degree(arxiv_id),
                "out_degree": self.graph.out_degree(arxiv_id),
                "s2_citations": s2_meta.get("s2_citation_count", 0),
            })

            if len(ranked_papers) >= top_n:
                break

        return ranked_papers

    def trace_idea_lineage(self, paper_arxiv_id: str, papers: List[Dict],
                           max_depth: int = 5) -> Dict:
        """
        Trace the citation chain for a specific paper.

        Shows both:
        - Ancestors: papers this paper builds upon (transitively)
        - Descendants: papers that build upon this paper

        Args:
            paper_arxiv_id: arXiv ID of the paper to trace
            papers: List of paper dicts
            max_depth: Maximum depth to trace

        Returns:
            Dict with ancestry and descendant information
        """
        if self.graph.number_of_nodes() == 0:
            self.build_citation_graph(papers)

        if paper_arxiv_id not in self.graph:
            return {"error": f"Paper {paper_arxiv_id} not found in graph"}

        paper_lookup = {p.get("arxiv_id"): p for p in papers if p.get("arxiv_id")}

        def get_paper_info(arxiv_id: str) -> Dict:
            paper = paper_lookup.get(arxiv_id, {})
            return {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", "")[:100],
                "year": paper.get("year"),
                "category": paper.get("primary_category", ""),
            }

        # Trace ancestors (papers this paper cites, transitively)
        ancestors_by_depth = {}
        visited = {paper_arxiv_id}
        current_level = [paper_arxiv_id]

        for depth in range(1, max_depth + 1):
            next_level = []
            for node in current_level:
                for successor in self.graph.successors(node):  # Outgoing edges = references
                    if successor not in visited and successor in paper_lookup:
                        visited.add(successor)
                        next_level.append(successor)

            if next_level:
                ancestors_by_depth[depth] = [get_paper_info(n) for n in next_level]
            current_level = next_level

            if not current_level:
                break

        # Trace descendants (papers that cite this paper, transitively)
        descendants_by_depth = {}
        visited = {paper_arxiv_id}
        current_level = [paper_arxiv_id]

        for depth in range(1, max_depth + 1):
            next_level = []
            for node in current_level:
                for predecessor in self.graph.predecessors(node):  # Incoming edges = citations
                    if predecessor not in visited and predecessor in paper_lookup:
                        visited.add(predecessor)
                        next_level.append(predecessor)

            if next_level:
                descendants_by_depth[depth] = [get_paper_info(n) for n in next_level]
            current_level = next_level

            if not current_level:
                break

        # Find root papers (papers with no outgoing references in dataset)
        root_ancestors = []
        for arxiv_id in self.graph.nodes():
            if self.graph.out_degree(arxiv_id) == 0 and self.graph.in_degree(arxiv_id) > 0:
                if nx.has_path(self.graph, paper_arxiv_id, arxiv_id):
                    root_ancestors.append(get_paper_info(arxiv_id))

        root_paper = paper_lookup.get(paper_arxiv_id, {})

        return {
            "paper": get_paper_info(paper_arxiv_id),
            "ancestors_by_depth": ancestors_by_depth,
            "total_ancestors": sum(len(v) for v in ancestors_by_depth.values()),
            "descendants_by_depth": descendants_by_depth,
            "total_descendants": sum(len(v) for v in descendants_by_depth.values()),
            "root_ancestors": root_ancestors[:10],  # Foundational papers
            "direct_references": [
                get_paper_info(n) for n in list(self.graph.successors(paper_arxiv_id))[:20]
            ],
            "direct_citations": [
                get_paper_info(n) for n in list(self.graph.predecessors(paper_arxiv_id))[:20]
            ],
        }

    def export_graph(self, output_path: str) -> None:
        """
        Export the citation graph as JSON for visualization.

        The output format is compatible with D3.js and other graph visualization tools.

        Args:
            output_path: Path to write the JSON file
        """
        if self.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty, nothing to export")
            return

        # Build nodes list
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            s2_meta = self.paper_metadata.get(node, {})
            nodes.append({
                "id": node,
                "title": attrs.get("title", s2_meta.get("s2_title", ""))[:100],
                "year": attrs.get("year", s2_meta.get("s2_year")),
                "category": attrs.get("category", ""),
                "authors": attrs.get("authors", s2_meta.get("s2_authors", []))[:3],
                "in_degree": self.graph.in_degree(node),
                "out_degree": self.graph.out_degree(node),
                "s2_citations": s2_meta.get("s2_citation_count", 0),
            })

        # Build edges list
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": attrs.get("type", "cites"),
            })

        # Compute some graph statistics
        stats = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 6) if self.graph.number_of_nodes() > 0 else 0,
            "avg_in_degree": round(
                sum(d for _, d in self.graph.in_degree()) / max(self.graph.number_of_nodes(), 1), 2
            ),
            "avg_out_degree": round(
                sum(d for _, d in self.graph.out_degree()) / max(self.graph.number_of_nodes(), 1), 2
            ),
        }

        # Check if graph is weakly connected
        if self.graph.number_of_nodes() > 0:
            try:
                stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(
                    self.graph
                )
            except Exception:
                stats["num_weakly_connected_components"] = None

        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "statistics": stats,
            "papers_not_found_in_s2": list(self.not_found)[:100],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"    Exported graph to {output_path}")
        print(f"    Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")

    def get_graph_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the citation graph.

        Returns:
            Dict with various graph metrics
        """
        if self.graph.number_of_nodes() == 0:
            return {"error": "Graph is empty"}

        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]

        stats = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 6),
            "in_degree": {
                "min": min(in_degrees),
                "max": max(in_degrees),
                "mean": round(sum(in_degrees) / len(in_degrees), 2),
                "median": sorted(in_degrees)[len(in_degrees) // 2],
            },
            "out_degree": {
                "min": min(out_degrees),
                "max": max(out_degrees),
                "mean": round(sum(out_degrees) / len(out_degrees), 2),
                "median": sorted(out_degrees)[len(out_degrees) // 2],
            },
            "papers_not_found_in_s2": len(self.not_found),
        }

        # Connected components
        try:
            stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(
                self.graph
            )
            components = list(nx.weakly_connected_components(self.graph))
            if components:
                stats["largest_component_size"] = max(len(c) for c in components)
        except Exception:
            pass

        # DAG properties
        try:
            stats["is_dag"] = nx.is_directed_acyclic_graph(self.graph)
        except Exception:
            pass

        return stats

    def analyze_all(self, papers: List[Dict]) -> Dict:
        """
        Run all citation graph analyses.

        Args:
            papers: List of paper dicts

        Returns:
            Dict with all analysis results
        """
        print(f"    Running full citation graph analysis for {len(papers)} papers...")

        # Build graph first
        self.build_citation_graph(papers)

        results = {
            "graph_statistics": self.get_graph_statistics(),
            "foundational_papers": self.find_foundational_papers(papers, top_n=30),
            "bridge_papers": self.find_bridge_papers(papers, top_n=30),
            "pagerank_ranking": self.compute_pagerank(papers, top_n=50),
            "citation_cliques": self.find_citation_cliques(papers, min_size=3)[:20],
        }

        return results
