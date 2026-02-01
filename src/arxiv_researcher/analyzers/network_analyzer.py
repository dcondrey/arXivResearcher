"""
Network Analysis Module
=======================
Analyze citation networks and co-authorship patterns:
- Citation graph construction
- Co-authorship network
- Centrality metrics
- Community detection
- Bridge papers identification
"""

from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
import math


class NetworkAnalyzer:
    """Analyze research networks (citations, co-authorship)."""

    def __init__(self):
        self.citation_graph = defaultdict(set)  # paper -> papers it cites
        self.cited_by_graph = defaultdict(set)  # paper -> papers that cite it
        self.coauthor_graph = defaultdict(set)  # author -> co-authors
        self.author_papers = defaultdict(list)  # author -> papers
        self.paper_authors = {}  # paper -> authors

    def build_networks(self, papers: List[Dict]) -> Dict:
        """Build all networks from paper data."""
        print(f"    Building networks for {len(papers)} papers...")

        # Build author networks
        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            authors = paper.get("author_list", [])

            self.paper_authors[arxiv_id] = authors

            for author in authors:
                self.author_papers[author].append(arxiv_id)

            # Co-authorship edges
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    self.coauthor_graph[a1].add(a2)
                    self.coauthor_graph[a2].add(a1)

        # Citation graph (if we have reference data)
        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            # References would come from S2 data
            references = paper.get("reference_arxiv_ids", [])
            for ref in references:
                self.citation_graph[arxiv_id].add(ref)
                self.cited_by_graph[ref].add(arxiv_id)

        return self.compute_all_metrics(papers)

    def compute_all_metrics(self, papers: List[Dict]) -> Dict:
        """Compute all network metrics."""
        return {
            "author_metrics": self.compute_author_metrics(),
            "paper_metrics": self.compute_paper_metrics(papers),
            "collaboration_patterns": self.analyze_collaboration_patterns(papers),
            "network_statistics": self.compute_network_stats(),
        }

    def compute_author_metrics(self) -> Dict:
        """Compute author-level network metrics."""
        metrics = {}

        for author, coauthors in self.coauthor_graph.items():
            papers = self.author_papers[author]

            # Degree centrality (number of unique co-authors)
            degree = len(coauthors)

            # Productivity
            paper_count = len(papers)

            # Collaboration diversity (unique co-authors per paper)
            collab_diversity = degree / paper_count if paper_count else 0

            metrics[author] = {
                "coauthor_count": degree,
                "paper_count": paper_count,
                "collaboration_diversity": round(collab_diversity, 2),
            }

        # Compute betweenness centrality approximation
        betweenness = self._approximate_betweenness(self.coauthor_graph, sample_size=100)
        for author, score in betweenness.items():
            if author in metrics:
                metrics[author]["betweenness_centrality"] = round(score, 4)

        # Rank authors
        sorted_by_degree = sorted(metrics.items(), key=lambda x: x[1]["coauthor_count"], reverse=True)
        sorted_by_papers = sorted(metrics.items(), key=lambda x: x[1]["paper_count"], reverse=True)
        sorted_by_between = sorted(metrics.items(),
                                   key=lambda x: x[1].get("betweenness_centrality", 0), reverse=True)

        return {
            "all_authors": metrics,
            "top_connected": [{"name": a, **m} for a, m in sorted_by_degree[:100]],
            "top_productive": [{"name": a, **m} for a, m in sorted_by_papers[:100]],
            "top_bridges": [{"name": a, **m} for a, m in sorted_by_between[:100]],
            "total_authors": len(metrics),
        }

    def compute_paper_metrics(self, papers: List[Dict]) -> Dict:
        """Compute paper-level network metrics."""
        paper_metrics = {}

        paper_lookup = {p.get("arxiv_id", ""): p for p in papers}

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")

            # Citation metrics
            in_degree = len(self.cited_by_graph.get(arxiv_id, set()))
            out_degree = len(self.citation_graph.get(arxiv_id, set()))

            # Author team metrics
            authors = paper.get("author_list", [])
            team_size = len(authors)

            # Team connectivity (how connected are the authors to each other's networks)
            team_total_connections = sum(
                len(self.coauthor_graph.get(a, set()))
                for a in authors
            )

            # Cross-category (papers that appear in multiple categories)
            categories = paper.get("category_list", [])
            is_cross_category = len(set(c.split(".")[0] for c in categories)) > 1

            paper_metrics[arxiv_id] = {
                "cited_by_count_internal": in_degree,  # Within dataset
                "reference_count_internal": out_degree,
                "team_size": team_size,
                "team_total_connections": team_total_connections,
                "is_cross_category": is_cross_category,
            }

        return paper_metrics

    def analyze_collaboration_patterns(self, papers: List[Dict]) -> Dict:
        """Analyze collaboration patterns over time and by category."""
        patterns = {
            "by_category": defaultdict(lambda: {"papers": 0, "total_authors": 0, "solo": 0}),
            "by_year": defaultdict(lambda: {"papers": 0, "total_authors": 0, "avg_team_size": 0}),
            "team_size_distribution": Counter(),
            "international_collaboration": 0,
            "cross_institution": 0,
        }

        for paper in papers:
            category = paper.get("primary_category", "unknown")
            year = paper.get("year", "unknown")
            authors = paper.get("author_list", [])
            team_size = len(authors)

            # Category patterns
            patterns["by_category"][category]["papers"] += 1
            patterns["by_category"][category]["total_authors"] += team_size
            if team_size == 1:
                patterns["by_category"][category]["solo"] += 1

            # Year patterns
            patterns["by_year"][year]["papers"] += 1
            patterns["by_year"][year]["total_authors"] += team_size

            # Team size distribution
            if team_size == 1:
                patterns["team_size_distribution"]["solo"] += 1
            elif team_size <= 3:
                patterns["team_size_distribution"]["small (2-3)"] += 1
            elif team_size <= 6:
                patterns["team_size_distribution"]["medium (4-6)"] += 1
            else:
                patterns["team_size_distribution"]["large (7+)"] += 1

            # Cross-institution (heuristic: different affiliations)
            institutions = paper.get("institutions", "").split("; ")
            if len(set(i.strip() for i in institutions if i.strip())) > 1:
                patterns["cross_institution"] += 1

            # International (multiple countries)
            countries = paper.get("institution_countries", "").split(", ")
            if len(set(c.strip() for c in countries if c.strip())) > 1:
                patterns["international_collaboration"] += 1

        # Compute averages
        for cat, data in patterns["by_category"].items():
            if data["papers"] > 0:
                data["avg_team_size"] = round(data["total_authors"] / data["papers"], 2)
                data["solo_rate"] = round(data["solo"] / data["papers"] * 100, 1)

        for year, data in patterns["by_year"].items():
            if data["papers"] > 0:
                data["avg_team_size"] = round(data["total_authors"] / data["papers"], 2)

        patterns["by_category"] = dict(patterns["by_category"])
        patterns["by_year"] = dict(sorted(patterns["by_year"].items()))
        patterns["team_size_distribution"] = dict(patterns["team_size_distribution"])

        return patterns

    def compute_network_stats(self) -> Dict:
        """Compute overall network statistics."""
        # Author network stats
        author_degrees = [len(coauthors) for coauthors in self.coauthor_graph.values()]

        if author_degrees:
            avg_coauthors = sum(author_degrees) / len(author_degrees)
            max_coauthors = max(author_degrees)
            density = sum(author_degrees) / (len(self.coauthor_graph) * (len(self.coauthor_graph) - 1)) if len(self.coauthor_graph) > 1 else 0
        else:
            avg_coauthors = max_coauthors = density = 0

        # Find connected components (simplified)
        components = self._find_components(self.coauthor_graph)

        return {
            "author_network": {
                "total_nodes": len(self.coauthor_graph),
                "total_edges": sum(author_degrees) // 2,
                "avg_degree": round(avg_coauthors, 2),
                "max_degree": max_coauthors,
                "density": round(density, 6),
                "num_components": len(components),
                "largest_component_size": max(len(c) for c in components) if components else 0,
            },
            "citation_network": {
                "total_papers": len(self.paper_authors),
                "papers_with_internal_citations": len(self.cited_by_graph),
            }
        }

    def _find_components(self, graph: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find connected components using BFS."""
        visited = set()
        components = []

        for node in graph:
            if node in visited:
                continue

            component = set()
            queue = [node]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in graph.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if component:
                components.append(component)

        return components

    def _approximate_betweenness(self, graph: Dict[str, Set[str]],
                                  sample_size: int = 100) -> Dict[str, float]:
        """Approximate betweenness centrality using sampling."""
        nodes = list(graph.keys())
        if len(nodes) < 3:
            return {}

        betweenness = defaultdict(float)

        # Sample random source-target pairs
        import random
        sample_size = min(sample_size, len(nodes))
        sampled_sources = random.sample(nodes, sample_size)

        for source in sampled_sources:
            # BFS to find shortest paths
            distances = {source: 0}
            predecessors = defaultdict(list)
            queue = [source]

            while queue:
                current = queue.pop(0)
                current_dist = distances[current]

                for neighbor in graph.get(current, set()):
                    if neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
                        predecessors[neighbor].append(current)
                    elif distances[neighbor] == current_dist + 1:
                        predecessors[neighbor].append(current)

            # Accumulate betweenness
            dependency = defaultdict(float)

            for node in sorted(distances.keys(), key=lambda x: distances[x], reverse=True):
                if node == source:
                    continue

                for pred in predecessors[node]:
                    dependency[pred] += (1 + dependency[node]) / len(predecessors[node])

                if node != source:
                    betweenness[node] += dependency[node]

        # Normalize
        n = len(nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            for node in betweenness:
                betweenness[node] *= norm

        return dict(betweenness)

    def find_bridge_papers(self, papers: List[Dict], top_n: int = 50) -> List[Dict]:
        """Find papers that bridge different research communities."""
        bridge_scores = []

        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            categories = paper.get("category_list", [])
            authors = paper.get("author_list", [])

            # Category bridging score
            unique_top_cats = len(set(c.split(".")[0] for c in categories))

            # Author network bridging
            author_communities = set()
            for author in authors:
                # Use primary category of author's most common papers
                author_cats = []
                for pid in self.author_papers.get(author, []):
                    # Would need to look up paper categories
                    pass

            # Methods bridging (from text analysis)
            methods = paper.get("methods_detected", [])

            score = (
                unique_top_cats * 2 +
                len(categories) +
                (1 if paper.get("is_cross_category") else 0)
            )

            bridge_scores.append({
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "bridge_score": score,
                "categories": categories,
                "category_count": len(categories),
                "unique_top_level_categories": unique_top_cats,
            })

        bridge_scores.sort(key=lambda x: x["bridge_score"], reverse=True)
        return bridge_scores[:top_n]

    def find_influential_authors(self, papers: List[Dict], top_n: int = 100) -> List[Dict]:
        """Find most influential authors based on network position and citations."""
        author_influence = defaultdict(lambda: {
            "total_citations": 0,
            "paper_count": 0,
            "coauthor_count": 0,
            "avg_citations": 0,
            "h_index_approx": 0,
            "categories": Counter(),
        })

        for paper in papers:
            authors = paper.get("author_list", [])
            citations = paper.get("citations") or 0
            category = paper.get("primary_category", "")

            for author in authors:
                author_influence[author]["total_citations"] += citations
                author_influence[author]["paper_count"] += 1
                author_influence[author]["categories"][category] += 1

        # Add network metrics
        for author in author_influence:
            data = author_influence[author]
            data["coauthor_count"] = len(self.coauthor_graph.get(author, set()))
            data["avg_citations"] = round(
                data["total_citations"] / data["paper_count"], 1
            ) if data["paper_count"] else 0
            data["primary_category"] = data["categories"].most_common(1)[0][0] if data["categories"] else ""
            data["categories"] = dict(data["categories"].most_common(5))

        # Compute composite influence score
        for author, data in author_influence.items():
            data["influence_score"] = (
                math.log(1 + data["total_citations"]) * 2 +
                math.log(1 + data["paper_count"]) * 1.5 +
                math.log(1 + data["coauthor_count"]) * 0.5
            )

        sorted_authors = sorted(
            author_influence.items(),
            key=lambda x: x[1]["influence_score"],
            reverse=True
        )

        return [{"name": a, **d} for a, d in sorted_authors[:top_n]]

    def export_for_visualization(self, papers: List[Dict], output_dir: str):
        """Export network data in formats suitable for visualization."""
        import json
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Coauthor network (top 500 authors by paper count)
        top_authors = sorted(
            self.author_papers.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:500]
        top_author_set = set(a for a, _ in top_authors)

        coauthor_nodes = []
        coauthor_edges = []
        seen_edges = set()

        for author, papers_list in top_authors:
            coauthor_nodes.append({
                "id": author,
                "paper_count": len(papers_list),
                "coauthor_count": len(self.coauthor_graph.get(author, set())),
            })

            for coauthor in self.coauthor_graph.get(author, set()):
                if coauthor in top_author_set:
                    edge_key = tuple(sorted([author, coauthor]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        coauthor_edges.append({
                            "source": author,
                            "target": coauthor,
                        })

        coauthor_network = {
            "nodes": coauthor_nodes,
            "edges": coauthor_edges,
        }

        with open(os.path.join(output_dir, "coauthor_network.json"), "w") as f:
            json.dump(coauthor_network, f, indent=2)

        print(f"    Exported network with {len(coauthor_nodes)} nodes, {len(coauthor_edges)} edges")
