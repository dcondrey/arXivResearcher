"""
Semantic Embedding Analysis Module
==================================
Analyze papers using dense vector embeddings:
- Generate semantic embeddings from title + abstract
- Cluster papers by semantic similarity
- Find similar papers to a given paper
- Identify research gaps in embedding space
- Visualize the research landscape
- Detect emerging clusters over time
"""

import os
import json
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np


class EmbeddingAnalyzer:
    """Analyze papers using semantic embeddings and clustering.

    This analyzer uses sentence-transformers to generate dense vector
    embeddings of paper titles and abstracts, enabling semantic similarity
    search, clustering, and research landscape visualization.

    Attributes:
        model_name: Name of the sentence-transformers model to use.
        model: The loaded sentence-transformers model (lazy-loaded).
        embeddings_cache: In-memory cache of computed embeddings.
        cache_dir: Directory for persistent embedding cache.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the embedding analyzer.

        Args:
            model_name: Name of the sentence-transformers model.
                Default is 'all-MiniLM-L6-v2' which is fast and high quality.
            cache_dir: Directory to cache embeddings to disk.
                If None, embeddings are only cached in memory.
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                If None, auto-detects based on availability.
        """
        self.model_name = model_name
        self.model = None
        self.device = device
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding analysis. "
                    "Install it with: pip install sentence-transformers"
                )

            print(f"    Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"    Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")

    def _get_paper_text(self, paper: Dict) -> str:
        """Extract text from paper for embedding.

        Combines title and abstract, handling missing abstracts gracefully.

        Args:
            paper: Paper dictionary with 'title' and optionally 'abstract'.

        Returns:
            Combined text string for embedding.
        """
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""

        # Clean up text
        title = title.strip()
        abstract = abstract.strip()

        if abstract:
            return f"{title}. {abstract}"
        else:
            # If no abstract, just use title (repeated for slightly more context)
            return title

    def _get_paper_id(self, paper: Dict) -> str:
        """Get a unique identifier for a paper.

        Args:
            paper: Paper dictionary.

        Returns:
            Unique string identifier (arxiv_id if available, else hash of title).
        """
        if paper.get("arxiv_id"):
            return paper["arxiv_id"]
        else:
            # Fall back to hash of title if no arxiv_id
            title = paper.get("title", "")
            return hashlib.md5(title.encode()).hexdigest()[:16]

    def _get_cache_path(self, paper_id: str) -> Optional[Path]:
        """Get the file path for a cached embedding."""
        if self.cache_dir:
            return self.cache_dir / f"{paper_id}.npy"
        return None

    def _load_cached_embedding(self, paper_id: str) -> Optional[np.ndarray]:
        """Load an embedding from cache (memory or disk)."""
        # Check memory cache first
        if paper_id in self.embeddings_cache:
            return self.embeddings_cache[paper_id]

        # Check disk cache
        cache_path = self._get_cache_path(paper_id)
        if cache_path and cache_path.exists():
            embedding = np.load(cache_path)
            self.embeddings_cache[paper_id] = embedding
            return embedding

        return None

    def _save_embedding_to_cache(self, paper_id: str, embedding: np.ndarray):
        """Save an embedding to cache (memory and optionally disk)."""
        self.embeddings_cache[paper_id] = embedding

        cache_path = self._get_cache_path(paper_id)
        if cache_path:
            np.save(cache_path, embedding)

    def embed_papers(
        self,
        papers: List[Dict],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for a list of papers.

        Args:
            papers: List of paper dictionaries with 'title' and 'abstract'.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress during embedding.

        Returns:
            Dictionary mapping paper IDs to embedding vectors (numpy arrays).
        """
        self._load_model()

        # Separate papers that need embedding from those in cache
        papers_to_embed = []
        paper_ids = []
        embeddings = {}

        for paper in papers:
            paper_id = self._get_paper_id(paper)
            cached = self._load_cached_embedding(paper_id)

            if cached is not None:
                embeddings[paper_id] = cached
            else:
                papers_to_embed.append(paper)
                paper_ids.append(paper_id)

        if papers_to_embed:
            if show_progress:
                print(f"    Generating embeddings for {len(papers_to_embed)} papers "
                      f"({len(embeddings)} cached)...")

            # Get texts for embedding
            texts = [self._get_paper_text(p) for p in papers_to_embed]

            # Generate embeddings in batches
            new_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            # Store embeddings
            for paper_id, embedding in zip(paper_ids, new_embeddings):
                self._save_embedding_to_cache(paper_id, embedding)
                embeddings[paper_id] = embedding

        return embeddings

    def cluster_papers(
        self,
        papers: List[Dict],
        n_clusters: int = 20,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        method: str = "kmeans",
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Cluster papers by semantic similarity.

        Args:
            papers: List of paper dictionaries.
            n_clusters: Number of clusters to create.
            embeddings: Pre-computed embeddings (optional, will compute if not provided).
            method: Clustering method ('kmeans' or 'agglomerative').
            random_state: Random state for reproducibility.

        Returns:
            Dictionary with:
                - cluster_assignments: Dict mapping paper_id to cluster_id
                - cluster_labels: Dict mapping cluster_id to representative label
                - cluster_sizes: Dict mapping cluster_id to number of papers
                - cluster_centers: Dict mapping cluster_id to center embedding
                - paper_distances: Dict mapping paper_id to distance from cluster center
                - silhouette_score: Overall clustering quality score
        """
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install it with: pip install scikit-learn"
            )

        # Get embeddings if not provided
        if embeddings is None:
            embeddings = self.embed_papers(papers)

        # Build embedding matrix (ordered by paper index)
        paper_ids = [self._get_paper_id(p) for p in papers]
        valid_papers = []
        valid_ids = []
        valid_embeddings = []

        for paper, paper_id in zip(papers, paper_ids):
            if paper_id in embeddings:
                valid_papers.append(paper)
                valid_ids.append(paper_id)
                valid_embeddings.append(embeddings[paper_id])

        if len(valid_embeddings) < n_clusters:
            n_clusters = max(2, len(valid_embeddings) // 2)
            print(f"    Adjusted n_clusters to {n_clusters} due to small dataset size")

        embedding_matrix = np.vstack(valid_embeddings)

        print(f"    Clustering {len(valid_embeddings)} papers into {n_clusters} clusters...")

        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
            )
            cluster_labels = clusterer.fit_predict(embedding_matrix)
            cluster_centers = clusterer.cluster_centers_
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(embedding_matrix)
            # Compute centers for agglomerative
            cluster_centers = []
            for i in range(n_clusters):
                mask = cluster_labels == i
                if mask.sum() > 0:
                    cluster_centers.append(embedding_matrix[mask].mean(axis=0))
                else:
                    cluster_centers.append(np.zeros(embedding_matrix.shape[1]))
            cluster_centers = np.array(cluster_centers)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Compute silhouette score
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(embedding_matrix, cluster_labels)
        else:
            sil_score = 0.0

        # Build result dictionaries
        cluster_assignments = {}
        cluster_papers = defaultdict(list)
        paper_distances = {}

        for idx, (paper_id, label) in enumerate(zip(valid_ids, cluster_labels)):
            cluster_assignments[paper_id] = int(label)
            cluster_papers[int(label)].append(valid_papers[idx])

            # Compute distance to cluster center
            dist = np.linalg.norm(valid_embeddings[idx] - cluster_centers[label])
            paper_distances[paper_id] = float(dist)

        # Generate cluster labels from most common keywords/titles
        cluster_labels_dict = {}
        cluster_sizes = {}

        for cluster_id, cluster_paper_list in cluster_papers.items():
            cluster_sizes[cluster_id] = len(cluster_paper_list)

            # Extract keywords from titles for labeling
            title_words = defaultdict(int)
            for p in cluster_paper_list:
                title = p.get("title", "").lower()
                words = [w for w in title.split() if len(w) > 3 and w.isalpha()]
                for word in words:
                    if word not in {"with", "from", "that", "this", "using", "based"}:
                        title_words[word] += 1

            # Get top keywords
            top_words = sorted(title_words.items(), key=lambda x: x[1], reverse=True)[:3]
            label = ", ".join(w for w, _ in top_words) if top_words else f"Cluster {cluster_id}"
            cluster_labels_dict[cluster_id] = label

        # Convert cluster centers to serializable format
        centers_dict = {
            int(i): cluster_centers[i].tolist()
            for i in range(len(cluster_centers))
        }

        return {
            "cluster_assignments": cluster_assignments,
            "cluster_labels": cluster_labels_dict,
            "cluster_sizes": cluster_sizes,
            "cluster_centers": centers_dict,
            "paper_distances": paper_distances,
            "silhouette_score": float(sil_score),
            "n_clusters": n_clusters,
            "method": method,
            "total_papers": len(valid_ids),
        }

    def find_similar_papers(
        self,
        paper: Dict,
        papers: List[Dict],
        top_n: int = 10,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        exclude_self: bool = True,
    ) -> List[Dict]:
        """Find papers most similar to a given paper.

        Args:
            paper: The reference paper to find similar papers for.
            papers: List of candidate papers to search.
            top_n: Number of similar papers to return.
            embeddings: Pre-computed embeddings (optional).
            exclude_self: Whether to exclude the reference paper from results.

        Returns:
            List of dictionaries with similar paper info and similarity scores.
        """
        self._load_model()

        # Get embedding for reference paper
        paper_id = self._get_paper_id(paper)

        if embeddings and paper_id in embeddings:
            query_embedding = embeddings[paper_id]
        else:
            query_text = self._get_paper_text(paper)
            query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]

        # Get embeddings for all candidates
        if embeddings is None:
            embeddings = self.embed_papers(papers)

        # Compute similarities
        similarities = []

        for candidate in papers:
            candidate_id = self._get_paper_id(candidate)

            if exclude_self and candidate_id == paper_id:
                continue

            if candidate_id not in embeddings:
                continue

            candidate_embedding = embeddings[candidate_id]

            # Cosine similarity
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )

            similarities.append({
                "arxiv_id": candidate.get("arxiv_id", candidate_id),
                "title": candidate.get("title", ""),
                "similarity_score": float(similarity),
                "primary_category": candidate.get("primary_category", ""),
                "citations": candidate.get("citations"),
                "published_date": candidate.get("published_date", ""),
            })

        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

        return similarities[:top_n]

    def find_embedding_gaps(
        self,
        papers: List[Dict],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        n_regions: int = 10,
        density_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Find sparse regions in embedding space that represent potential research gaps.

        This method identifies areas of the semantic space with low paper density,
        which may represent underexplored research directions.

        Args:
            papers: List of paper dictionaries.
            embeddings: Pre-computed embeddings (optional).
            n_regions: Number of sparse regions to identify.
            density_threshold: Papers below this density percentile are considered sparse.

        Returns:
            Dictionary with:
                - sparse_regions: List of sparse region descriptions
                - boundary_papers: Papers at the edges of clusters (potential bridges)
                - density_scores: Per-paper local density scores
                - overall_coverage: Measure of how well-covered the space is
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError(
                "scikit-learn is required for gap analysis. "
                "Install it with: pip install scikit-learn"
            )

        # Get embeddings
        if embeddings is None:
            embeddings = self.embed_papers(papers)

        # Build embedding matrix
        paper_ids = [self._get_paper_id(p) for p in papers]
        valid_papers = []
        valid_ids = []
        valid_embeddings = []

        for paper, paper_id in zip(papers, paper_ids):
            if paper_id in embeddings:
                valid_papers.append(paper)
                valid_ids.append(paper_id)
                valid_embeddings.append(embeddings[paper_id])

        if len(valid_embeddings) < 10:
            return {
                "sparse_regions": [],
                "boundary_papers": [],
                "density_scores": {},
                "overall_coverage": 1.0,
                "message": "Not enough papers for gap analysis",
            }

        embedding_matrix = np.vstack(valid_embeddings)

        print(f"    Analyzing embedding space for {len(valid_embeddings)} papers...")

        # Compute local density using k-nearest neighbors
        k = min(10, len(valid_embeddings) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(embedding_matrix)

        distances, indices = nn.kneighbors(embedding_matrix)

        # Average distance to k neighbors as inverse density proxy
        avg_distances = distances[:, 1:].mean(axis=1)  # Exclude self

        # Normalize to 0-1 range (higher = more isolated)
        min_dist = avg_distances.min()
        max_dist = avg_distances.max()
        if max_dist > min_dist:
            isolation_scores = (avg_distances - min_dist) / (max_dist - min_dist)
        else:
            isolation_scores = np.zeros_like(avg_distances)

        # Find papers in sparse regions (high isolation)
        threshold = np.percentile(isolation_scores, 100 - density_threshold * 100)
        sparse_mask = isolation_scores >= threshold

        # Identify sparse regions
        sparse_papers = []
        for idx in np.where(sparse_mask)[0]:
            paper = valid_papers[idx]
            sparse_papers.append({
                "arxiv_id": paper.get("arxiv_id", valid_ids[idx]),
                "title": paper.get("title", ""),
                "isolation_score": float(isolation_scores[idx]),
                "primary_category": paper.get("primary_category", ""),
                "avg_neighbor_distance": float(avg_distances[idx]),
            })

        sparse_papers.sort(key=lambda x: x["isolation_score"], reverse=True)

        # Find boundary papers (between clusters)
        # These have high variance in their neighbor distances
        neighbor_dist_variance = distances[:, 1:].var(axis=1)
        boundary_threshold = np.percentile(neighbor_dist_variance, 75)
        boundary_mask = neighbor_dist_variance >= boundary_threshold

        boundary_papers = []
        for idx in np.where(boundary_mask)[0]:
            paper = valid_papers[idx]
            boundary_papers.append({
                "arxiv_id": paper.get("arxiv_id", valid_ids[idx]),
                "title": paper.get("title", ""),
                "distance_variance": float(neighbor_dist_variance[idx]),
                "primary_category": paper.get("primary_category", ""),
            })

        boundary_papers.sort(key=lambda x: x["distance_variance"], reverse=True)

        # Overall coverage metric (inverse of average isolation)
        overall_coverage = float(1.0 - isolation_scores.mean())

        # Density scores for all papers
        density_scores = {
            valid_ids[idx]: float(1.0 - isolation_scores[idx])
            for idx in range(len(valid_ids))
        }

        # Identify potential gap directions by analyzing sparse paper topics
        gap_topics = defaultdict(int)
        for sp in sparse_papers[:20]:
            title_words = sp["title"].lower().split()
            for word in title_words:
                if len(word) > 4 and word.isalpha():
                    gap_topics[word] += 1

        top_gap_topics = sorted(gap_topics.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "sparse_regions": sparse_papers[:n_regions],
            "boundary_papers": boundary_papers[:n_regions],
            "density_scores": density_scores,
            "overall_coverage": overall_coverage,
            "potential_gap_topics": [t for t, _ in top_gap_topics],
            "total_papers_analyzed": len(valid_embeddings),
            "sparse_paper_count": int(sparse_mask.sum()),
        }

    def visualize_landscape(
        self,
        papers: List[Dict],
        output_path: str,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        method: str = "umap",
        perplexity: int = 30,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        cluster_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate 2D projection data for visualization of the research landscape.

        Args:
            papers: List of paper dictionaries.
            output_path: Path to save the visualization data (JSON).
            embeddings: Pre-computed embeddings (optional).
            method: Dimensionality reduction method ('umap' or 'tsne').
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance parameter for UMAP.
            cluster_info: Optional cluster information to include.

        Returns:
            Dictionary with projection data and metadata.
        """
        # Get embeddings
        if embeddings is None:
            embeddings = self.embed_papers(papers)

        # Build embedding matrix
        paper_ids = [self._get_paper_id(p) for p in papers]
        valid_papers = []
        valid_ids = []
        valid_embeddings = []

        for paper, paper_id in zip(papers, paper_ids):
            if paper_id in embeddings:
                valid_papers.append(paper)
                valid_ids.append(paper_id)
                valid_embeddings.append(embeddings[paper_id])

        if len(valid_embeddings) < 5:
            return {
                "points": [],
                "message": "Not enough papers for visualization",
            }

        embedding_matrix = np.vstack(valid_embeddings)

        print(f"    Computing {method.upper()} projection for {len(valid_embeddings)} papers...")

        # Perform dimensionality reduction
        if method == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "umap-learn is required for UMAP visualization. "
                    "Install it with: pip install umap-learn"
                )

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, len(valid_embeddings) - 1),
                min_dist=min_dist,
                metric="cosine",
                random_state=42,
            )
            projection = reducer.fit_transform(embedding_matrix)

        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ImportError(
                    "scikit-learn is required for t-SNE visualization. "
                    "Install it with: pip install scikit-learn"
                )

            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(valid_embeddings) // 3),
                random_state=42,
            )
            projection = tsne.fit_transform(embedding_matrix)
        else:
            raise ValueError(f"Unknown visualization method: {method}")

        # Build visualization data
        points = []
        for idx, paper in enumerate(valid_papers):
            paper_id = valid_ids[idx]
            point = {
                "arxiv_id": paper.get("arxiv_id", paper_id),
                "title": paper.get("title", ""),
                "x": float(projection[idx, 0]),
                "y": float(projection[idx, 1]),
                "primary_category": paper.get("primary_category", ""),
                "citations": paper.get("citations"),
                "year": paper.get("year", ""),
            }

            # Add cluster info if available
            if cluster_info and paper_id in cluster_info.get("cluster_assignments", {}):
                cluster_id = cluster_info["cluster_assignments"][paper_id]
                point["cluster_id"] = cluster_id
                point["cluster_label"] = cluster_info.get("cluster_labels", {}).get(cluster_id, "")

            points.append(point)

        result = {
            "points": points,
            "method": method,
            "total_papers": len(points),
            "bounds": {
                "x_min": float(projection[:, 0].min()),
                "x_max": float(projection[:, 0].max()),
                "y_min": float(projection[:, 1].min()),
                "y_max": float(projection[:, 1].max()),
            },
        }

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"    Saved visualization data to {output_path}")

        return result

    def detect_emerging_clusters(
        self,
        papers: List[Dict],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        time_field: str = "published_date",
        n_clusters: int = 30,
        recent_fraction: float = 0.25,
    ) -> Dict[str, Any]:
        """Detect clusters that are growing over time (emerging research areas).

        This method identifies clusters where recent papers are overrepresented,
        suggesting emerging research directions.

        Args:
            papers: List of paper dictionaries with time information.
            embeddings: Pre-computed embeddings (optional).
            time_field: Field to use for temporal analysis.
            n_clusters: Number of clusters to analyze.
            recent_fraction: Fraction of time period to consider "recent".

        Returns:
            Dictionary with:
                - emerging_clusters: Clusters with growing representation
                - declining_clusters: Clusters with shrinking representation
                - stable_clusters: Clusters with stable representation
                - cluster_dynamics: Detailed dynamics for each cluster
        """
        # Get embeddings and cluster
        if embeddings is None:
            embeddings = self.embed_papers(papers)

        cluster_info = self.cluster_papers(
            papers,
            n_clusters=n_clusters,
            embeddings=embeddings,
        )

        # Sort papers by time
        papers_with_time = []
        for paper in papers:
            time_val = paper.get(time_field, "")
            paper_id = self._get_paper_id(paper)
            if time_val and paper_id in cluster_info["cluster_assignments"]:
                papers_with_time.append((time_val, paper_id, paper))

        papers_with_time.sort(key=lambda x: x[0])

        if len(papers_with_time) < 10:
            return {
                "emerging_clusters": [],
                "declining_clusters": [],
                "stable_clusters": [],
                "cluster_dynamics": {},
                "message": "Not enough papers with time information",
            }

        # Split into early and recent periods
        split_idx = int(len(papers_with_time) * (1 - recent_fraction))
        early_papers = papers_with_time[:split_idx]
        recent_papers = papers_with_time[split_idx:]

        early_period = f"{early_papers[0][0][:7]} to {early_papers[-1][0][:7]}" if early_papers else ""
        recent_period = f"{recent_papers[0][0][:7]} to {recent_papers[-1][0][:7]}" if recent_papers else ""

        print(f"    Analyzing cluster dynamics: early ({len(early_papers)} papers) "
              f"vs recent ({len(recent_papers)} papers)...")

        # Count cluster membership in each period
        early_counts = defaultdict(int)
        recent_counts = defaultdict(int)

        for _, paper_id, _ in early_papers:
            cluster = cluster_info["cluster_assignments"][paper_id]
            early_counts[cluster] += 1

        for _, paper_id, _ in recent_papers:
            cluster = cluster_info["cluster_assignments"][paper_id]
            recent_counts[cluster] += 1

        # Analyze dynamics
        all_clusters = set(early_counts.keys()) | set(recent_counts.keys())
        cluster_dynamics = {}

        total_early = len(early_papers)
        total_recent = len(recent_papers)

        for cluster_id in all_clusters:
            early_count = early_counts.get(cluster_id, 0)
            recent_count = recent_counts.get(cluster_id, 0)

            early_rate = early_count / total_early if total_early > 0 else 0
            recent_rate = recent_count / total_recent if total_recent > 0 else 0

            if early_rate > 0:
                growth_ratio = recent_rate / early_rate
            elif recent_count > 0:
                growth_ratio = float("inf")
            else:
                growth_ratio = 1.0

            cluster_dynamics[cluster_id] = {
                "cluster_id": cluster_id,
                "cluster_label": cluster_info["cluster_labels"].get(cluster_id, f"Cluster {cluster_id}"),
                "early_count": early_count,
                "recent_count": recent_count,
                "early_rate": round(early_rate * 100, 2),
                "recent_rate": round(recent_rate * 100, 2),
                "growth_ratio": round(growth_ratio, 3) if growth_ratio != float("inf") else "new",
                "total_size": cluster_info["cluster_sizes"].get(cluster_id, 0),
            }

        # Classify clusters
        emerging = []
        declining = []
        stable = []

        for cluster_id, dynamics in cluster_dynamics.items():
            growth = dynamics["growth_ratio"]

            if growth == "new" or (isinstance(growth, (int, float)) and growth > 1.5):
                emerging.append(dynamics)
            elif isinstance(growth, (int, float)) and growth < 0.7:
                declining.append(dynamics)
            else:
                stable.append(dynamics)

        # Sort by growth
        emerging.sort(
            key=lambda x: x["recent_count"] * (100 if x["growth_ratio"] == "new" else x["growth_ratio"]),
            reverse=True
        )
        declining.sort(key=lambda x: x["growth_ratio"])

        return {
            "emerging_clusters": emerging,
            "declining_clusters": declining,
            "stable_clusters": stable,
            "cluster_dynamics": cluster_dynamics,
            "early_period": early_period,
            "recent_period": recent_period,
            "total_clusters": len(all_clusters),
            "emerging_count": len(emerging),
            "declining_count": len(declining),
            "stable_count": len(stable),
        }

    def analyze_all(
        self,
        papers: List[Dict],
        output_dir: Optional[str] = None,
        n_clusters: int = 20,
    ) -> Tuple[List[Dict], Dict]:
        """Run comprehensive embedding analysis on a corpus of papers.

        Args:
            papers: List of paper dictionaries.
            output_dir: Optional directory to save outputs.
            n_clusters: Number of clusters for analysis.

        Returns:
            Tuple of (updated papers list, corpus-level analysis dict).
        """
        print(f"    Running embedding analysis for {len(papers)} papers...")

        # Generate embeddings
        embeddings = self.embed_papers(papers)

        # Cluster papers
        cluster_info = self.cluster_papers(
            papers,
            n_clusters=n_clusters,
            embeddings=embeddings,
        )

        # Add cluster info to papers
        for paper in papers:
            paper_id = self._get_paper_id(paper)
            if paper_id in cluster_info["cluster_assignments"]:
                cluster_id = cluster_info["cluster_assignments"][paper_id]
                paper["semantic_cluster_id"] = cluster_id
                paper["semantic_cluster_label"] = cluster_info["cluster_labels"].get(cluster_id, "")
                paper["cluster_distance"] = cluster_info["paper_distances"].get(paper_id, 0)

        # Find embedding gaps
        gap_info = self.find_embedding_gaps(papers, embeddings=embeddings)

        # Detect emerging clusters
        emerging_info = self.detect_emerging_clusters(papers, embeddings=embeddings)

        # Generate visualization if output_dir provided
        visualization_info = None
        if output_dir:
            viz_path = os.path.join(output_dir, "embedding_landscape.json")
            visualization_info = self.visualize_landscape(
                papers,
                output_path=viz_path,
                embeddings=embeddings,
                cluster_info=cluster_info,
            )

        # Compile corpus analysis
        corpus_analysis = {
            "clustering": {
                "n_clusters": cluster_info["n_clusters"],
                "silhouette_score": cluster_info["silhouette_score"],
                "cluster_sizes": cluster_info["cluster_sizes"],
                "cluster_labels": cluster_info["cluster_labels"],
            },
            "embedding_gaps": {
                "overall_coverage": gap_info["overall_coverage"],
                "sparse_regions": gap_info["sparse_regions"][:10],
                "boundary_papers": gap_info["boundary_papers"][:10],
                "potential_gap_topics": gap_info.get("potential_gap_topics", []),
            },
            "emerging_clusters": {
                "emerging": emerging_info["emerging_clusters"][:10],
                "declining": emerging_info["declining_clusters"][:10],
                "early_period": emerging_info.get("early_period", ""),
                "recent_period": emerging_info.get("recent_period", ""),
            },
            "embedding_stats": {
                "papers_embedded": len(embeddings),
                "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else 384,
                "model_name": self.model_name,
            },
        }

        if visualization_info:
            corpus_analysis["visualization"] = {
                "output_path": output_dir,
                "total_points": visualization_info.get("total_papers", 0),
                "bounds": visualization_info.get("bounds", {}),
            }

        return papers, corpus_analysis

    def clear_cache(self, memory_only: bool = False):
        """Clear the embedding cache.

        Args:
            memory_only: If True, only clear memory cache, keep disk cache.
        """
        self.embeddings_cache.clear()

        if not memory_only and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            print(f"    Cleared embedding cache at {self.cache_dir}")
