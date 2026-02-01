"""
SOTA Benchmark Tracking Module
==============================
Track state-of-the-art benchmarks and results from Papers With Code:
- Fetch benchmarks for papers
- Track SOTA progression over time
- Identify solved/active/underserved benchmarks
- Compare methods on benchmarks
- Find benchmark gaps
"""

import json
import time
import urllib.request
import urllib.parse
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import os


# Papers With Code API endpoints
PWC_API_BASE = "https://paperswithcode.com/api/v1"
PWC_TASKS_API = f"{PWC_API_BASE}/tasks/"
PWC_DATASETS_API = f"{PWC_API_BASE}/datasets/"
PWC_SOTA_API = f"{PWC_API_BASE}/sota/"
PWC_EVALUATIONS_API = f"{PWC_API_BASE}/evaluations/"
PWC_PAPERS_API = f"{PWC_API_BASE}/papers/"


class BenchmarkTracker:
    """
    Track SOTA benchmarks and results from Papers With Code.

    This class provides methods to:
    - Fetch benchmark results for arXiv papers
    - Track SOTA progression over time for tasks
    - Identify solved, active, and underserved benchmarks
    - Compare methods across benchmarks
    - Find gaps in benchmark coverage
    """

    def __init__(self, cache_dir: Optional[str] = None, rate_limit: float = 0.5):
        """
        Initialize the BenchmarkTracker.

        Args:
            cache_dir: Directory to cache API results. If None, no caching.
            rate_limit: Minimum seconds between API calls (default: 0.5)
        """
        self.rate_limit = rate_limit
        self.last_api_call = 0
        self.cache_dir = cache_dir
        self._cache = {}

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()

    def _load_cache(self):
        """Load cached data from disk if available."""
        if not self.cache_dir:
            return

        cache_file = os.path.join(self.cache_dir, "pwc_benchmark_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        if not self.cache_dir:
            return

        cache_file = os.path.join(self.cache_dir, "pwc_benchmark_cache.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception:
            pass

    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_api_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_api_call = time.time()

    def _api_request(self, url: str, cache_key: Optional[str] = None) -> Optional[Dict]:
        """
        Make an API request with rate limiting and optional caching.

        Args:
            url: The URL to request
            cache_key: Optional cache key for this request

        Returns:
            JSON response as dict, or None on error
        """
        # Check cache first
        if cache_key and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Cache for 24 hours
            if cached.get("timestamp", 0) > time.time() - 86400:
                return cached.get("data")

        self._wait_for_rate_limit()

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "research-benchmark-tracker/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Cache the result
            if cache_key:
                self._cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": data
                }

            return data

        except Exception as e:
            return None

    def _paginate_api(self, base_url: str, max_pages: int = 10) -> List[Dict]:
        """
        Fetch paginated results from PWC API.

        Args:
            base_url: Base URL for the endpoint
            max_pages: Maximum number of pages to fetch

        Returns:
            List of all results across pages
        """
        results = []
        page = 1

        while page <= max_pages:
            separator = "&" if "?" in base_url else "?"
            url = f"{base_url}{separator}page={page}"

            data = self._api_request(url)
            if not data:
                break

            page_results = data.get("results", [])
            if not page_results:
                break

            results.extend(page_results)

            # Check if there's a next page
            if not data.get("next"):
                break

            page += 1

        return results

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def fetch_benchmarks_for_papers(self, papers: List[Dict]) -> Dict:
        """
        Get benchmark results for papers in the dataset.

        Args:
            papers: List of paper dicts with 'arxiv_id' or 'arxiv_id_base' field

        Returns:
            Dict with benchmark data keyed by arxiv_id, plus summary statistics
        """
        print(f"    Fetching benchmarks for {len(papers)} papers...")

        paper_benchmarks = {}
        papers_with_benchmarks = 0
        total_benchmark_entries = 0

        for idx, paper in enumerate(papers):
            if idx > 0 and idx % 50 == 0:
                print(f"      Processed {idx}/{len(papers)} papers...")

            arxiv_id = paper.get("arxiv_id_base") or paper.get("arxiv_id", "").split("v")[0]
            if not arxiv_id:
                continue

            # First, get the paper's PWC ID
            cache_key = f"paper_{arxiv_id}"
            paper_url = f"{PWC_PAPERS_API}?arxiv_id={arxiv_id}"
            paper_data = self._api_request(paper_url, cache_key)

            if not paper_data or not paper_data.get("results"):
                continue

            pwc_paper = paper_data["results"][0]
            pwc_paper_id = pwc_paper.get("id")

            if not pwc_paper_id:
                continue

            # Get evaluations for this paper
            eval_cache_key = f"eval_{pwc_paper_id}"
            eval_url = f"{PWC_PAPERS_API}{pwc_paper_id}/results/"
            eval_data = self._api_request(eval_url, eval_cache_key)

            evaluations = []
            if eval_data:
                results = eval_data.get("results", []) if isinstance(eval_data, dict) else eval_data
                for result in results:
                    evaluations.append({
                        "task": result.get("task", ""),
                        "dataset": result.get("dataset", ""),
                        "metric": result.get("metric", ""),
                        "value": result.get("value"),
                        "rank": result.get("rank"),
                        "is_sota": result.get("rank") == 1 if result.get("rank") else False,
                    })

            if evaluations:
                papers_with_benchmarks += 1
                total_benchmark_entries += len(evaluations)
                paper_benchmarks[arxiv_id] = {
                    "pwc_id": pwc_paper_id,
                    "pwc_url": pwc_paper.get("url_abs", ""),
                    "evaluations": evaluations,
                    "tasks": list(set(e["task"] for e in evaluations if e["task"])),
                    "datasets": list(set(e["dataset"] for e in evaluations if e["dataset"])),
                    "has_sota": any(e.get("is_sota") for e in evaluations),
                }

        self._save_cache()

        # Compute summary
        all_tasks = Counter()
        all_datasets = Counter()
        sota_papers = []

        for arxiv_id, data in paper_benchmarks.items():
            for task in data.get("tasks", []):
                all_tasks[task] += 1
            for dataset in data.get("datasets", []):
                all_datasets[dataset] += 1
            if data.get("has_sota"):
                sota_papers.append(arxiv_id)

        return {
            "paper_benchmarks": paper_benchmarks,
            "summary": {
                "papers_analyzed": len(papers),
                "papers_with_benchmarks": papers_with_benchmarks,
                "coverage_rate": round(papers_with_benchmarks / len(papers) * 100, 1) if papers else 0,
                "total_benchmark_entries": total_benchmark_entries,
                "unique_tasks": len(all_tasks),
                "unique_datasets": len(all_datasets),
                "sota_papers_count": len(sota_papers),
            },
            "top_tasks": all_tasks.most_common(20),
            "top_datasets": all_datasets.most_common(20),
            "sota_paper_ids": sota_papers,
        }

    def track_sota_progression(self, task_name: str, dataset_name: Optional[str] = None,
                                max_results: int = 100) -> Dict:
        """
        Track SOTA progression over time for a task.

        Args:
            task_name: Name of the task (e.g., "Image Classification")
            dataset_name: Optional specific dataset (e.g., "ImageNet")
            max_results: Maximum number of results to fetch

        Returns:
            Dict with SOTA progression data
        """
        print(f"    Tracking SOTA for: {task_name}" + (f" on {dataset_name}" if dataset_name else ""))

        # Search for the task
        task_url = f"{PWC_TASKS_API}?q={urllib.parse.quote(task_name)}"
        task_data = self._api_request(task_url, f"task_search_{task_name}")

        if not task_data or not task_data.get("results"):
            return {"error": f"Task not found: {task_name}"}

        task_info = task_data["results"][0]
        task_id = task_info.get("id")

        # Get SOTA results for this task
        sota_results = []

        if dataset_name:
            # Get specific dataset
            dataset_url = f"{PWC_DATASETS_API}?q={urllib.parse.quote(dataset_name)}"
            dataset_data = self._api_request(dataset_url, f"dataset_search_{dataset_name}")

            if dataset_data and dataset_data.get("results"):
                dataset_id = dataset_data["results"][0].get("id")
                sota_url = f"{PWC_SOTA_API}?task={task_id}&dataset={dataset_id}"
                sota_data = self._paginate_api(sota_url, max_pages=5)
                sota_results = sota_data
        else:
            # Get all SOTA for task
            sota_url = f"{PWC_SOTA_API}?task={task_id}"
            sota_results = self._paginate_api(sota_url, max_pages=5)

        if not sota_results:
            return {
                "task": task_name,
                "dataset": dataset_name,
                "error": "No SOTA results found",
            }

        # Process results to build progression
        progression = []
        methods_by_metric = defaultdict(list)

        for result in sota_results:
            entry = {
                "paper_title": result.get("paper_title", ""),
                "paper_url": result.get("paper_url", ""),
                "method_name": result.get("method_name", ""),
                "metric_name": result.get("metric_name", ""),
                "metric_value": result.get("metric_value"),
                "dataset": result.get("dataset_name", ""),
                "evaluation_date": result.get("evaluation_date", ""),
                "rank": result.get("rank"),
            }
            progression.append(entry)

            if entry["metric_name"]:
                methods_by_metric[entry["metric_name"]].append(entry)

        # Sort by date if available, otherwise by value
        for metric, entries in methods_by_metric.items():
            entries.sort(key=lambda x: x.get("evaluation_date") or "", reverse=True)

        # Calculate improvement metrics
        sota_history = {}
        for metric, entries in methods_by_metric.items():
            if len(entries) >= 2:
                current_best = entries[0]
                oldest = entries[-1]

                current_value = current_best.get("metric_value")
                oldest_value = oldest.get("metric_value")

                if current_value is not None and oldest_value is not None:
                    try:
                        improvement = float(current_value) - float(oldest_value)
                        improvement_pct = (improvement / abs(float(oldest_value))) * 100 if oldest_value != 0 else 0
                    except (ValueError, TypeError):
                        improvement = None
                        improvement_pct = None
                else:
                    improvement = None
                    improvement_pct = None

                sota_history[metric] = {
                    "current_sota": {
                        "value": current_value,
                        "method": current_best.get("method_name"),
                        "paper": current_best.get("paper_title"),
                    },
                    "oldest_record": {
                        "value": oldest_value,
                        "method": oldest.get("method_name"),
                    },
                    "total_improvement": improvement,
                    "improvement_percent": round(improvement_pct, 2) if improvement_pct else None,
                    "submission_count": len(entries),
                }

        self._save_cache()

        return {
            "task": task_name,
            "task_id": task_id,
            "dataset": dataset_name,
            "total_results": len(sota_results),
            "metrics_tracked": list(methods_by_metric.keys()),
            "sota_by_metric": sota_history,
            "all_results": progression[:max_results],
        }

    def find_solved_benchmarks(self, min_submissions: int = 10,
                                plateau_threshold: float = 0.01) -> Dict:
        """
        Find benchmarks where progress has plateaued (potentially "solved").

        Args:
            min_submissions: Minimum submissions to consider
            plateau_threshold: Max relative improvement in recent submissions to consider plateaued

        Returns:
            Dict with potentially solved benchmarks
        """
        print("    Finding solved/plateaued benchmarks...")

        # Get popular SOTA benchmarks
        sota_data = self._paginate_api(PWC_SOTA_API, max_pages=20)

        if not sota_data:
            return {"error": "Could not fetch SOTA data"}

        # Group by task+dataset+metric
        benchmark_results = defaultdict(list)

        for result in sota_data:
            key = (
                result.get("task_name", ""),
                result.get("dataset_name", ""),
                result.get("metric_name", "")
            )
            if all(key):
                benchmark_results[key].append({
                    "value": result.get("metric_value"),
                    "date": result.get("evaluation_date", ""),
                    "method": result.get("method_name", ""),
                    "paper": result.get("paper_title", ""),
                })

        solved_benchmarks = []

        for (task, dataset, metric), results in benchmark_results.items():
            if len(results) < min_submissions:
                continue

            # Sort by date
            results.sort(key=lambda x: x.get("date") or "", reverse=True)

            # Get values
            values = []
            for r in results:
                try:
                    if r["value"] is not None:
                        values.append(float(r["value"]))
                except (ValueError, TypeError):
                    pass

            if len(values) < min_submissions:
                continue

            # Check for plateau: recent improvements are minimal
            recent_values = values[:5]
            older_values = values[5:15] if len(values) > 5 else []

            if not older_values:
                continue

            recent_best = max(recent_values) if recent_values else 0
            older_best = max(older_values) if older_values else 0

            if older_best == 0:
                continue

            relative_improvement = abs(recent_best - older_best) / abs(older_best)

            if relative_improvement < plateau_threshold:
                solved_benchmarks.append({
                    "task": task,
                    "dataset": dataset,
                    "metric": metric,
                    "submission_count": len(results),
                    "current_sota": recent_best,
                    "recent_improvement": round(relative_improvement * 100, 3),
                    "top_method": results[0].get("method"),
                    "plateau_confidence": "high" if relative_improvement < 0.005 else "medium",
                })

        # Sort by submission count (more submissions = more confident)
        solved_benchmarks.sort(key=lambda x: x["submission_count"], reverse=True)

        self._save_cache()

        return {
            "potentially_solved": solved_benchmarks[:50],
            "total_analyzed": len(benchmark_results),
            "solved_count": len(solved_benchmarks),
            "analysis_note": f"Benchmarks with <{plateau_threshold*100}% improvement in recent submissions",
        }

    def find_active_benchmarks(self, recent_days: int = 180,
                                min_recent_submissions: int = 5) -> Dict:
        """
        Find benchmarks with recent improvements and active development.

        Args:
            recent_days: Consider submissions within this many days as "recent"
            min_recent_submissions: Minimum recent submissions to consider active

        Returns:
            Dict with active benchmarks
        """
        print("    Finding active benchmarks...")

        cutoff_date = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")

        # Get recent SOTA data
        sota_data = self._paginate_api(PWC_SOTA_API, max_pages=30)

        if not sota_data:
            return {"error": "Could not fetch SOTA data"}

        # Group by task+dataset+metric
        benchmark_activity = defaultdict(lambda: {
            "recent": [],
            "older": [],
            "all_methods": set(),
        })

        for result in sota_data:
            key = (
                result.get("task_name", ""),
                result.get("dataset_name", ""),
                result.get("metric_name", "")
            )
            if not all(key):
                continue

            entry = {
                "value": result.get("metric_value"),
                "date": result.get("evaluation_date", ""),
                "method": result.get("method_name", ""),
                "paper": result.get("paper_title", ""),
            }

            eval_date = entry.get("date", "")
            if eval_date >= cutoff_date:
                benchmark_activity[key]["recent"].append(entry)
            else:
                benchmark_activity[key]["older"].append(entry)

            if entry.get("method"):
                benchmark_activity[key]["all_methods"].add(entry["method"])

        active_benchmarks = []

        for (task, dataset, metric), activity in benchmark_activity.items():
            recent_count = len(activity["recent"])

            if recent_count < min_recent_submissions:
                continue

            # Calculate improvement velocity
            recent_values = []
            for r in activity["recent"]:
                try:
                    if r["value"] is not None:
                        recent_values.append(float(r["value"]))
                except (ValueError, TypeError):
                    pass

            older_values = []
            for r in activity["older"][:10]:
                try:
                    if r["value"] is not None:
                        older_values.append(float(r["value"]))
                except (ValueError, TypeError):
                    pass

            improvement = 0
            if recent_values and older_values:
                recent_best = max(recent_values)
                older_best = max(older_values)
                if older_best != 0:
                    improvement = (recent_best - older_best) / abs(older_best) * 100

            active_benchmarks.append({
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "recent_submissions": recent_count,
                "total_submissions": recent_count + len(activity["older"]),
                "unique_methods": len(activity["all_methods"]),
                "recent_improvement_pct": round(improvement, 2),
                "activity_level": "high" if recent_count >= 10 else "medium",
                "latest_method": activity["recent"][0].get("method") if activity["recent"] else None,
            })

        # Sort by recent activity
        active_benchmarks.sort(key=lambda x: x["recent_submissions"], reverse=True)

        self._save_cache()

        return {
            "active_benchmarks": active_benchmarks[:50],
            "total_analyzed": len(benchmark_activity),
            "active_count": len(active_benchmarks),
            "recent_window_days": recent_days,
        }

    def find_underserved_benchmarks(self, max_submissions: int = 5,
                                     min_importance: int = 3) -> Dict:
        """
        Find tasks/benchmarks with few submissions (opportunities).

        Args:
            max_submissions: Consider benchmarks with at most this many submissions
            min_importance: Minimum "importance" score (based on task popularity)

        Returns:
            Dict with underserved benchmarks
        """
        print("    Finding underserved benchmarks...")

        # Get all tasks to understand importance
        tasks_data = self._paginate_api(PWC_TASKS_API, max_pages=10)

        task_importance = {}
        for task in tasks_data:
            task_id = task.get("id", "")
            task_name = task.get("name", "")
            # Importance based on paper count
            paper_count = task.get("paper_count", 0)
            if paper_count >= min_importance:
                task_importance[task_name] = paper_count

        # Get SOTA data
        sota_data = self._paginate_api(PWC_SOTA_API, max_pages=20)

        # Count submissions per benchmark
        benchmark_counts = Counter()
        benchmark_details = {}

        for result in sota_data:
            key = (
                result.get("task_name", ""),
                result.get("dataset_name", ""),
                result.get("metric_name", "")
            )
            if all(key):
                benchmark_counts[key] += 1
                if key not in benchmark_details:
                    benchmark_details[key] = {
                        "best_value": result.get("metric_value"),
                        "best_method": result.get("method_name"),
                    }

        underserved = []

        for (task, dataset, metric), count in benchmark_counts.items():
            if count > max_submissions:
                continue

            importance = task_importance.get(task, 0)

            if importance < min_importance:
                continue

            underserved.append({
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "submission_count": count,
                "task_paper_count": importance,
                "opportunity_score": round(importance / max(count, 1), 1),
                "current_best": benchmark_details.get((task, dataset, metric), {}),
            })

        # Sort by opportunity score
        underserved.sort(key=lambda x: x["opportunity_score"], reverse=True)

        self._save_cache()

        return {
            "underserved_benchmarks": underserved[:50],
            "total_analyzed": len(benchmark_counts),
            "underserved_count": len(underserved),
            "criteria": {
                "max_submissions": max_submissions,
                "min_task_papers": min_importance,
            },
        }

    def identify_sota_papers(self, papers: List[Dict]) -> Dict:
        """
        Identify papers that held SOTA at some point.

        Args:
            papers: List of paper dicts with arxiv_id

        Returns:
            Dict with SOTA paper information
        """
        print(f"    Identifying SOTA papers from {len(papers)} papers...")

        # First get benchmark data for all papers
        benchmark_data = self.fetch_benchmarks_for_papers(papers)
        paper_benchmarks = benchmark_data.get("paper_benchmarks", {})

        sota_papers = []
        current_sota_papers = []
        former_sota_papers = []

        for arxiv_id, data in paper_benchmarks.items():
            evaluations = data.get("evaluations", [])
            sota_evals = [e for e in evaluations if e.get("is_sota") or e.get("rank") == 1]

            if sota_evals:
                paper_info = {
                    "arxiv_id": arxiv_id,
                    "pwc_url": data.get("pwc_url", ""),
                    "sota_benchmarks": [
                        {
                            "task": e["task"],
                            "dataset": e["dataset"],
                            "metric": e["metric"],
                            "value": e["value"],
                        }
                        for e in sota_evals
                    ],
                    "total_sota_count": len(sota_evals),
                    "tasks": list(set(e["task"] for e in sota_evals)),
                }

                sota_papers.append(paper_info)

                # Check if still current SOTA
                still_sota = any(e.get("rank") == 1 for e in sota_evals)
                if still_sota:
                    current_sota_papers.append(paper_info)
                else:
                    former_sota_papers.append(paper_info)

        # Sort by SOTA count
        sota_papers.sort(key=lambda x: x["total_sota_count"], reverse=True)
        current_sota_papers.sort(key=lambda x: x["total_sota_count"], reverse=True)

        return {
            "all_sota_papers": sota_papers,
            "current_sota_papers": current_sota_papers,
            "former_sota_papers": former_sota_papers,
            "summary": {
                "total_papers_analyzed": len(papers),
                "papers_with_benchmarks": len(paper_benchmarks),
                "papers_ever_sota": len(sota_papers),
                "papers_currently_sota": len(current_sota_papers),
            },
        }

    def compare_methods_on_benchmark(self, task_name: str, dataset_name: str,
                                      methods: Optional[List[str]] = None) -> Dict:
        """
        Compare specific methods on a benchmark.

        Args:
            task_name: Name of the task
            dataset_name: Name of the dataset
            methods: Optional list of method names to compare. If None, compare top methods.

        Returns:
            Dict with method comparison
        """
        print(f"    Comparing methods on {task_name} / {dataset_name}...")

        # Get task ID
        task_url = f"{PWC_TASKS_API}?q={urllib.parse.quote(task_name)}"
        task_data = self._api_request(task_url, f"task_search_{task_name}")

        if not task_data or not task_data.get("results"):
            return {"error": f"Task not found: {task_name}"}

        task_id = task_data["results"][0].get("id")

        # Get dataset ID
        dataset_url = f"{PWC_DATASETS_API}?q={urllib.parse.quote(dataset_name)}"
        dataset_data = self._api_request(dataset_url, f"dataset_search_{dataset_name}")

        if not dataset_data or not dataset_data.get("results"):
            return {"error": f"Dataset not found: {dataset_name}"}

        dataset_id = dataset_data["results"][0].get("id")

        # Get evaluations
        eval_url = f"{PWC_EVALUATIONS_API}?task={task_id}&dataset={dataset_id}"
        eval_data = self._paginate_api(eval_url, max_pages=10)

        if not eval_data:
            return {
                "task": task_name,
                "dataset": dataset_name,
                "error": "No evaluation data found",
            }

        # Group by method
        method_results = defaultdict(lambda: defaultdict(list))

        for result in eval_data:
            method = result.get("method_name", "Unknown")
            metric = result.get("metric_name", "")
            value = result.get("metric_value")

            if methods and method not in methods:
                continue

            if metric and value is not None:
                method_results[method][metric].append({
                    "value": value,
                    "paper": result.get("paper_title", ""),
                    "date": result.get("evaluation_date", ""),
                })

        # Build comparison table
        all_metrics = set()
        for method_data in method_results.values():
            all_metrics.update(method_data.keys())

        comparison = []

        for method, metrics_data in method_results.items():
            entry = {
                "method": method,
                "metrics": {},
                "best_paper": None,
            }

            for metric in all_metrics:
                if metric in metrics_data:
                    values = metrics_data[metric]
                    try:
                        best_idx = 0
                        best_val = float(values[0]["value"])
                        for i, v in enumerate(values[1:], 1):
                            val = float(v["value"])
                            if val > best_val:
                                best_val = val
                                best_idx = i
                        entry["metrics"][metric] = {
                            "best_value": best_val,
                            "paper": values[best_idx]["paper"],
                            "submission_count": len(values),
                        }
                        if entry["best_paper"] is None:
                            entry["best_paper"] = values[best_idx]["paper"]
                    except (ValueError, TypeError):
                        entry["metrics"][metric] = {"best_value": values[0]["value"]}

            comparison.append(entry)

        # Rank methods by each metric
        rankings = {}
        for metric in all_metrics:
            ranked = []
            for entry in comparison:
                if metric in entry["metrics"]:
                    try:
                        value = float(entry["metrics"][metric]["best_value"])
                        ranked.append((entry["method"], value))
                    except (ValueError, TypeError):
                        pass
            ranked.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [{"rank": i+1, "method": m, "value": v} for i, (m, v) in enumerate(ranked)]

        self._save_cache()

        return {
            "task": task_name,
            "dataset": dataset_name,
            "methods_compared": len(comparison),
            "metrics": list(all_metrics),
            "method_comparison": comparison,
            "rankings_by_metric": rankings,
        }

    def find_benchmark_gaps(self) -> Dict:
        """
        Find tasks/areas without good benchmarks.

        Returns:
            Dict with benchmark gap analysis
        """
        print("    Analyzing benchmark gaps...")

        # Get all tasks
        tasks_data = self._paginate_api(PWC_TASKS_API, max_pages=20)

        if not tasks_data:
            return {"error": "Could not fetch tasks"}

        # Get SOTA coverage
        sota_data = self._paginate_api(PWC_SOTA_API, max_pages=20)

        # Which tasks have SOTA benchmarks?
        tasks_with_sota = set()
        for result in sota_data:
            task_name = result.get("task_name", "")
            if task_name:
                tasks_with_sota.add(task_name)

        # Analyze gaps
        no_benchmark_tasks = []
        low_benchmark_tasks = []

        task_sota_counts = Counter()
        for result in sota_data:
            task_name = result.get("task_name", "")
            if task_name:
                task_sota_counts[task_name] += 1

        for task in tasks_data:
            task_name = task.get("name", "")
            paper_count = task.get("paper_count", 0)
            area = task.get("area", {})
            area_name = area.get("name", "") if isinstance(area, dict) else ""

            if task_name not in tasks_with_sota:
                if paper_count >= 5:  # Has papers but no benchmarks
                    no_benchmark_tasks.append({
                        "task": task_name,
                        "paper_count": paper_count,
                        "area": area_name,
                        "gap_type": "no_benchmark",
                        "priority": "high" if paper_count >= 20 else "medium",
                    })
            elif task_sota_counts.get(task_name, 0) < 3:
                if paper_count >= 10:  # Popular task but few benchmark results
                    low_benchmark_tasks.append({
                        "task": task_name,
                        "paper_count": paper_count,
                        "sota_count": task_sota_counts.get(task_name, 0),
                        "area": area_name,
                        "gap_type": "low_coverage",
                    })

        # Sort by paper count
        no_benchmark_tasks.sort(key=lambda x: x["paper_count"], reverse=True)
        low_benchmark_tasks.sort(key=lambda x: x["paper_count"], reverse=True)

        # Identify areas with gaps
        area_gaps = defaultdict(lambda: {"no_benchmark": 0, "low_benchmark": 0, "tasks": []})

        for task in no_benchmark_tasks:
            area = task.get("area", "Unknown")
            area_gaps[area]["no_benchmark"] += 1
            area_gaps[area]["tasks"].append(task["task"])

        for task in low_benchmark_tasks:
            area = task.get("area", "Unknown")
            area_gaps[area]["low_benchmark"] += 1

        self._save_cache()

        return {
            "tasks_without_benchmarks": no_benchmark_tasks[:30],
            "tasks_with_low_coverage": low_benchmark_tasks[:30],
            "area_analysis": dict(area_gaps),
            "summary": {
                "total_tasks_analyzed": len(tasks_data),
                "tasks_with_sota": len(tasks_with_sota),
                "tasks_without_benchmarks": len(no_benchmark_tasks),
                "tasks_with_low_coverage": len(low_benchmark_tasks),
            },
        }

    # =========================================================================
    # METRICS TRACKING
    # =========================================================================

    def get_benchmark_metrics(self, task_name: Optional[str] = None) -> Dict:
        """
        Get comprehensive metrics for benchmarks.

        Args:
            task_name: Optional task to filter by

        Returns:
            Dict with benchmark metrics over time
        """
        print("    Computing benchmark metrics...")

        # Build query
        base_url = PWC_SOTA_API
        if task_name:
            task_url = f"{PWC_TASKS_API}?q={urllib.parse.quote(task_name)}"
            task_data = self._api_request(task_url)
            if task_data and task_data.get("results"):
                task_id = task_data["results"][0].get("id")
                base_url = f"{PWC_SOTA_API}?task={task_id}"

        sota_data = self._paginate_api(base_url, max_pages=30)

        if not sota_data:
            return {"error": "No SOTA data available"}

        # Group by benchmark and analyze
        benchmark_metrics = defaultdict(lambda: {
            "submissions": [],
            "methods": set(),
            "score_progression": [],
            "dates": [],
        })

        for result in sota_data:
            key = (
                result.get("task_name", ""),
                result.get("dataset_name", ""),
                result.get("metric_name", "")
            )
            if not all(key):
                continue

            value = result.get("metric_value")
            date = result.get("evaluation_date", "")
            method = result.get("method_name", "")

            benchmark_metrics[key]["submissions"].append(result)
            benchmark_metrics[key]["methods"].add(method)

            if value is not None and date:
                try:
                    benchmark_metrics[key]["score_progression"].append({
                        "date": date,
                        "value": float(value),
                        "method": method,
                    })
                except (ValueError, TypeError):
                    pass

            if date:
                benchmark_metrics[key]["dates"].append(date)

        # Compute per-benchmark metrics
        results = []

        for (task, dataset, metric), data in benchmark_metrics.items():
            submissions = data["submissions"]
            progression = sorted(data["score_progression"], key=lambda x: x["date"])

            # Time between improvements
            improvement_intervals = []
            if len(progression) >= 2:
                best_so_far = progression[0]["value"]
                last_improvement_date = progression[0]["date"]

                for entry in progression[1:]:
                    if entry["value"] > best_so_far:
                        try:
                            d1 = datetime.strptime(last_improvement_date[:10], "%Y-%m-%d")
                            d2 = datetime.strptime(entry["date"][:10], "%Y-%m-%d")
                            interval = (d2 - d1).days
                            if interval > 0:
                                improvement_intervals.append(interval)
                            last_improvement_date = entry["date"]
                            best_so_far = entry["value"]
                        except ValueError:
                            pass

            # Dominant methods
            method_counts = Counter(s.get("method_name") for s in submissions if s.get("method_name"))

            results.append({
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "total_submissions": len(submissions),
                "unique_methods": len(data["methods"]),
                "best_score": max(p["value"] for p in progression) if progression else None,
                "score_range": {
                    "min": min(p["value"] for p in progression) if progression else None,
                    "max": max(p["value"] for p in progression) if progression else None,
                } if progression else None,
                "avg_days_between_improvements": (
                    round(sum(improvement_intervals) / len(improvement_intervals), 1)
                    if improvement_intervals else None
                ),
                "dominant_methods": method_counts.most_common(5),
                "date_range": {
                    "first": min(data["dates"]) if data["dates"] else None,
                    "last": max(data["dates"]) if data["dates"] else None,
                },
            })

        # Sort by activity
        results.sort(key=lambda x: x["total_submissions"], reverse=True)

        self._save_cache()

        return {
            "benchmark_metrics": results[:100],
            "total_benchmarks": len(benchmark_metrics),
            "summary": {
                "avg_submissions_per_benchmark": round(
                    sum(r["total_submissions"] for r in results) / len(results), 1
                ) if results else 0,
                "avg_methods_per_benchmark": round(
                    sum(r["unique_methods"] for r in results) / len(results), 1
                ) if results else 0,
            },
        }

    # =========================================================================
    # FULL ANALYSIS
    # =========================================================================

    def analyze_benchmarks(self, papers: Optional[List[Dict]] = None) -> Dict:
        """
        Run comprehensive benchmark analysis.

        Args:
            papers: Optional list of papers to analyze

        Returns:
            Dict with full benchmark analysis
        """
        print(f"    Running comprehensive benchmark analysis...")

        results = {
            "active_benchmarks": self.find_active_benchmarks(),
            "solved_benchmarks": self.find_solved_benchmarks(),
            "underserved_benchmarks": self.find_underserved_benchmarks(),
            "benchmark_gaps": self.find_benchmark_gaps(),
            "metrics": self.get_benchmark_metrics(),
        }

        if papers:
            results["paper_benchmarks"] = self.fetch_benchmarks_for_papers(papers)
            results["sota_papers"] = self.identify_sota_papers(papers)

        # Save final cache
        self._save_cache()

        return results
