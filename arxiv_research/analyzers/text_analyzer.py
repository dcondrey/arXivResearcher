"""
Text Analysis Module
====================
Extract insights from paper titles and abstracts:
- Keyword extraction (TF-IDF, statistical)
- Method/technique detection
- Dataset mentions
- Contribution type classification
- Novelty indicators
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
import json


# Common ML/AI methods and techniques
METHODS_PATTERNS = {
    # Neural architectures
    "transformer": r"\btransformers?\b",
    "attention": r"\b(?:self-)?attention\b",
    "cnn": r"\b(?:cnn|convolutional neural network)s?\b",
    "rnn": r"\b(?:rnn|recurrent neural network|lstm|gru)s?\b",
    "gan": r"\b(?:gan|generative adversarial)s?\b",
    "vae": r"\b(?:vae|variational auto-?encoder)s?\b",
    "diffusion": r"\bdiffusion (?:model|process)s?\b",
    "autoencoder": r"\bauto-?encoders?\b",
    "mlp": r"\b(?:mlp|multi-?layer perceptron)s?\b",
    "graph_neural_network": r"\b(?:gnn|graph neural network|gcn|graph convolutional)s?\b",
    "neural_network": r"\bneural networks?\b",

    # Learning paradigms
    "reinforcement_learning": r"\breinforcement learning\b",
    "supervised_learning": r"\bsupervised learning\b",
    "unsupervised_learning": r"\bunsupervised learning\b",
    "self_supervised": r"\bself-?supervised\b",
    "contrastive_learning": r"\bcontrastive learning\b",
    "meta_learning": r"\bmeta-?learning\b",
    "transfer_learning": r"\btransfer learning\b",
    "federated_learning": r"\bfederated learning\b",
    "active_learning": r"\bactive learning\b",
    "online_learning": r"\bonline learning\b",
    "few_shot": r"\bfew-?shot\b",
    "zero_shot": r"\bzero-?shot\b",
    "multi_task": r"\bmulti-?task\b",
    "curriculum_learning": r"\bcurriculum learning\b",

    # Specific techniques
    "fine_tuning": r"\bfine-?tun(?:e|ing)\b",
    "pre_training": r"\bpre-?train(?:ed|ing)?\b",
    "prompting": r"\bprompt(?:ing|s)?\b",
    "in_context_learning": r"\bin-?context learning\b",
    "chain_of_thought": r"\bchain-?of-?thought\b",
    "rlhf": r"\b(?:rlhf|reinforcement learning (?:from|with) human feedback)\b",
    "distillation": r"\b(?:knowledge )?distillation\b",
    "pruning": r"\bpruning\b",
    "quantization": r"\bquantization\b",
    "dropout": r"\bdropout\b",
    "batch_normalization": r"\bbatch normali[sz]ation\b",
    "layer_normalization": r"\blayer normali[sz]ation\b",
    "regularization": r"\bregulari[sz]ation\b",
    "data_augmentation": r"\bdata augmentation\b",
    "ensemble": r"\bensemble\b",
    "bagging": r"\bbagging\b",
    "boosting": r"\bboosting\b",
    "cross_validation": r"\bcross-?validation\b",

    # Optimization
    "adam": r"\badam\b",
    "sgd": r"\b(?:sgd|stochastic gradient descent)\b",
    "gradient_descent": r"\bgradient descent\b",
    "backpropagation": r"\bbackpropagation\b",

    # NLP specific
    "bert": r"\bbert\b",
    "gpt": r"\bgpt(?:-\d)?\b",
    "llm": r"\b(?:llm|large language model)s?\b",
    "word_embeddings": r"\bword (?:embeddings?|vectors?)\b",
    "tokenization": r"\btokeni[sz]ation\b",
    "named_entity_recognition": r"\b(?:ner|named entity recognition)\b",
    "sentiment_analysis": r"\bsentiment analysis\b",
    "machine_translation": r"\bmachine translation\b",
    "question_answering": r"\bquestion answering\b",
    "text_generation": r"\btext generation\b",
    "summarization": r"\bsummari[sz]ation\b",

    # Vision specific
    "object_detection": r"\bobject detection\b",
    "image_classification": r"\bimage classification\b",
    "semantic_segmentation": r"\bsemantic segmentation\b",
    "instance_segmentation": r"\binstance segmentation\b",
    "pose_estimation": r"\bpose estimation\b",
    "optical_flow": r"\boptical flow\b",
    "depth_estimation": r"\bdepth estimation\b",
    "image_generation": r"\bimage generation\b",
    "super_resolution": r"\bsuper-?resolution\b",
    "style_transfer": r"\bstyle transfer\b",

    # Other ML
    "clustering": r"\bclustering\b",
    "dimensionality_reduction": r"\bdimensionality reduction\b",
    "anomaly_detection": r"\banomaly detection\b",
    "recommendation": r"\brecommend(?:ation|er)s?\b",
    "time_series": r"\btime series\b",
    "forecasting": r"\bforecast(?:ing)?\b",

    # Classical ML
    "random_forest": r"\brandom forest\b",
    "decision_tree": r"\bdecision trees?\b",
    "svm": r"\b(?:svm|support vector machine)s?\b",
    "logistic_regression": r"\blogistic regression\b",
    "linear_regression": r"\blinear regression\b",
    "naive_bayes": r"\bnaive bayes\b",
    "knn": r"\b(?:knn|k-?nearest neighbor)s?\b",
    "pca": r"\b(?:pca|principal component analysis)\b",
    "bayesian": r"\bbayesian\b",
    "monte_carlo": r"\bmonte carlo\b",
    "markov": r"\bmarkov\b",
}

# Common datasets
DATASET_PATTERNS = {
    # Vision
    "imagenet": r"\bimagenet\b",
    "cifar": r"\bcifar-?(?:10|100)?\b",
    "mnist": r"\bmnist\b",
    "coco": r"\b(?:ms-?)?coco\b",
    "pascal_voc": r"\bpascal voc\b",
    "cityscapes": r"\bcityscapes\b",
    "ade20k": r"\bade20k\b",
    "celeba": r"\bceleb-?a\b",
    "lfw": r"\blfw\b",
    "kinetics": r"\bkinetics(?:-\d+)?\b",
    "youtube": r"\byoutube-?\d*[mk]?\b",

    # NLP
    "glue": r"\bglue\b",
    "superglue": r"\bsuperglue\b",
    "squad": r"\bsquad(?:\s*[12]\.?\d*)?\b",
    "wikitext": r"\bwikitext(?:-\d+)?\b",
    "wikipedia": r"\bwikipedia\b",
    "common_crawl": r"\bcommon crawl\b",
    "openwebtext": r"\bopenwebtext\b",
    "bookcorpus": r"\bbookcorpus\b",
    "pile": r"\bthe pile\b",
    "c4": r"\bc4\b",
    "lambada": r"\blambada\b",
    "hellaswag": r"\bhellaswag\b",
    "mmlu": r"\bmmlu\b",
    "humaneval": r"\bhumaneval\b",
    "gsm8k": r"\bgsm-?8k\b",
    "math": r"\bmath\b",

    # Multi-modal
    "laion": r"\blaion(?:-\d+[bmk])?\b",
    "conceptual_captions": r"\bconceptual captions\b",
    "visual_genome": r"\bvisual genome\b",
    "flickr": r"\bflickr(?:\d+k)?\b",

    # Audio
    "librispeech": r"\blibrispeech\b",
    "audioset": r"\baudioset\b",
    "voxceleb": r"\bvoxceleb\b",

    # Graphs
    "ogb": r"\bogb\b",
    "planetoid": r"\bplanetoid\b",
    "cora": r"\bcora\b",
    "citeseer": r"\bciteseer\b",
    "pubmed": r"\bpubmed\b",

    # RL
    "atari": r"\batari\b",
    "mujoco": r"\bmujoco\b",
    "openai_gym": r"\b(?:openai )?gym\b",
    "dmc": r"\bdm control\b",
}

# Contribution type indicators
CONTRIBUTION_PATTERNS = {
    "new_method": [
        r"\bwe propose\b", r"\bwe introduce\b", r"\bwe present\b",
        r"\bnovel (?:method|approach|algorithm|framework|architecture|model|technique)\b",
        r"\bnew (?:method|approach|algorithm|framework|architecture|model|technique)\b",
    ],
    "new_dataset": [
        r"\bwe (?:introduce|release|present|create|construct|build) (?:a )?(?:new )?dataset\b",
        r"\bnew benchmark\b", r"\bnovel dataset\b",
        r"\bdataset (?:consisting|containing|comprising)\b",
    ],
    "empirical_study": [
        r"\bempirical (?:study|analysis|evaluation|investigation)\b",
        r"\bwe (?:study|analyze|investigate|examine|evaluate)\b",
        r"\bcomparative (?:study|analysis)\b",
        r"\bablation stud(?:y|ies)\b",
    ],
    "theoretical": [
        r"\btheoretical (?:analysis|framework|foundation|guarantee|bound)\b",
        r"\bwe prove\b", r"\bwe show that\b", r"\btheorem\b",
        r"\bconvergence (?:analysis|guarantee|proof)\b",
    ],
    "survey": [
        r"\bsurvey\b", r"\breview\b", r"\boverview\b", r"\btutorial\b",
        r"\bcomprehensive (?:study|analysis|review)\b",
        r"\bstate-?of-?the-?art\b.*\breview\b",
    ],
    "application": [
        r"\bapplication (?:of|to)\b", r"\bapplied to\b",
        r"\breal-?world\b", r"\bpractical\b", r"\bcase study\b",
        r"\bdeployment\b", r"\bproduction\b",
    ],
    "improvement": [
        r"\bimprove(?:s|d|ment)?\b", r"\boutperform\b", r"\bstate-?of-?the-?art\b",
        r"\bachieve(?:s|d)? (?:new )?(?:state-?of-?the-?art|sota)\b",
        r"\bbeat(?:s|ing)?\b", r"\bsurpass\b",
    ],
    "analysis": [
        r"\bunderstanding\b", r"\binterpretab(?:le|ility)\b", r"\bexplainab(?:le|ility)\b",
        r"\bvisuali[sz](?:e|ation)\b", r"\binsight\b",
    ],
    "efficiency": [
        r"\befficient\b", r"\bfast(?:er)?\b", r"\blightweight\b",
        r"\bscalable\b", r"\bspeed-?up\b", r"\baccelerat\b",
        r"\breduc(?:e|ing|tion) (?:computation|memory|cost)\b",
    ],
    "robustness": [
        r"\brobust(?:ness)?\b", r"\badversarial\b", r"\bout-?of-?distribution\b",
        r"\bgenerali[sz](?:e|ation)\b", r"\btransfer(?:ability)?\b",
    ],
}

# Novelty indicators
NOVELTY_PHRASES = [
    r"\bfirst\b.*\bto\b", r"\bfirst time\b", r"\bfor the first time\b",
    r"\bnovel\b", r"\bnew\b", r"\bunprecedented\b",
    r"\bstate-?of-?the-?art\b", r"\bsota\b",
    r"\boutperform(?:s|ing)?\b", r"\bsurpass(?:es|ing)?\b",
    r"\bsignificant(?:ly)? (?:improve|better|outperform)\b",
]


class TextAnalyzer:
    """Analyze paper text to extract insights."""

    def __init__(self):
        self.stopwords = self._load_stopwords()
        self.idf_cache = {}

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords list."""
        # Common English stopwords + academic filler words
        return {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "they", "them", "their",
            "we", "our", "us", "i", "my", "you", "your", "he", "she", "him", "her",
            "what", "which", "who", "whom", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there",
            "into", "over", "after", "before", "between", "under", "again",
            "further", "then", "once", "during", "while", "about", "against",
            "above", "below", "up", "down", "out", "off", "through", "because",
            # Academic filler
            "paper", "propose", "proposed", "show", "shows", "shown", "study",
            "result", "results", "method", "methods", "approach", "approaches",
            "work", "works", "based", "using", "use", "used", "new", "novel",
            "present", "presents", "presented", "demonstrate", "demonstrated",
            "achieve", "achieved", "model", "models", "data", "dataset",
            "experiment", "experiments", "experimental", "performance",
            "evaluate", "evaluated", "evaluation", "task", "tasks",
            "problem", "problems", "solution", "solutions", "existing",
            "previous", "recent", "recently", "however", "although", "thus",
            "therefore", "moreover", "furthermore", "additionally", "specifically",
        }

    def analyze_paper(self, paper: Dict) -> Dict:
        """Run all analyses on a single paper."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {abstract}".lower()

        # Extract various features
        methods = self.extract_methods(text)
        datasets = self.extract_datasets(text)
        contribution_types = self.classify_contribution(text)
        novelty_score = self.compute_novelty_score(text)
        keywords = self.extract_keywords_single(title, abstract)

        return {
            "methods_detected": methods,
            "methods_count": len(methods),
            "datasets_mentioned": datasets,
            "datasets_count": len(datasets),
            "contribution_types": contribution_types,
            "primary_contribution": contribution_types[0] if contribution_types else "",
            "novelty_score": novelty_score,
            "keywords": keywords,
            "has_theoretical_contribution": "theoretical" in contribution_types,
            "has_empirical_contribution": "empirical_study" in contribution_types,
            "is_survey_paper": "survey" in contribution_types,
            "is_application_paper": "application" in contribution_types,
            "claims_sota": bool(re.search(r"\bstate-?of-?the-?art|sota\b", text)),
            "has_code_mention": bool(re.search(r"\bcode\b.*\bavailable\b|\bgithub\b|\bopen-?source\b", text)),
        }

    def extract_methods(self, text: str) -> List[str]:
        """Extract mentioned methods/techniques."""
        found = []
        for method, pattern in METHODS_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(method)
        return found

    def extract_datasets(self, text: str) -> List[str]:
        """Extract mentioned datasets."""
        found = []
        for dataset, pattern in DATASET_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(dataset)
        return found

    def classify_contribution(self, text: str) -> List[str]:
        """Classify the type of contribution."""
        types = []
        for ctype, patterns in CONTRIBUTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if ctype not in types:
                        types.append(ctype)
                    break
        return types

    def compute_novelty_score(self, text: str) -> float:
        """Compute a novelty indicator score (0-1)."""
        matches = 0
        for pattern in NOVELTY_PHRASES:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        # Normalize to 0-1 range
        return min(matches / 5.0, 1.0)

    def extract_keywords_single(self, title: str, abstract: str, top_n: int = 10) -> List[str]:
        """Extract keywords from a single paper using statistical methods."""
        text = f"{title} {title} {title} {abstract}"  # Weight title higher

        # Tokenize and clean
        words = re.findall(r'\b[a-z][a-z\-]+[a-z]\b', text.lower())
        words = [w for w in words if w not in self.stopwords and len(w) > 2]

        # Get bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

        # Count frequencies
        word_freq = Counter(words)
        bigram_freq = Counter(bigrams)

        # Score by frequency * position (earlier = better)
        scored = []
        for word, freq in word_freq.most_common(50):
            first_pos = text.lower().find(word)
            position_score = 1.0 / (1 + first_pos / 100)  # Earlier is better
            scored.append((word, freq * position_score * (1 if word in title.lower() else 0.5)))

        for bigram, freq in bigram_freq.most_common(30):
            if freq >= 2:
                first_pos = text.lower().find(bigram)
                position_score = 1.0 / (1 + first_pos / 100)
                scored.append((bigram, freq * position_score * 1.5))  # Boost bigrams

        # Sort and dedupe
        scored.sort(key=lambda x: x[1], reverse=True)
        seen_words = set()
        keywords = []
        for kw, score in scored:
            words_in_kw = set(kw.split())
            if not words_in_kw & seen_words:
                keywords.append(kw)
                seen_words.update(words_in_kw)
            if len(keywords) >= top_n:
                break

        return keywords

    def compute_corpus_tfidf(self, papers: List[Dict], top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Compute TF-IDF keywords for each paper in a corpus."""
        # Build document frequency
        doc_freq = Counter()
        doc_terms = []

        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            words = set(re.findall(r'\b[a-z][a-z\-]+[a-z]\b', text))
            words = {w for w in words if w not in self.stopwords and len(w) > 2}
            doc_terms.append(words)
            doc_freq.update(words)

        n_docs = len(papers)

        # Compute TF-IDF for each document
        results = {}
        for idx, paper in enumerate(papers):
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            words = re.findall(r'\b[a-z][a-z\-]+[a-z]\b', text)
            words = [w for w in words if w not in self.stopwords and len(w) > 2]

            tf = Counter(words)
            max_tf = max(tf.values()) if tf else 1

            tfidf_scores = []
            for word, count in tf.items():
                tf_score = count / max_tf
                idf_score = math.log(n_docs / (1 + doc_freq[word]))
                tfidf_scores.append((word, tf_score * idf_score))

            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            results[paper.get("arxiv_id", idx)] = tfidf_scores[:top_n]

        return results

    def extract_corpus_topics(self, papers: List[Dict], n_topics: int = 20) -> Dict:
        """Extract corpus-wide topics using co-occurrence."""
        # Build co-occurrence matrix
        word_cooccur = defaultdict(Counter)
        word_freq = Counter()

        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            words = list(set(re.findall(r'\b[a-z][a-z\-]+[a-z]\b', text)))
            words = [w for w in words if w not in self.stopwords and len(w) > 2]

            word_freq.update(words)

            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    word_cooccur[w1][w2] += 1
                    word_cooccur[w2][w1] += 1

        # Find topic clusters using frequent co-occurring words
        topics = []
        used_words = set()

        for word, freq in word_freq.most_common(200):
            if word in used_words:
                continue

            # Get words that frequently co-occur
            cooccur = word_cooccur[word]
            related = [w for w, c in cooccur.most_common(10)
                      if w not in used_words and c >= 3][:5]

            if related:
                topic = [word] + related
                topics.append({
                    "core_word": word,
                    "related_words": related,
                    "frequency": freq,
                })
                used_words.add(word)
                used_words.update(related)

            if len(topics) >= n_topics:
                break

        return {"topics": topics, "word_frequencies": dict(word_freq.most_common(100))}

    def find_emerging_terms(self, papers: List[Dict],
                           time_field: str = "published_date") -> Dict[str, Dict]:
        """Find terms that are emerging (growing in frequency over time)."""
        # Group papers by time period
        periods = defaultdict(list)
        for paper in papers:
            date = paper.get(time_field, "")
            if date:
                period = date[:7]  # YYYY-MM
                periods[period].append(paper)

        sorted_periods = sorted(periods.keys())
        if len(sorted_periods) < 2:
            return {}

        # Split into early and recent
        mid = len(sorted_periods) // 2
        early_periods = sorted_periods[:mid]
        recent_periods = sorted_periods[mid:]

        # Count terms in each half
        def count_terms(period_list):
            term_count = Counter()
            total = 0
            for period in period_list:
                for paper in periods[period]:
                    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
                    words = set(re.findall(r'\b[a-z][a-z\-]+[a-z]\b', text))
                    words = {w for w in words if w not in self.stopwords and len(w) > 3}
                    term_count.update(words)
                    total += 1
            return term_count, total

        early_counts, early_total = count_terms(early_periods)
        recent_counts, recent_total = count_terms(recent_periods)

        # Find emerging terms (much more frequent in recent)
        emerging = {}
        for term in recent_counts:
            recent_rate = recent_counts[term] / recent_total
            early_rate = early_counts.get(term, 0) / early_total if early_total else 0

            if recent_counts[term] >= 5:  # Minimum frequency
                if early_rate == 0:
                    growth = float('inf')
                else:
                    growth = recent_rate / early_rate

                if growth > 1.5:  # At least 50% growth
                    emerging[term] = {
                        "recent_count": recent_counts[term],
                        "early_count": early_counts.get(term, 0),
                        "growth_rate": round(growth, 2) if growth != float('inf') else "new",
                        "recent_rate": round(recent_rate * 100, 2),
                    }

        # Sort by growth
        sorted_emerging = sorted(
            emerging.items(),
            key=lambda x: x[1]["recent_count"] * (x[1]["growth_rate"] if isinstance(x[1]["growth_rate"], float) else 100),
            reverse=True
        )

        return {
            "emerging_terms": dict(sorted_emerging[:50]),
            "early_period": f"{early_periods[0]} to {early_periods[-1]}",
            "recent_period": f"{recent_periods[0]} to {recent_periods[-1]}",
        }

    def analyze_all(self, papers: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Run all analyses on a corpus of papers."""
        print(f"    Analyzing text for {len(papers)} papers...")

        # Per-paper analysis
        for paper in papers:
            analysis = self.analyze_paper(paper)
            paper.update(analysis)

        # Corpus-level analysis
        corpus_analysis = {
            "topics": self.extract_corpus_topics(papers),
            "emerging_terms": self.find_emerging_terms(papers),
            "method_frequency": self._count_methods(papers),
            "dataset_frequency": self._count_datasets(papers),
            "contribution_distribution": self._count_contributions(papers),
        }

        return papers, corpus_analysis

    def _count_methods(self, papers: List[Dict]) -> Dict[str, int]:
        """Count method frequencies across corpus."""
        counts = Counter()
        for paper in papers:
            counts.update(paper.get("methods_detected", []))
        return dict(counts.most_common(50))

    def _count_datasets(self, papers: List[Dict]) -> Dict[str, int]:
        """Count dataset frequencies across corpus."""
        counts = Counter()
        for paper in papers:
            counts.update(paper.get("datasets_mentioned", []))
        return dict(counts.most_common(50))

    def _count_contributions(self, papers: List[Dict]) -> Dict[str, int]:
        """Count contribution type frequencies."""
        counts = Counter()
        for paper in papers:
            counts.update(paper.get("contribution_types", []))
        return dict(counts)
