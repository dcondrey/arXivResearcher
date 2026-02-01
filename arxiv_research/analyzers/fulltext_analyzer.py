"""
Full-Text PDF Analysis Module
=============================
Download and analyze arXiv PDFs for deeper insights:
- Extract full text from PDFs
- Parse paper sections (intro, methods, results, limitations, conclusion)
- Extract limitations and future work (research gap goldmines!)
- Detailed method extraction beyond abstracts
- Paper structure analysis
"""

import re
import os
import time
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# Section header patterns - handles various formats:
# "5. Limitations", "5 Limitations", "LIMITATIONS", "5.1 Limitations", etc.
SECTION_PATTERNS = {
    "abstract": [
        r"^#{0,2}\s*(?:Abstract|ABSTRACT)\s*$",
        r"^(?:\d+\.?\s*)?(?:Abstract|ABSTRACT)\s*$",
    ],
    "introduction": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)\s*$",
        r"^(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)\s*$",
        r"^I\.\s*(?:Introduction|INTRODUCTION)\s*$",
    ],
    "related_work": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Related\s+Work|RELATED\s+WORK|Background|BACKGROUND|Prior\s+Work|PRIOR\s+WORK)\s*$",
        r"^(?:\d+\.?\s*)?(?:Related\s+Work|RELATED\s+WORK|Background|BACKGROUND|Literature\s+Review|LITERATURE\s+REVIEW)\s*$",
        r"^II\.\s*(?:Related\s+Work|Background)\s*$",
    ],
    "methods": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Method(?:s|ology)?|METHOD(?:S|OLOGY)?|Approach|APPROACH|Model|MODEL|Framework|FRAMEWORK)\s*$",
        r"^(?:\d+\.?\s*)?(?:Method(?:s|ology)?|METHOD(?:S|OLOGY)?|Proposed\s+(?:Method|Approach)|Our\s+(?:Method|Approach))\s*$",
        r"^III\.\s*(?:Method(?:s|ology)?|Approach|Model)\s*$",
        r"^(?:\d+\.?\s*)?(?:Technical\s+Approach|TECHNICAL\s+APPROACH)\s*$",
    ],
    "experiments": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Experiment(?:s|al)?(?:\s+(?:Results?|Setup|Settings?))?|EXPERIMENT(?:S|AL)?)\s*$",
        r"^(?:\d+\.?\s*)?(?:Experiment(?:s|al)?|Evaluation|EVALUATION|Results?\s+and\s+(?:Analysis|Discussion))\s*$",
        r"^IV\.\s*(?:Experiment(?:s)?|Evaluation)\s*$",
        r"^(?:\d+\.?\s*)?(?:Empirical\s+(?:Study|Evaluation)|EMPIRICAL\s+(?:STUDY|EVALUATION))\s*$",
    ],
    "results": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Results?|RESULTS?)\s*$",
        r"^(?:\d+\.?\s*)?(?:Results?\s*(?:and\s+(?:Analysis|Discussion))?)\s*$",
        r"^V\.\s*(?:Results?)\s*$",
        r"^(?:\d+\.?\s*)?(?:Main\s+Results?|MAIN\s+RESULTS?)\s*$",
    ],
    "discussion": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Discussion|DISCUSSION)\s*$",
        r"^(?:\d+\.?\s*)?(?:Discussion|Analysis|ANALYSIS)\s*$",
        r"^VI\.\s*(?:Discussion)\s*$",
    ],
    "limitations": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Limitation(?:s)?|LIMITATION(?:S)?)\s*$",
        r"^(?:\d+\.?\d*\.?\s*)?(?:Limitation(?:s)?|LIMITATION(?:S)?)\s*$",
        r"^(?:\d+\.?\s*)?(?:Limitation(?:s)?\s+and\s+Future\s+Work)\s*$",
        r"^(?:Limitation(?:s)?\s+of\s+(?:Our|This|the)\s+(?:Work|Study|Approach|Method))\s*$",
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Threats\s+to\s+Validity|THREATS\s+TO\s+VALIDITY)\s*$",
    ],
    "future_work": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Future\s+Work|FUTURE\s+WORK|Future\s+Directions?|FUTURE\s+DIRECTIONS?)\s*$",
        r"^(?:\d+\.?\d*\.?\s*)?(?:Future\s+Work|FUTURE\s+WORK)\s*$",
        r"^(?:\d+\.?\s*)?(?:Future\s+Research\s+Directions?)\s*$",
        r"^(?:Open\s+(?:Problems?|Questions?)|OPEN\s+(?:PROBLEMS?|QUESTIONS?))\s*$",
    ],
    "conclusion": [
        r"^#{0,2}\s*(?:\d+\.?\s*)?(?:Conclusion(?:s)?|CONCLUSION(?:S)?)\s*$",
        r"^(?:\d+\.?\s*)?(?:Conclusion(?:s)?|CONCLUSION(?:S)?|Summary|SUMMARY)\s*$",
        r"^(?:VII|VIII|IX|X)\.\s*(?:Conclusion(?:s)?)\s*$",
        r"^(?:\d+\.?\s*)?(?:Concluding\s+Remarks?|CONCLUDING\s+REMARKS?)\s*$",
    ],
    "acknowledgments": [
        r"^#{0,2}\s*(?:Acknowledgment(?:s)?|ACKNOWLEDGMENT(?:S)?)\s*$",
        r"^(?:Acknowledgment(?:s)?|ACKNOWLEDGMENT(?:S)?)\s*$",
    ],
    "references": [
        r"^#{0,2}\s*(?:Reference(?:s)?|REFERENCE(?:S)?|Bibliography|BIBLIOGRAPHY)\s*$",
        r"^(?:Reference(?:s)?|REFERENCE(?:S)?)\s*$",
    ],
    "appendix": [
        r"^#{0,2}\s*(?:Appendix|APPENDIX|Appendices|APPENDICES)(?:\s+[A-Z])?\.?\s*$",
        r"^(?:Appendix|APPENDIX)(?:\s+[A-Z])?(?:\.|\:)?\s*",
        r"^(?:Supplementary\s+Material(?:s)?|SUPPLEMENTARY\s+MATERIAL(?:S)?)\s*$",
    ],
}

# Patterns for in-text limitation mentions (when no dedicated section exists)
INLINE_LIMITATION_PATTERNS = [
    r"(?:our|this|the)\s+(?:work|method|approach|model|study)\s+(?:has|have|is)\s+(?:several\s+)?(?:some\s+)?(?:limitations?|limited)",
    r"(?:one|a)\s+(?:key\s+)?limitation\s+(?:of|is)\s+",
    r"(?:we|our\s+work)\s+(?:do(?:es)?\s+not|cannot|fail(?:s)?\s+to)",
    r"(?:it\s+is|there\s+(?:is|are)|we\s+found)\s+(?:difficult|challenging|hard)\s+to",
    r"(?:however|unfortunately|admittedly),?\s+(?:our|this|the)\s+(?:method|approach|model)",
    r"(?:this\s+)?(?:approach|method|model)\s+(?:struggles?|fails?)\s+(?:with|to|when)",
    r"(?:does\s+not\s+(?:generalize|scale|work)\s+well)",
    r"(?:is\s+(?:not\s+(?:able|designed|intended)|limited)\s+to)",
    r"(?:remains?\s+(?:a|an)\s+(?:open|unsolved|challenging)\s+(?:problem|question|issue))",
    r"(?:require(?:s|d)?\s+(?:further|additional|more)\s+(?:research|investigation|study))",
]

# Patterns for future work mentions
INLINE_FUTURE_WORK_PATTERNS = [
    r"(?:in\s+)?(?:the\s+)?future(?:\s+work)?(?:,)?\s+(?:we|it|one)\s+(?:will|would|could|should|plan\s+to|aim\s+to|intend\s+to)",
    r"(?:we|it)\s+(?:leave|defer|postpone)\s+(?:this|it|these)\s+(?:for|to)\s+future\s+(?:work|research|study)",
    r"(?:an?\s+)?(?:interesting|promising|natural|important)\s+(?:direction|avenue|extension|area)\s+(?:for\s+future\s+(?:work|research)|would\s+be)",
    r"(?:we\s+plan\s+to|we\s+will|we\s+aim\s+to|we\s+intend\s+to)\s+(?:extend|explore|investigate|study|address)",
    r"(?:it\s+would\s+be\s+(?:interesting|valuable|worthwhile)\s+to)",
    r"(?:further\s+(?:research|investigation|study)\s+(?:is\s+needed|should|could))",
    r"(?:open\s+(?:questions?|problems?|issues?)\s+(?:include|remain|are))",
    r"(?:remains?\s+to\s+be\s+(?:seen|explored|investigated|studied))",
]


class FullTextAnalyzer:
    """Download and analyze arXiv PDFs for deep text extraction."""

    def __init__(self, cache_dir: Optional[str] = None, rate_limit: float = 3.0):
        """
        Initialize the full-text analyzer.

        Args:
            cache_dir: Directory for caching PDFs and extracted text.
                      Defaults to ~/.arxiv_cache
            rate_limit: Seconds between arXiv requests (default 3.0 to be polite)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.arxiv_cache")

        self.cache_dir = Path(cache_dir)
        self.pdf_cache_dir = self.cache_dir / "pdfs"
        self.text_cache_dir = self.cache_dir / "texts"
        self.rate_limit = rate_limit
        self._last_request_time = 0

        # Create cache directories
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)

        # Compile regex patterns
        self._compiled_section_patterns = {}
        for section, patterns in SECTION_PATTERNS.items():
            self._compiled_section_patterns[section] = [
                re.compile(p, re.MULTILINE | re.IGNORECASE) for p in patterns
            ]

        self._compiled_limitation_patterns = [
            re.compile(p, re.IGNORECASE) for p in INLINE_LIMITATION_PATTERNS
        ]
        self._compiled_future_work_patterns = [
            re.compile(p, re.IGNORECASE) for p in INLINE_FUTURE_WORK_PATTERNS
        ]

        # Try to import PDF library
        self._pdf_library = None
        try:
            import fitz  # PyMuPDF
            self._pdf_library = "pymupdf"
        except ImportError:
            try:
                import pdfplumber
                self._pdf_library = "pdfplumber"
            except ImportError:
                print("Warning: Neither PyMuPDF (fitz) nor pdfplumber is installed.")
                print("Install with: pip install pymupdf  or  pip install pdfplumber")

    def _rate_limit_wait(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get_arxiv_pdf_url(self, arxiv_id: str) -> str:
        """Get the PDF URL for an arXiv ID."""
        # Clean the arxiv_id
        arxiv_id = arxiv_id.strip()
        if arxiv_id.startswith("arXiv:"):
            arxiv_id = arxiv_id[6:]

        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    def _get_cached_pdf_path(self, arxiv_id: str) -> Path:
        """Get the cache path for a PDF."""
        # Sanitize arxiv_id for filesystem
        safe_id = arxiv_id.replace("/", "_").replace(":", "_")
        return self.pdf_cache_dir / f"{safe_id}.pdf"

    def _get_cached_text_path(self, arxiv_id: str) -> Path:
        """Get the cache path for extracted text."""
        safe_id = arxiv_id.replace("/", "_").replace(":", "_")
        return self.text_cache_dir / f"{safe_id}.txt"

    def download_pdf(self, arxiv_id: str, cache_dir: Optional[str] = None) -> Optional[Path]:
        """
        Download a PDF from arXiv with caching.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.00001" or "cs.LG/0001001")
            cache_dir: Optional override for cache directory

        Returns:
            Path to the downloaded PDF, or None if download failed
        """
        if cache_dir:
            pdf_path = Path(cache_dir) / f"{arxiv_id.replace('/', '_')}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            pdf_path = self._get_cached_pdf_path(arxiv_id)

        # Check if already cached
        if pdf_path.exists():
            return pdf_path

        # Rate limit
        self._rate_limit_wait()

        url = self._get_arxiv_pdf_url(arxiv_id)

        try:
            # Set a user agent to be a good citizen
            headers = {
                "User-Agent": "arXiv-research-tool/1.0 (academic research; respects rate limits)"
            }
            request = Request(url, headers=headers)

            with urlopen(request, timeout=60) as response:
                content = response.read()

                # Verify it's a PDF
                if not content.startswith(b"%PDF"):
                    print(f"Warning: {arxiv_id} - Response is not a PDF")
                    return None

                # Save to cache
                with open(pdf_path, "wb") as f:
                    f.write(content)

                return pdf_path

        except HTTPError as e:
            print(f"HTTP Error downloading {arxiv_id}: {e.code} {e.reason}")
            return None
        except URLError as e:
            print(f"URL Error downloading {arxiv_id}: {e.reason}")
            return None
        except Exception as e:
            print(f"Error downloading {arxiv_id}: {e}")
            return None

    def extract_text(self, pdf_path: str | Path) -> Optional[str]:
        """
        Extract full text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text, or None if extraction failed
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            return None

        # Check text cache
        arxiv_id = pdf_path.stem
        text_cache_path = self._get_cached_text_path(arxiv_id)

        if text_cache_path.exists():
            with open(text_cache_path, "r", encoding="utf-8") as f:
                return f.read()

        # Extract text based on available library
        text = None

        if self._pdf_library == "pymupdf":
            text = self._extract_with_pymupdf(pdf_path)
        elif self._pdf_library == "pdfplumber":
            text = self._extract_with_pdfplumber(pdf_path)
        else:
            print("No PDF library available. Install pymupdf or pdfplumber.")
            return None

        if text:
            # Cache extracted text
            with open(text_cache_path, "w", encoding="utf-8") as f:
                f.write(text)

        return text

    def _extract_with_pymupdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text using PyMuPDF."""
        try:
            import fitz

            text_parts = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text_parts.append(page.get_text())

            return "\n".join(text_parts)

        except Exception as e:
            print(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return None

    def _extract_with_pdfplumber(self, pdf_path: Path) -> Optional[str]:
        """Extract text using pdfplumber."""
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n".join(text_parts)

        except Exception as e:
            print(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return None

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Parse text into sections.

        Args:
            text: Full text of the paper

        Returns:
            Dictionary mapping section names to their content
        """
        if not text:
            return {}

        lines = text.split("\n")
        sections = {}
        current_section = "preamble"
        current_content = []
        section_order = []

        for line in lines:
            stripped = line.strip()

            # Check if this line is a section header
            found_section = None
            for section, patterns in self._compiled_section_patterns.items():
                for pattern in patterns:
                    if pattern.match(stripped):
                        found_section = section
                        break
                if found_section:
                    break

            if found_section:
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections[current_section] = content
                        if current_section not in section_order:
                            section_order.append(current_section)

                current_section = found_section
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections[current_section] = content
                if current_section not in section_order:
                    section_order.append(current_section)

        # Store section order for structure analysis
        sections["_section_order"] = section_order

        return sections

    def extract_limitations(self, papers: List[Dict]) -> List[Dict]:
        """
        Extract limitation sections from papers - goldmine for research gaps!

        Args:
            papers: List of paper dictionaries with arxiv_id

        Returns:
            List of papers with extracted limitations
        """
        results = []
        total = len(papers)

        print(f"    Extracting limitations from {total} papers...")

        for i, paper in enumerate(papers):
            if (i + 1) % 10 == 0:
                print(f"      Progress: {i + 1}/{total}")

            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id:
                continue

            result = {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "limitations_section": None,
                "inline_limitations": [],
                "limitation_themes": [],
                "has_limitations": False,
            }

            # Try to get cached text or download/extract
            text = self._get_paper_text(arxiv_id)

            if text:
                sections = self.extract_sections(text)

                # Check for dedicated limitations section
                if "limitations" in sections:
                    result["limitations_section"] = sections["limitations"]
                    result["has_limitations"] = True

                # Also check conclusion for embedded limitations
                conclusion_text = sections.get("conclusion", "")

                # Find inline limitation mentions
                all_text = text if not result["limitations_section"] else conclusion_text
                inline_mentions = self._find_inline_limitations(all_text)
                result["inline_limitations"] = inline_mentions

                if inline_mentions and not result["has_limitations"]:
                    result["has_limitations"] = True

                # Extract limitation themes
                if result["has_limitations"]:
                    limitation_text = result["limitations_section"] or " ".join(inline_mentions)
                    result["limitation_themes"] = self._extract_limitation_themes(limitation_text)

            results.append(result)

        return results

    def _find_inline_limitations(self, text: str, context_chars: int = 200) -> List[str]:
        """Find inline limitation mentions with context."""
        mentions = []

        for pattern in self._compiled_limitation_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - context_chars // 2)
                end = min(len(text), match.end() + context_chars // 2)
                context = text[start:end].strip()
                # Clean up whitespace
                context = re.sub(r"\s+", " ", context)
                if context and context not in mentions:
                    mentions.append(context)

        return mentions[:10]  # Limit to 10 mentions

    def _extract_limitation_themes(self, text: str) -> List[str]:
        """Extract common limitation themes from text."""
        themes = []

        theme_patterns = {
            "scalability": r"\b(?:scal(?:e|ability|ing)|large-?scale|computational cost|memory|efficiency)\b",
            "generalization": r"\b(?:generali[sz](?:e|ation)|domain|transfer|out-of-distribution|ood)\b",
            "data_requirements": r"\b(?:data|labeled|annotation|training (?:data|examples)|sample(?:s)?)\b",
            "assumptions": r"\b(?:assum(?:e|ption|ing)|require(?:s|ment)?|condition|constraint)\b",
            "evaluation": r"\b(?:evaluat(?:e|ion)|benchmark|metric|dataset|test(?:ing)?)\b",
            "theoretical": r"\b(?:theoretic(?:al)?|formal|proof|guarantee|bound)\b",
            "implementation": r"\b(?:implement(?:ation)?|engineering|practical|deploy(?:ment)?)\b",
            "interpretability": r"\b(?:interpret(?:ability)?|explain(?:ability)?|understand(?:ing)?|black-?box)\b",
            "bias_fairness": r"\b(?:bias|fair(?:ness)?|ethical|discriminat(?:e|ion))\b",
            "robustness": r"\b(?:robust(?:ness)?|adversarial|attack|noise|perturbation)\b",
        }

        text_lower = text.lower()
        for theme, pattern in theme_patterns.items():
            if re.search(pattern, text_lower):
                themes.append(theme)

        return themes

    def extract_future_work(self, papers: List[Dict]) -> List[Dict]:
        """
        Extract future work sections - authors telling you what to write!

        Args:
            papers: List of paper dictionaries with arxiv_id

        Returns:
            List of papers with extracted future work suggestions
        """
        results = []
        total = len(papers)

        print(f"    Extracting future work from {total} papers...")

        for i, paper in enumerate(papers):
            if (i + 1) % 10 == 0:
                print(f"      Progress: {i + 1}/{total}")

            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id:
                continue

            result = {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "future_work_section": None,
                "inline_future_work": [],
                "future_directions": [],
                "has_future_work": False,
            }

            text = self._get_paper_text(arxiv_id)

            if text:
                sections = self.extract_sections(text)

                # Check for dedicated future work section
                if "future_work" in sections:
                    result["future_work_section"] = sections["future_work"]
                    result["has_future_work"] = True

                # Also check conclusion and limitations for embedded future work
                conclusion_text = sections.get("conclusion", "")
                limitations_text = sections.get("limitations", "")

                # Find inline future work mentions
                search_text = conclusion_text + " " + limitations_text
                if not result["future_work_section"]:
                    search_text = text  # Search full text if no dedicated section

                inline_mentions = self._find_inline_future_work(search_text)
                result["inline_future_work"] = inline_mentions

                if inline_mentions and not result["has_future_work"]:
                    result["has_future_work"] = True

                # Extract specific future directions
                if result["has_future_work"]:
                    future_text = result["future_work_section"] or " ".join(inline_mentions)
                    result["future_directions"] = self._extract_future_directions(future_text)

            results.append(result)

        return results

    def _find_inline_future_work(self, text: str, context_chars: int = 250) -> List[str]:
        """Find inline future work mentions with context."""
        mentions = []

        for pattern in self._compiled_future_work_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - context_chars // 4)
                end = min(len(text), match.end() + context_chars)
                context = text[start:end].strip()
                # Clean up whitespace
                context = re.sub(r"\s+", " ", context)
                if context and context not in mentions:
                    mentions.append(context)

        return mentions[:10]

    def _extract_future_directions(self, text: str) -> List[str]:
        """Extract specific future research directions."""
        directions = []

        # Split into sentences
        sentences = re.split(r"[.!?]\s+", text)

        direction_indicators = [
            r"(?:we\s+)?(?:plan|aim|intend|hope)\s+to",
            r"(?:would\s+be\s+)?(?:interesting|valuable|worthwhile)\s+to",
            r"future\s+(?:work|research)\s+(?:could|should|will|may)",
            r"(?:an?\s+)?(?:natural|promising|interesting)\s+(?:direction|extension)",
            r"remains?\s+(?:to\s+be|an?\s+open)",
            r"(?:further|additional)\s+(?:research|investigation|study)",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            for pattern in direction_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Clean and add
                    clean_sentence = re.sub(r"\s+", " ", sentence)
                    if clean_sentence not in directions:
                        directions.append(clean_sentence)
                    break

        return directions[:15]

    def extract_methods_detailed(self, text: str) -> Dict[str, any]:
        """
        Extract detailed method information beyond abstract.

        Args:
            text: Full text of the paper

        Returns:
            Dictionary with detailed method information
        """
        sections = self.extract_sections(text)
        methods_text = sections.get("methods", "")

        if not methods_text:
            # Try alternative section names
            for alt in ["approach", "model", "framework", "methodology"]:
                if alt in sections:
                    methods_text = sections[alt]
                    break

        result = {
            "has_methods_section": bool(methods_text),
            "methods_length": len(methods_text),
            "algorithms_mentioned": [],
            "hyperparameters": [],
            "model_components": [],
            "training_details": [],
            "implementation_details": [],
        }

        if not methods_text:
            return result

        # Extract algorithms
        algo_patterns = [
            r"Algorithm\s+\d+",
            r"(?:we\s+use|using|employ)\s+(?:the\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+algorithm",
            r"(?:SGD|Adam|AdamW|RMSprop|Momentum)",
        ]
        for pattern in algo_patterns:
            matches = re.findall(pattern, methods_text, re.IGNORECASE)
            result["algorithms_mentioned"].extend(matches)

        # Extract hyperparameters
        hyperparam_patterns = [
            r"learning\s+rate\s+(?:of\s+)?([0-9.e\-]+)",
            r"batch\s+size\s+(?:of\s+)?(\d+)",
            r"(?:trained\s+for\s+)?(\d+)\s+epochs?",
            r"dropout\s+(?:rate\s+)?(?:of\s+)?([0-9.]+)",
            r"weight\s+decay\s+(?:of\s+)?([0-9.e\-]+)",
            r"hidden\s+(?:dimension|size)\s+(?:of\s+)?(\d+)",
            r"(\d+)\s+(?:attention\s+)?heads?",
            r"(\d+)\s+(?:transformer\s+)?layers?",
        ]
        for pattern in hyperparam_patterns:
            matches = re.findall(pattern, methods_text, re.IGNORECASE)
            if matches:
                result["hyperparameters"].append({
                    "pattern": pattern.split("(")[0].strip(),
                    "values": matches[:5]
                })

        # Extract model components
        component_patterns = [
            r"(?:encoder|decoder|attention|transformer|embedding|layer|module|block|head)",
        ]
        for pattern in component_patterns:
            matches = re.findall(pattern, methods_text, re.IGNORECASE)
            result["model_components"] = list(set(matches))

        # Extract training details
        training_patterns = [
            r"(?:trained|fine-tuned?)\s+(?:on|using|with)\s+([^.]+)",
            r"(?:GPU|TPU)s?\s+(?:with\s+)?(\d+\s*GB)?",
            r"training\s+(?:took|required)\s+([^.]+)",
        ]
        for pattern in training_patterns:
            matches = re.findall(pattern, methods_text, re.IGNORECASE)
            result["training_details"].extend([m.strip() for m in matches if m.strip()])

        return result

    def analyze_paper_structure(self, papers: List[Dict]) -> Dict:
        """
        Analyze paper structure statistics.

        Args:
            papers: List of paper dictionaries with arxiv_id

        Returns:
            Statistics on paper structures
        """
        results = {
            "papers_analyzed": 0,
            "papers_with_full_text": 0,
            "avg_length_chars": 0,
            "avg_sections": 0,
            "section_frequency": Counter(),
            "section_lengths": defaultdict(list),
            "papers_with_limitations": 0,
            "papers_with_future_work": 0,
            "figure_counts": [],
            "table_counts": [],
            "equation_counts": [],
            "reference_counts": [],
            "structure_patterns": Counter(),
        }

        total_chars = 0
        total_sections = 0
        total = len(papers)

        print(f"    Analyzing structure of {total} papers...")

        for i, paper in enumerate(papers):
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i + 1}/{total}")

            arxiv_id = paper.get("arxiv_id", "")
            if not arxiv_id:
                continue

            results["papers_analyzed"] += 1

            text = self._get_paper_text(arxiv_id)

            if not text:
                continue

            results["papers_with_full_text"] += 1
            total_chars += len(text)

            sections = self.extract_sections(text)
            section_order = sections.get("_section_order", [])

            total_sections += len(section_order)

            # Count section occurrences
            for section in section_order:
                results["section_frequency"][section] += 1
                if section in sections:
                    results["section_lengths"][section].append(len(sections[section]))

            # Check for specific sections
            if "limitations" in section_order:
                results["papers_with_limitations"] += 1
            if "future_work" in section_order:
                results["papers_with_future_work"] += 1

            # Count figures, tables, equations
            fig_count = len(re.findall(r"(?:Figure|Fig\.?)\s*\d+", text, re.IGNORECASE))
            table_count = len(re.findall(r"Table\s*\d+", text, re.IGNORECASE))
            eq_count = len(re.findall(r"(?:Equation|Eq\.?)\s*\(?(\d+)\)?", text, re.IGNORECASE))

            results["figure_counts"].append(fig_count)
            results["table_counts"].append(table_count)
            results["equation_counts"].append(eq_count)

            # Count references (rough estimate)
            ref_section = sections.get("references", "")
            ref_count = len(re.findall(r"\[\d+\]|\[\w+\d+\]", ref_section))
            results["reference_counts"].append(ref_count)

            # Track structure patterns
            # Create a simplified structure signature
            key_sections = ["introduction", "methods", "experiments", "results", "conclusion"]
            pattern = "-".join([s[:3] for s in section_order if s in key_sections])
            if pattern:
                results["structure_patterns"][pattern] += 1

        # Compute averages
        n = results["papers_with_full_text"]
        if n > 0:
            results["avg_length_chars"] = round(total_chars / n)
            results["avg_sections"] = round(total_sections / n, 1)
            results["avg_figures"] = round(sum(results["figure_counts"]) / n, 1)
            results["avg_tables"] = round(sum(results["table_counts"]) / n, 1)
            results["avg_equations"] = round(sum(results["equation_counts"]) / n, 1)
            results["avg_references"] = round(sum(results["reference_counts"]) / n, 1)

            # Compute section length averages
            avg_section_lengths = {}
            for section, lengths in results["section_lengths"].items():
                if lengths:
                    avg_section_lengths[section] = round(sum(lengths) / len(lengths))
            results["avg_section_lengths"] = avg_section_lengths

        # Convert to serializable format
        results["section_frequency"] = dict(results["section_frequency"].most_common())
        results["structure_patterns"] = dict(results["structure_patterns"].most_common(20))

        # Remove raw lists to save space
        del results["figure_counts"]
        del results["table_counts"]
        del results["equation_counts"]
        del results["reference_counts"]
        del results["section_lengths"]

        return results

    def _get_paper_text(self, arxiv_id: str) -> Optional[str]:
        """Get paper text from cache or by downloading/extracting."""
        # Check text cache first
        text_path = self._get_cached_text_path(arxiv_id)
        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                return f.read()

        # Check if PDF is cached
        pdf_path = self._get_cached_pdf_path(arxiv_id)
        if pdf_path.exists():
            return self.extract_text(pdf_path)

        # Need to download
        pdf_path = self.download_pdf(arxiv_id)
        if pdf_path:
            return self.extract_text(pdf_path)

        return None

    def batch_download(self, arxiv_ids: List[str], show_progress: bool = True) -> Dict[str, Path]:
        """
        Download multiple PDFs with progress indication.

        Args:
            arxiv_ids: List of arXiv IDs to download
            show_progress: Whether to show progress updates

        Returns:
            Dictionary mapping arxiv_id to downloaded PDF path (or None if failed)
        """
        results = {}
        total = len(arxiv_ids)

        if show_progress:
            print(f"    Downloading {total} PDFs (rate limit: {self.rate_limit}s between requests)...")

        for i, arxiv_id in enumerate(arxiv_ids):
            if show_progress and (i + 1) % 5 == 0:
                print(f"      Progress: {i + 1}/{total}")

            pdf_path = self.download_pdf(arxiv_id)
            results[arxiv_id] = pdf_path

        if show_progress:
            success = sum(1 for p in results.values() if p is not None)
            print(f"    Downloaded {success}/{total} PDFs successfully")

        return results

    def batch_extract_text(self, pdf_paths: Dict[str, Path], show_progress: bool = True) -> Dict[str, str]:
        """
        Extract text from multiple PDFs with progress indication.

        Args:
            pdf_paths: Dictionary mapping arxiv_id to PDF path
            show_progress: Whether to show progress updates

        Returns:
            Dictionary mapping arxiv_id to extracted text
        """
        results = {}
        total = len(pdf_paths)

        if show_progress:
            print(f"    Extracting text from {total} PDFs...")

        for i, (arxiv_id, pdf_path) in enumerate(pdf_paths.items()):
            if show_progress and (i + 1) % 10 == 0:
                print(f"      Progress: {i + 1}/{total}")

            if pdf_path and pdf_path.exists():
                text = self.extract_text(pdf_path)
                results[arxiv_id] = text
            else:
                results[arxiv_id] = None

        if show_progress:
            success = sum(1 for t in results.values() if t is not None)
            print(f"    Extracted text from {success}/{total} PDFs successfully")

        return results

    def analyze_papers(self, papers: List[Dict], download_missing: bool = False) -> Dict:
        """
        Run full text analysis on a list of papers.

        Args:
            papers: List of paper dictionaries with arxiv_id
            download_missing: Whether to download PDFs that aren't cached

        Returns:
            Comprehensive analysis results
        """
        print(f"    Running full-text analysis on {len(papers)} papers...")

        # Get arxiv_ids
        arxiv_ids = [p.get("arxiv_id", "") for p in papers if p.get("arxiv_id")]

        # Download missing PDFs if requested
        if download_missing:
            missing = [aid for aid in arxiv_ids if not self._get_cached_pdf_path(aid).exists()]
            if missing:
                print(f"      Downloading {len(missing)} missing PDFs...")
                self.batch_download(missing)

        # Extract limitations
        limitations_results = self.extract_limitations(papers)

        # Extract future work
        future_work_results = self.extract_future_work(papers)

        # Analyze structure
        structure_stats = self.analyze_paper_structure(papers)

        # Aggregate limitation themes
        all_limitation_themes = Counter()
        for result in limitations_results:
            all_limitation_themes.update(result.get("limitation_themes", []))

        # Aggregate future directions
        all_future_directions = []
        for result in future_work_results:
            all_future_directions.extend(result.get("future_directions", []))

        return {
            "structure_analysis": structure_stats,
            "limitation_analysis": {
                "papers_with_limitations": sum(1 for r in limitations_results if r["has_limitations"]),
                "papers_with_limitation_section": sum(1 for r in limitations_results if r["limitations_section"]),
                "limitation_theme_frequency": dict(all_limitation_themes.most_common()),
                "paper_limitations": limitations_results,
            },
            "future_work_analysis": {
                "papers_with_future_work": sum(1 for r in future_work_results if r["has_future_work"]),
                "papers_with_future_work_section": sum(1 for r in future_work_results if r["future_work_section"]),
                "all_future_directions": all_future_directions[:100],  # Top 100
                "paper_future_work": future_work_results,
            },
            "summary": {
                "total_papers": len(papers),
                "papers_analyzed": structure_stats.get("papers_with_full_text", 0),
                "cache_dir": str(self.cache_dir),
            },
        }

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear the cache directory.

        Args:
            older_than_days: If specified, only clear files older than this many days
        """
        import shutil
        from datetime import datetime, timedelta

        if older_than_days is None:
            # Clear everything
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
            self.text_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Cleared cache at {self.cache_dir}")
        else:
            # Clear old files
            cutoff = datetime.now() - timedelta(days=older_than_days)
            cleared = 0

            for cache_subdir in [self.pdf_cache_dir, self.text_cache_dir]:
                for file_path in cache_subdir.iterdir():
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff:
                            file_path.unlink()
                            cleared += 1

            print(f"Cleared {cleared} files older than {older_than_days} days")
