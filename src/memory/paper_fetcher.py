from __future__ import annotations
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import IO

_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_NS = "http://www.w3.org/2005/Atom"
_PDF_BASE = "https://arxiv.org/pdf/"


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    full_text: str | None = None


def normalize_arxiv_id(raw: str) -> str:
    """Normalize URL or bare ID to short form, e.g. '2305.10601'.

    Accepts:
    - "2305.10601"
    - "https://arxiv.org/abs/2305.10601"
    - "https://arxiv.org/pdf/2305.10601"
    - "arxiv:2305.10601"
    """
    raw = raw.strip()
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', raw)
    if m:
        return m.group(1).split("v")[0]
    if raw.lower().startswith("arxiv:"):
        raw = raw[6:]
    if re.fullmatch(r'[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?', raw):
        return raw.split("v")[0]
    raise ValueError(f"Cannot parse ArXiv ID from: {raw!r}")


def fetch_arxiv(arxiv_id: str, include_pdf: bool = False, timeout: float = 30.0) -> ArxivPaper:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    url = f"{_ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}"

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)
    ns = {"atom": _ARXIV_NS}
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"No ArXiv entry found for ID: {arxiv_id}")

    title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
    abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
    published = (entry.findtext("atom:published", "", ns) or "")[:10]

    authors = [
        (a.findtext("atom:name", "", ns) or "").strip()
        for a in entry.findall("atom:author", ns)
    ]

    categories = [
        t.get("term", "")
        for t in entry.findall("atom:category", ns)
        if t.get("term")
    ]

    full_text = None
    if include_pdf:
        full_text = _extract_pdf_text(arxiv_id, timeout=timeout)

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        published=published,
        full_text=full_text,
    )


def fetch_multiple(
    arxiv_ids: list[str],
    include_pdf: bool = False,
    timeout: float = 30.0,
) -> list[ArxivPaper]:
    import warnings
    results = []
    for raw_id in arxiv_ids:
        try:
            results.append(fetch_arxiv(raw_id, include_pdf=include_pdf, timeout=timeout))
        except Exception as exc:
            warnings.warn(f"paper_fetcher: skipping {raw_id!r}: {exc}", stacklevel=2)
    return results


def _extract_pdf_text(arxiv_id: str, timeout: float = 60.0) -> str | None:
    try:
        import pypdf  # type: ignore
    except ImportError:
        return None

    pdf_url = f"{_PDF_BASE}{urllib.parse.quote(arxiv_id)}"
    try:
        import io
        with urllib.request.urlopen(pdf_url, timeout=timeout) as resp:
            pdf_bytes = resp.read()
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except Exception:
        return None
