"""IGIO axis classifier.

Maps each chunk to one of four content axes derived from clinical/Mayring
research practice:

    - issue        — what problem or question is being addressed
    - goal         — what is the desired state or aim
    - intervention — what action, change, or implementation is being made
    - outcome      — what result, effect, or evidence has been observed

This is *orthogonal* to the existing Mayring `category_labels` (api/data_access/
domain/...): a chunk in `data_access` can be an `intervention` (change to a
repository) or an `outcome` (test result). The two dimensions are stored
independently on the same `chunks` row.

The classifier is LLM-only on purpose — accuracy of the IGIO axis is more
load-bearing than its compute cost (a misclassified `outcome` pollutes the
context-injection that the model sees on later sessions). Heuristic shortcuts
exist in `_FAST_HINTS` only to skip the LLM round-trip for unambiguous cases
that show up at high volume (e.g. test-result chunks).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from src.ollama_client import generate

logger = logging.getLogger(__name__)


VALID_AXES: tuple[str, ...] = ("issue", "goal", "intervention", "outcome")


@dataclass(frozen=True)
class IgioVerdict:
    axis: str          # one of VALID_AXES, or "" for "could not classify"
    confidence: float  # 0.0 .. 1.0
    rationale: str


_FAST_HINTS: tuple[tuple[re.Pattern, str, float], ...] = (
    # Test results / pytest outputs → outcome
    (re.compile(r"\b(\d+\s+passed|\d+\s+failed|tests? grün|tests? red)\b", re.I),
     "outcome", 0.92),
    # Stack traces / explicit bug reports → issue
    (re.compile(r"\bTraceback \(most recent call last\)", re.I),
     "issue", 0.95),
    (re.compile(r"\b(?:bug|broken|regression|TODO|FIXME)\b|\bfix(?:e[ds])?:", re.I),
     "issue", 0.78),
    # Plan/intervention markers
    (re.compile(r"^##?\s*Plan\b|\bImplemented\b|\brefactor(?:ed|ing)\b", re.I | re.M),
     "intervention", 0.75),
)


def _fast_classify(text: str) -> IgioVerdict | None:
    for pattern, axis, conf in _FAST_HINTS:
        if pattern.search(text):
            return IgioVerdict(axis=axis, confidence=conf,
                               rationale=f"keyword:{pattern.pattern[:32]}")
    return None


_PROMPT_SYSTEM = """You classify a text snippet into EXACTLY ONE of four
content axes. Respond with strict JSON only — no prose, no markdown fences.

The four axes (these are the ONLY valid `axis` values):
  issue        — surfaces a problem, bug, missing piece, or open question
  goal         — names a desired state, target, or success criterion
  intervention — describes a change, implementation, refactor, or action taken
  outcome      — reports a result, effect, test outcome, or measurable evidence

DO NOT use other words like "tests", "testing", "test", "fix", "bug", or
"implementation". A test function reporting pass/fail is `outcome`. A bug
report or open question is `issue`. A code change or refactor is
`intervention`. A goal statement is `goal`.

If the snippet does not fit any axis, return axis="" with low confidence.

Output schema (no other keys, no nested objects):
  {"axis": "<issue|goal|intervention|outcome|>", "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}
"""

# Common LLM-produced near-misses → canonical axis. The LLM keeps suggesting
# "tests"/"testing" for test functions even when the prompt forbids it; rather
# than discarding 20% of throughput we accept the synonym at a slight
# confidence penalty.
_AXIS_ALIASES: dict[str, str] = {
    "test": "outcome",
    "tests": "outcome",
    "testing": "outcome",
    "result": "outcome",
    "results": "outcome",
    "fix": "intervention",
    "fixes": "intervention",
    "refactor": "intervention",
    "implementation": "intervention",
    "bug": "issue",
    "problem": "issue",
    "question": "issue",
}
_ALIAS_PENALTY = 0.1


def _build_user_prompt(text: str, category_labels: list[str]) -> str:
    cats = ", ".join(category_labels) if category_labels else "(none)"
    snippet = text if len(text) <= 1500 else text[:1500] + "…"
    return (
        f"Mayring categories already assigned: {cats}\n\n"
        f"Snippet:\n```\n{snippet}\n```\n\n"
        "Respond with JSON only."
    )


def _strip_markdown_fence(raw: str) -> str:
    """Strip a leading ```json … ``` fence if the LLM wrapped its reply.

    Tolerant by design — trims trailing whitespace and accepts arbitrary text
    after the closing fence (some models append "Let me know if…" prose).
    Strategy: detect leading fence, drop optional language tag, then trim from
    the LAST ``` onwards. Strict endswith() would silently lose otherwise-
    valid fenced JSON.
    """
    if not raw.lstrip().startswith("```"):
        return raw
    body = raw.lstrip().lstrip("`").lstrip()
    # Drop optional language tag on the first line
    if "\n" in body:
        first, rest = body.split("\n", 1)
        if not first.strip().startswith("{"):
            body = rest
    body = body.rstrip()
    last_fence = body.rfind("```")
    if last_fence != -1:
        body = body[:last_fence]
    return body.strip()


def _parse_verdict(raw: str) -> IgioVerdict | None:
    """Parse the LLM reply into an IgioVerdict, or None on failure.

    Tolerates: leading/trailing prose, ```json fences, and JSON objects with
    nested arrays or braces inside `rationale`. Strategy: slice from the first
    `{` to the matching last `}` (greedy), then a single json.loads.
    """
    raw = _strip_markdown_fence(raw.strip())
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    axis = str(obj.get("axis", "")).strip().lower()
    aliased = False
    if axis and axis not in VALID_AXES:
        mapped = _AXIS_ALIASES.get(axis)
        if mapped is None:
            return None
        axis = mapped
        aliased = True
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if aliased:
        conf = max(0.0, conf - _ALIAS_PENALTY)
    rationale_raw = obj.get("rationale", "")
    rationale = (str(rationale_raw) if rationale_raw is not None else "")[:200]
    return IgioVerdict(
        axis=axis,
        confidence=max(0.0, min(1.0, conf)),
        rationale=rationale,
    )


def classify_chunk(
    text: str,
    category_labels: list[str] | None = None,
    *,
    ollama_url: str,
    model: str,
    timeout: float = 30.0,
) -> IgioVerdict:
    """Classify a single chunk into an IGIO axis.

    Returns a verdict with axis="" and confidence=0.0 when the LLM call fails
    or returns malformed output (the caller should leave the chunk unclassified
    rather than persist a bad label).
    """
    if not text.strip():
        return IgioVerdict(axis="", confidence=0.0, rationale="empty text")

    fast = _fast_classify(text)
    if fast is not None:
        return fast

    prompt = _build_user_prompt(text, category_labels or [])
    try:
        raw = generate(
            ollama_url, model, prompt,
            system=_PROMPT_SYSTEM,
            stream=False,
            timeout=timeout,
            max_retries=1,
        )
    except Exception as e:
        logger.warning("igio classify: LLM call failed (%s)", e)
        return IgioVerdict(axis="", confidence=0.0, rationale=f"llm_error: {e}")

    verdict = _parse_verdict(raw)
    if verdict is None:
        logger.warning("igio classify: could not parse LLM output: %r", raw[:120])
        return IgioVerdict(axis="", confidence=0.0, rationale="parse_error")
    return verdict


def classify_batch(
    chunks: Iterable[tuple[str, str, list[str]]],
    *,
    ollama_url: str,
    model: str,
    timeout: float = 30.0,
) -> list[tuple[str, IgioVerdict]]:
    """Classify a batch of (chunk_id, text, category_labels) tuples.

    Returns a list of (chunk_id, verdict) — order preserved. The function does
    not persist; callers wire the verdicts into `chunks.igio_*` themselves so
    the persistence path stays testable in isolation.
    """
    out: list[tuple[str, IgioVerdict]] = []
    for chunk_id, text, cats in chunks:
        v = classify_chunk(text, cats, ollama_url=ollama_url, model=model, timeout=timeout)
        out.append((chunk_id, v))
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
