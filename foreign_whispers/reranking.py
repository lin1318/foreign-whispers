"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import argostranslate.translate

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """

    """
    Strategy:
    1. Compress the English source text first.
    2. Translate each shorter English candidate into Spanish.
    3. Prefer candidates that fit the character budget.
    4. Use word-boundary truncation only as a fallback.
    """
    max_chars = int(target_duration_s * 15)
    candidates: list[TranslationCandidate] = []

    candidates.append(
        TranslationCandidate(
            text=baseline_es,
            char_count=len(baseline_es),
            brevity_rationale="Baseline translation"
        )
    )

    if len(baseline_es) <= max_chars:
        return candidates

    def clean_text(text: str) -> str:
        return " ".join(text.strip().split())

    def add_unique(items: list[str], text: str):
        text = clean_text(text)
        if text and text not in items:
            items.append(text)

    def truncate_by_words(text: str, limit: int) -> str:
        words = clean_text(text).split()
        kept = []

        for word in words:
            candidate = " ".join(kept + [word])
            if len(candidate) > limit:
                break
            kept.append(word)

        return " ".join(kept)

    def add_candidate(
        candidates: list[TranslationCandidate],
        text: str,
        rationale: str,
    ):
        text = clean_text(text)

        if not text:
            return

        if len(text) >= len(baseline_es):
            return

        if any(c.text == text for c in candidates):
            return

        candidates.append(
            TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            )
        )

    source = clean_text(source_text)
    baseline_es = clean_text(baseline_es)

    source_candidates: list[str] = []

    add_unique(source_candidates, source)

    for sep in [".", ",", ";", ":"]:
        if sep in source:
            add_unique(source_candidates, source.split(sep)[0])

    shortened = source
    replacements = {
        "let me explain": "",
        "in order to": "to",
        "due to the fact that": "because",
        "at this point in time": "now",
        "over the past few days": "recently",
        "as you can see": "",
        "actually": "",
        "basically": "",
        "really": "",
        "very": "",
        "just": "",
        "kind of": "",
        "sort of": "",
        "you know": "",
        "I mean": "",
    }

    for old, new in replacements.items():
        shortened = shortened.replace(old, new)

    add_unique(source_candidates, shortened)

    approx_source_limit = max(20, int(max_chars * 0.75))
    add_unique(source_candidates, truncate_by_words(source, approx_source_limit))

    for short_en in source_candidates:
        try:
            translated_es = argostranslate.translate.translate(
                short_en,
                "en",
                "es",
            )
        except Exception as exc:
            logger.warning(
                "Translation failed for source candidate %r: %s",
                short_en,
                exc,
            )
            continue

        add_candidate(
            candidates,
            translated_es,
            f"Compressed source English before translation: {short_en}",
        )

    if candidates and all(c.char_count > max_chars for c in candidates):
        fallback = truncate_by_words(baseline_es, max_chars)
        add_candidate(
            candidates,
            fallback,
            "Fallback word-boundary truncation to fit duration budget.",
        )

    if not candidates:
        fallback = truncate_by_words(baseline_es, max_chars)
        add_candidate(
            candidates,
            fallback,
            "Fallback word-boundary truncation to fit duration budget.",
        )

    candidates.sort(
        key=lambda c: (
            c.char_count > max_chars,
            abs(c.char_count - max_chars)
        )
    )

    logger.info(
        "Generated %d shorter translation candidates for %.1fs budget (%d chars).",
        len(candidates),
        target_duration_s,
        max_chars,
    )

    return candidates
