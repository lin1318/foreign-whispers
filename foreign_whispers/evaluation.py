"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def normalize_to_unit(x: float) -> float:
    return max(0.0, min(1.0, x))


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict,
) -> dict:
    """Return a multi-dimensional dubbing quality scorecard.

    Scores are normalized to [0, 1], where higher is better.
    """
    if not metrics:
        return {
            "timing_accuracy": 0.0,
            "intelligibility": 0.0,
            "semantic_fidelity": 0.0,
            "naturalness": 0.0,
            "overall": 0.0,
        }

    # 1. Timing accuracy
    mae = float(align_report.get("mean_abs_duration_error_s", 0.0))
    severe_pct = float(align_report.get("pct_severe_stretch", 0.0)) / 100.0
    drift = abs(float(align_report.get("total_cumulative_drift_s", 0.0)))

    timing_accuracy = 1.0
    timing_accuracy *= 1.0 - normalize_to_unit(mae / 4.0)
    timing_accuracy *= 1.0 - normalize_to_unit(severe_pct)
    timing_accuracy *= 1.0 - normalize_to_unit(drift / 10.0)

    # 2. Intelligibility proxy:
    # large stretch factors often make speech sound unnatural or distorted
    avg_stretch_penalty = sum(
        max(0.0, a.stretch_factor - 1.0) for a in aligned
    ) / max(len(aligned), 1)
    intelligibility = 1.0 - normalize_to_unit(avg_stretch_penalty / 0.5)

    # 3. Semantic fidelity proxy:
    # very aggressive shortening can lose meaning, so compare length ratios
    ratios = [
        len(m.translated_text) / max(1, len(m.source_text))
        for m in metrics
    ]
    avg_ratio = sum(ratios) / len(ratios)
    semantic_fidelity = 1.0 - normalize_to_unit(abs(avg_ratio - 1.0) / 1.5)

    # 4. Naturalness:
    # penalize high variance in speech-rate / stretch across segments
    stretch_factors = [a.stretch_factor for a in aligned] or [1.0]
    mean_stretch = sum(stretch_factors) / len(stretch_factors)
    variance = sum((x - mean_stretch) ** 2 for x in stretch_factors) / len(stretch_factors)
    naturalness = 1.0 - normalize_to_unit(variance / 0.25)

    overall = (
        0.35 * timing_accuracy
        + 0.25 * intelligibility
        + 0.25 * semantic_fidelity
        + 0.15 * naturalness
    )

    return {
        "timing_accuracy": round(normalize_to_unit(timing_accuracy), 3),
        "intelligibility": round(normalize_to_unit(intelligibility), 3),
        "semantic_fidelity": round(normalize_to_unit(semantic_fidelity), 3),
        "naturalness": round(normalize_to_unit(naturalness), 3),
        "overall": round(normalize_to_unit(overall), 3),
    }