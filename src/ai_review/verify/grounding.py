"""Grounds LLM findings against the actual repo content.

This is the primary defense against confidently-wrong output. Schema
validation (already enforced by the LLM client) guarantees a Finding has
the right shape; it says nothing about whether it's true. For every
finding, in order:

1. Does `file` exist in the repo? If not, dropped outright — nothing to
   check further, and there's no plausible fix.
2. Is `line_start` within that file's actual line count? If not, dropped
   outright, same reasoning.
3. Does `code_snippet` actually appear near that location (a small line
   window, to tolerate the model being off by a line or two)? If not,
   the *location* might still be valid but we can no longer confirm the
   *content* — this is downweighted (confidence reduced), not dropped,
   since a snippet mismatch alone doesn't prove the finding is wrong
   (could be paraphrasing or minor formatting). The caller's severity/
   confidence filtering makes the final call on whether to show it.
4. Optional `verifier` hook: a second, cheaper LLM call re-checking
   "is this finding actually correct given this code?" Left as an
   injectable callable rather than implemented here — it needs a live
   model call, which isn't something a grounding/validation module
   should own directly. Wire a real implementation in llm/client.py and
   pass it in from pipeline.py.

`repo_root` must be the checkout at the commit the findings were
generated against — same requirement as diff/chunker.py's chunk_diff,
since we need the actual file content, not just the diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ai_review.schema.models import Finding

# Lines of slack allowed around a finding's reported location when
# searching for its code_snippet. Models occasionally report a line
# number a line or two off even when the underlying claim is correct.
LINE_TOLERANCE = 2

# Confidence multiplier applied when location is valid but the snippet
# text doesn't match. Downweight, don't drop — see module docstring.
SNIPPET_MISMATCH_PENALTY = 0.5


@dataclass
class GroundingReport:
    """What happened to every finding, not just the survivors — useful
    for debugging prompt/model changes and for the eval suite later.
    """

    grounded: list[Finding] = field(default_factory=list)  # what the caller should consider posting
    downweighted: list[Finding] = field(default_factory=list)  # subset of grounded, confidence reduced
    dropped: list[tuple[Finding, str]] = field(default_factory=list)  # excluded entirely, with reason


def _normalize(text: str) -> str:
    """Collapse whitespace so minor formatting differences (extra spaces,
    tabs vs spaces) don't cause false snippet-mismatch downweights.
    """
    return " ".join(text.split())


def ground_findings(
    findings: list[Finding],
    repo_root: Path | str,
    verifier: Callable[[Finding, str], bool] | None = None,
) -> GroundingReport:
    repo_root = Path(repo_root)
    report = GroundingReport()

    for finding in findings:
        file_path = repo_root / finding.file
        if not file_path.exists():
            report.dropped.append((finding, "file_not_found"))
            continue

        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            report.dropped.append((finding, "file_unreadable"))
            continue

        if finding.line_start < 1 or finding.line_start > len(lines):
            report.dropped.append((finding, "line_out_of_range"))
            continue

        window_start = max(1, finding.line_start - LINE_TOLERANCE)
        window_end = min(len(lines), max(finding.line_end, finding.line_start) + LINE_TOLERANCE)
        context = "\n".join(lines[window_start - 1 : window_end])

        if _normalize(finding.code_snippet) not in _normalize(context):
            downweighted = finding.model_copy(
                update={"confidence": finding.confidence * SNIPPET_MISMATCH_PENALTY}
            )
            report.grounded.append(downweighted)
            report.downweighted.append(downweighted)
            continue

        if verifier is not None and not verifier(finding, context):
            report.dropped.append((finding, "verifier_rejected"))
            continue

        report.grounded.append(finding)

    return report
