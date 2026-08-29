"""Posts review results back to the PR.

Idempotency design: one summary comment per PR, identified by a hidden
HTML marker (SUMMARY_MARKER). On every run: find the existing marked
comment and edit it in place, or create one if this is the first run.
Never create a second summary comment — that's what makes re-runs on
new pushes update instead of duplicate.

The comment body also embeds the reviewed commit SHA behind its own
hidden marker, so diff/fetcher.py can later scope a run to only the
commits pushed since the last review (extract_last_reviewed_sha).

It also embeds the full (unfiltered) findings list as base64-encoded
JSON behind a third marker (extract_prior_findings), so pipeline.py can
re-ground still-valid findings from earlier commits that have since
scrolled out of a since-SHA-scoped diff. Base64, not raw JSON: a
finding's message/suggested_fix is free text that could itself contain
the literal "-->" and prematurely close the HTML comment, corrupting
the rest of the markdown — the SHA marker gets away without this only
because SHAs are a constrained [0-9a-fA-F]+ alphabet.

Inline per-line comments are a deliberate v2 addition, not built here —
they need their own dedup story (tracking posted finding hashes so
unchanged findings aren't re-posted on every commit), which is a
meaningfully different problem from "one comment, keep it updated".

Talks to GitHub through a duck-typed `pull` object (needs
get_issue_comments() and create_issue_comment(body); comments need
.body and .edit(body)) rather than importing PyGithub types directly —
that's what makes this testable with a fake PR object, no network or
PyGithub install required. A real github.PullRequest.PullRequest
satisfies this interface as-is.
"""

from __future__ import annotations

import base64
import json
import re

from ai_review.schema.models import Finding, ReviewResult, Severity

SUMMARY_MARKER = "<!-- ai-review:summary -->"
_SHA_MARKER_PATTERN = re.compile(r"<!-- ai-review:last-reviewed-sha:([0-9a-fA-F]+) -->")
_FINDINGS_MARKER_PATTERN = re.compile(r"<!-- ai-review:findings:([A-Za-z0-9+/=]*) -->")


def build_summary_body(result: ReviewResult) -> str:
    """Pure formatting, no I/O — testable without a GitHub connection."""
    lines = [SUMMARY_MARKER, "", "## AI Code Review", ""]

    if not result.findings:
        lines.append("No issues found.")
    else:
        by_severity: dict[Severity, list] = {}
        for finding in result.findings:
            by_severity.setdefault(finding.severity, []).append(finding)

        # Most severe first: list(Severity) is INFO..CRITICAL in
        # ascending order (see schema/models.py), so reverse it.
        for severity in reversed(list(Severity)):
            findings = by_severity.get(severity)
            if not findings:
                continue
            lines.append(f"### {severity.value.title()} ({len(findings)})")
            for f in findings:
                lines.append(f"- **{f.file}:{f.line_start}** — {f.message}")
                if f.suggested_fix:
                    lines.append(f"  - suggested fix: `{f.suggested_fix}`")
            lines.append("")

    summary_line = f"_{len(result.findings)} finding(s) across {result.chunks_reviewed} reviewed chunk(s)"
    if result.chunks_skipped:
        summary_line += f", {result.chunks_skipped} skipped"
    summary_line += "._"
    lines.append(summary_line)
    lines.append(f"<!-- ai-review:last-reviewed-sha:{result.commit_sha} -->")

    # Unfiltered findings (not the severity-filtered set actually shown
    # above) — a finding just below today's min-severity could still
    # matter for a future run's filter, so carry everything forward.
    findings_json = json.dumps([f.model_dump(mode="json") for f in result.findings])
    encoded = base64.b64encode(findings_json.encode("utf-8")).decode("ascii")
    lines.append(f"<!-- ai-review:findings:{encoded} -->")

    return "\n".join(lines)


def extract_last_reviewed_sha(comment_body: str) -> str | None:
    match = _SHA_MARKER_PATTERN.search(comment_body)
    return match.group(1) if match else None


def extract_prior_findings(comment_body: str) -> list[Finding]:
    """Findings embedded in a prior run's summary comment, for
    pipeline.py to re-ground against current repo state. Fail-soft: a
    missing or corrupted marker returns an empty list rather than
    raising — a malformed marker must not crash the run, matching this
    codebase's posture at every other LLM/verify boundary.
    """
    match = _FINDINGS_MARKER_PATTERN.search(comment_body)
    if not match:
        return []
    try:
        raw = base64.b64decode(match.group(1)).decode("utf-8")
        return [Finding.model_validate(f) for f in json.loads(raw)]
    except Exception:
        return []


def find_summary_comment(pull):
    """Returns the existing marked summary comment, or None if this PR
    hasn't been reviewed yet.
    """
    for comment in pull.get_issue_comments():
        if SUMMARY_MARKER in comment.body:
            return comment
    return None


def post_summary(pull, result: ReviewResult):
    """Find-or-create the summary comment for this PR. Safe to call on
    every run — repeated calls edit the same comment rather than
    accumulating new ones.
    """
    body = build_summary_body(result)
    existing = find_summary_comment(pull)
    if existing is not None:
        existing.edit(body)
        return existing
    return pull.create_issue_comment(body)
