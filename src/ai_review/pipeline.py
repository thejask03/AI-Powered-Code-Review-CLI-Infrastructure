"""Orchestrates fetch -> chunk -> review -> ground -> post.

Deliberately the only module that knows about all the others — every
other module builds and tests in isolation; this just wires them
together now that each stage is real.

Since-SHA carry-forward: when a run is since_sha-scoped (only new
commits reviewed), this run's diff alone won't mention a still-valid
finding from an earlier commit once it's scrolled out of view — most
obviously when a later push doesn't touch the flagged file at all, up
to and including an empty diff wiping the summary clean. Fixed by
re-grounding the prior run's findings (read back off the existing
summary comment via github/poster.py's extract_prior_findings) against
the *current* repo_root state on every since-SHA-scoped run, and
merging survivors with this run's new findings. Deduped on
(file, line_start): a new finding at the same location wins over a
carried one, since it reflects current code. Only runs when since_sha
is set and there's something to carry — a first review has nothing to
carry forward, and no LLM client is constructed in that case.
"""

from __future__ import annotations

from pathlib import Path

from ai_review.diff.chunker import chunk_diff
from ai_review.diff.fetcher import fetch_pr_diff
from ai_review.github.poster import (
    extract_last_reviewed_sha,
    extract_prior_findings,
    find_summary_comment,
    post_summary,
)
from ai_review.llm.client import DEFAULT_MODEL, LLMClient
from ai_review.schema.models import Finding, ReviewResult, Severity
from ai_review.verify.grounding import ground_findings


def run(
    pull,
    repo_root: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    min_severity: Severity = Severity.SUGGESTION,
    dry_run: bool = False,
    post: bool = True,
) -> ReviewResult:
    """Review one PR end to end and, by default, post the result.

    `pull` needs `.number`, `.base.sha`, `.head.sha`, and the poster's
    duck-typed comment interface — a real PyGithub PullRequest, or a
    test double. `repo_root` must be a checkout at `pull.head.sha`.

    `post=False` runs the full pipeline — including hitting the real
    GitHub API to read the PR's diff and existing comments — but skips
    writing anything back. Useful for previewing against a real PR
    before you're ready to comment on it. Note this still makes read
    calls against GitHub even with post=False; only `dry_run=True`
    avoids the LLM call itself.

    Returns the full (unfiltered) ReviewResult; the posted comment (if
    any) is filtered to `min_severity` so low-severity noise doesn't
    show up in the PR by default, but callers get everything back for
    inspection or logging.
    """
    repo_root = Path(repo_root)

    since_sha: str | None = None
    prior_findings: list[Finding] = []
    existing_comment = find_summary_comment(pull)
    if existing_comment is not None:
        since_sha = extract_last_reviewed_sha(existing_comment.body)
        prior_findings = extract_prior_findings(existing_comment.body)

    def carry_forward(new_findings: list[Finding], client: LLMClient) -> list[Finding]:
        seen = {(f.file, f.line_start) for f in new_findings}
        carried = ground_findings(prior_findings, repo_root=repo_root, verifier=client.verify_finding).grounded
        return new_findings + [f for f in carried if (f.file, f.line_start) not in seen]

    diff = fetch_pr_diff(repo_root, base_sha=pull.base.sha, head_sha=pull.head.sha, since_sha=since_sha)

    if not diff.strip():
        # Nothing new since the last review (or an empty PR). Still post
        # when allowed, so the summary's embedded SHA marker advances —
        # otherwise the next run would think it still needs to review
        # from since_sha. Prior findings still carry forward: an
        # unrelated push that touches none of the previously-flagged
        # files shouldn't silently wipe them from the summary.
        findings = []
        if since_sha is not None and prior_findings:
            findings = carry_forward([], LLMClient(model=model, dry_run=dry_run))
        result = ReviewResult(pr_number=pull.number, commit_sha=pull.head.sha, findings=findings, chunks_reviewed=0)
        if post:
            post_summary(pull, result)
        return result

    chunks = chunk_diff(diff, repo_root=repo_root)

    client = LLMClient(model=model, dry_run=dry_run)
    all_findings: list[Finding] = []
    chunk_results = []
    for chunk in chunks:
        chunk_result = client.review_chunk(chunk)
        chunk_results.append(chunk_result)
        all_findings.extend(chunk_result.findings)

    grounding = ground_findings(all_findings, repo_root=repo_root, verifier=client.verify_finding)
    chunks_skipped = (
        sum(1 for c in chunks if c.is_partial)
        + sum(1 for r in chunk_results if r.is_failed)
    )

    findings = grounding.grounded
    if since_sha is not None and prior_findings:
        findings = carry_forward(findings, client)

    result = ReviewResult(
        pr_number=pull.number,
        commit_sha=pull.head.sha,
        findings=findings,
        chunks_reviewed=len(chunks),
        chunks_skipped=chunks_skipped,
    )

    posted = ReviewResult(
        pr_number=result.pr_number,
        commit_sha=result.commit_sha,
        findings=result.by_severity(min_severity),
        chunks_reviewed=result.chunks_reviewed,
        chunks_skipped=result.chunks_skipped,
    )
    if post:
        post_summary(pull, posted)

    return result
