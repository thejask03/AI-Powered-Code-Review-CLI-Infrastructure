"""Fetches the PR diff to review.

Design decision: shell out to `git diff`, not the GitHub REST API's diff
media type. The repo is already checked out at the PR head in CI —
chunker.py needs the working tree anyway to read full file content for
boundary detection — so a local `git diff <base>..<head>` is simpler
than juggling PyGithub plus a custom Accept header, and costs no extra
API calls or rate-limit budget.

Gotcha worth flagging loudly, because it's the single most common way
this breaks in practice: actions/checkout defaults to a shallow clone
(fetch-depth: 1), which only has the tip commit. `git diff` against any
other commit fails with "unknown revision" unless the consuming
workflow sets `fetch-depth: 0` (or explicitly fetches the base/since
SHA). This needs to be documented in the workflow YAML, not just here —
put a comment on the checkout step in action.yml / example workflows.

Since-last-review scoping: pipeline.py should read the last-reviewed
SHA off the existing summary comment (see github/poster.py's
extract_last_reviewed_sha) and pass it as `since_sha`, so only new
commits get reviewed on repeat runs. When there's no prior review, it
falls back to diffing against the PR's base branch — the whole PR,
once.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitDiffError(RuntimeError):
    """`git diff` failed — most often the shallow-clone gotcha above.
    Raised rather than swallowed: a silently empty diff would look like
    "nothing changed" instead of "we couldn't check", and those need
    very different handling upstream.
    """


def _run_git_diff(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "diff", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitDiffError(
            f"git diff {' '.join(args)} failed in {repo_root}: {result.stderr.strip()}\n"
            "If this is running in GitHub Actions, check that actions/checkout "
            "used fetch-depth: 0 (or fetched the specific SHA) — a shallow "
            "clone won't have the base/since commit available."
        )
    return result.stdout


def fetch_pr_diff(
    repo_root: Path | str,
    base_sha: str,
    head_sha: str,
    since_sha: str | None = None,
) -> str:
    """Diff for review. Scoped to `since_sha..head_sha` when a previous
    review exists (only new commits get reviewed); otherwise
    `base_sha..head_sha` (the full PR diff, on the first review).
    """
    from_ref = since_sha or base_sha
    return _run_git_diff(Path(repo_root), from_ref, head_sha)


def fetch_local_diff(repo_root: Path | str = ".", base_ref: str = "main") -> str:
    """Working-tree diff against a base ref — for local dev/testing, no
    PR or commit SHAs involved, just whatever's changed locally.
    """
    return _run_git_diff(Path(repo_root), base_ref)
