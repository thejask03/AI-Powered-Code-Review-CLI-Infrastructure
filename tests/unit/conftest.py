"""Shared test fakes/factories for tests/unit.

FakePullRequest/FakeComment satisfy github/poster.py's duck-typed `pull`
interface (get_issue_comments/create_issue_comment; comments need
.body/.edit) plus pipeline.py's direct attribute access (.number,
.base.sha, .head.sha) so the same fakes work for poster-only tests and
full-pipeline tests alike.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ai_review.schema.models import Category, Finding, ReviewResult, Severity


class FakeComment:
    def __init__(self, body: str):
        self.body = body
        self.edit_calls: list[str] = []

    def edit(self, body: str) -> None:
        self.edit_calls.append(body)
        self.body = body


class FakePullRequest:
    def __init__(self, existing_comments=None, number=1, base_sha="base000", head_sha="head000"):
        self._comments = list(existing_comments or [])
        self.created_bodies: list[str] = []
        self.number = number
        self.base = SimpleNamespace(sha=base_sha)
        self.head = SimpleNamespace(sha=head_sha)

    def get_issue_comments(self):
        return list(self._comments)

    def create_issue_comment(self, body: str):
        comment = FakeComment(body)
        self._comments.append(comment)
        self.created_bodies.append(body)
        return comment


def make_result(**overrides) -> ReviewResult:
    defaults = dict(pr_number=1, commit_sha="abc123", chunks_reviewed=2, chunks_skipped=0, findings=[])
    defaults.update(overrides)
    return ReviewResult(**defaults)


def make_finding(**overrides) -> Finding:
    defaults = dict(
        file="src/app.py",
        line_start=10,
        line_end=10,
        severity=Severity.WARNING,
        category=Category.BUG,
        message="Something looks off",
        code_snippet="x = 1",
        confidence=0.8,
    )
    defaults.update(overrides)
    return Finding(**defaults)


def _git(repo_root, *args):
    """Mirrors tests/unit/test_fetcher.py's local helper of the same
    name — shared here so test_pipeline.py can drive the same real
    temp-git-repo pattern.
    """
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def _init_repo(repo_root):
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")


@pytest.fixture
def git_repo(tmp_path):
    """A temp git repo with a base commit (app.py defining `existing()`)
    and a head commit that adds `added()`. Returns (repo_root, base_sha,
    head_sha).
    """
    _init_repo(tmp_path)
    (tmp_path / "app.py").write_text("def existing():\n    pass\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "app.py").write_text("def existing():\n    pass\n\n\ndef added():\n    return 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "head")
    head_sha = _git(tmp_path, "rev-parse", "HEAD")

    return tmp_path, base_sha, head_sha
