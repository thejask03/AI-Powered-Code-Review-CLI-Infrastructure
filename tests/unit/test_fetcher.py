import subprocess

import pytest

from ai_review.diff.fetcher import GitDiffError, fetch_local_diff, fetch_pr_diff


def _git(repo_root, *args):
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def _init_repo(repo_root):
    _git(repo_root, "init", "-q")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")


@pytest.fixture
def repo(tmp_path):
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


def test_fetch_pr_diff_shows_changes_between_base_and_head(repo):
    repo_root, base_sha, head_sha = repo
    diff = fetch_pr_diff(repo_root, base_sha, head_sha)
    assert "+def added():" in diff
    assert "+    return 1" in diff


def test_fetch_pr_diff_scoped_to_since_sha_only_shows_new_commits(repo):
    repo_root, base_sha, head_sha = repo

    (repo_root / "app.py").write_text(
        "def existing():\n    pass\n\n\ndef added():\n    return 1\n\n\ndef newer():\n    return 2\n"
    )
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-q", "-m", "third")
    newest_sha = _git(repo_root, "rev-parse", "HEAD")

    # Scoped since head_sha: should only see "newer" as an addition.
    # "def added" still appears as unchanged *context* around the real
    # change, which is normal git diff behavior — check for the '+'
    # prefix specifically, not bare presence of the text.
    scoped_diff = fetch_pr_diff(repo_root, base_sha, newest_sha, since_sha=head_sha)
    assert "+def newer():" in scoped_diff
    assert "+def added():" not in scoped_diff

    # Unscoped (first review): should see both
    full_diff = fetch_pr_diff(repo_root, base_sha, newest_sha)
    assert "+def added():" in full_diff
    assert "+def newer():" in full_diff


def test_fetch_pr_diff_raises_clear_error_on_missing_commit(repo):
    repo_root, base_sha, head_sha = repo
    with pytest.raises(GitDiffError, match="fetch-depth"):
        fetch_pr_diff(repo_root, "0000000000000000000000000000000000000000", head_sha)


def test_fetch_local_diff_shows_uncommitted_changes(repo):
    repo_root, base_sha, head_sha = repo
    _git(repo_root, "branch", "main", head_sha)  # base_ref for local diff

    (repo_root / "app.py").write_text(
        (repo_root / "app.py").read_text() + "\n\ndef uncommitted():\n    pass\n"
    )
    diff = fetch_local_diff(repo_root, base_ref="main")
    assert "+def uncommitted():" in diff
