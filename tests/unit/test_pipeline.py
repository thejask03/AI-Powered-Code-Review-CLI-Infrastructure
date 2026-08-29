from ai_review.diff.chunker import Chunk
from ai_review.github.poster import SUMMARY_MARKER, build_summary_body, extract_last_reviewed_sha
from ai_review.pipeline import run
from ai_review.schema.models import ChunkReviewResult, Severity
from conftest import FakeComment, FakePullRequest, _git, make_finding, make_result


def test_first_review_reviews_full_diff_and_posts(git_repo):
    repo_root, base_sha, head_sha = git_repo
    pull = FakePullRequest(base_sha=base_sha, head_sha=head_sha)

    result = run(pull, repo_root, dry_run=True)

    assert result.commit_sha == head_sha
    assert result.chunks_reviewed >= 1
    assert len(pull.created_bodies) == 1
    assert SUMMARY_MARKER in pull.created_bodies[0]
    assert extract_last_reviewed_sha(pull.created_bodies[0]) == head_sha


def test_rerun_scopes_diff_to_since_sha(git_repo, monkeypatch):
    repo_root, base_sha, head_sha = git_repo
    (repo_root / "app.py").write_text(
        (repo_root / "app.py").read_text() + "\n\ndef newer():\n    return 2\n"
    )
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-q", "-m", "third")
    newest_sha = _git(repo_root, "rev-parse", "HEAD")

    comment = FakeComment(build_summary_body(make_result(commit_sha=head_sha, findings=[])))
    pull = FakePullRequest(existing_comments=[comment], base_sha=base_sha, head_sha=newest_sha)

    captured: dict = {}
    import ai_review.pipeline as pipeline_module

    real_chunk_diff = pipeline_module.chunk_diff

    def spy_chunk_diff(diff_text, **kwargs):
        captured["diff_text"] = diff_text
        return real_chunk_diff(diff_text, **kwargs)

    monkeypatch.setattr(pipeline_module, "chunk_diff", spy_chunk_diff)

    result = run(pull, repo_root, dry_run=True)

    # scoped to since_sha=head_sha..newest_sha: only "newer" is new
    assert "+def newer():" in captured["diff_text"]
    assert "+def added():" not in captured["diff_text"]
    # result always reflects the final head, not since_sha or base_sha
    assert result.commit_sha == newest_sha


def test_empty_diff_still_posts_to_advance_sha_marker(git_repo):
    repo_root, base_sha, head_sha = git_repo
    comment = FakeComment(build_summary_body(make_result(commit_sha=head_sha, findings=[])))
    pull = FakePullRequest(existing_comments=[comment], base_sha=base_sha, head_sha=head_sha)

    result = run(pull, repo_root, dry_run=True)

    assert result.findings == []
    assert result.chunks_reviewed == 0
    assert len(comment.edit_calls) == 1
    assert extract_last_reviewed_sha(comment.body) == head_sha


def test_post_false_runs_full_pipeline_but_writes_nothing(git_repo):
    repo_root, base_sha, head_sha = git_repo
    pull = FakePullRequest(base_sha=base_sha, head_sha=head_sha)

    result = run(pull, repo_root, dry_run=True, post=False)

    assert result.chunks_reviewed >= 1
    assert len(result.findings) >= 1
    assert pull.created_bodies == []
    assert pull.get_issue_comments() == []


def test_min_severity_filters_posted_but_not_returned_result(git_repo, monkeypatch):
    repo_root, base_sha, head_sha = git_repo
    pull = FakePullRequest(base_sha=base_sha, head_sha=head_sha)

    chunk = Chunk(chunk_id="app.py:1-1", file="app.py", start_line=1, end_line=1, content="def existing():\n    pass\n")
    monkeypatch.setattr("ai_review.pipeline.chunk_diff", lambda diff_text, **kwargs: [chunk])

    findings = [
        make_finding(file="app.py", line_start=1, line_end=1, severity=Severity.INFO, code_snippet="def existing():", message="minor"),
        make_finding(file="app.py", line_start=1, line_end=1, severity=Severity.CRITICAL, code_snippet="def existing():", message="serious"),
    ]
    monkeypatch.setattr(
        "ai_review.llm.client.LLMClient.review_chunk",
        lambda self, c: ChunkReviewResult(chunk_id=c.chunk_id, findings=findings),
    )

    result = run(pull, repo_root, dry_run=True, min_severity=Severity.CRITICAL)

    assert len(result.findings) == 2  # returned result is unfiltered
    posted_body = pull.created_bodies[0]
    assert "serious" in posted_body
    assert "minor" not in posted_body  # filtered out of the posted comment


def test_chunks_skipped_counts_partial_and_failed_chunks(git_repo, monkeypatch):
    repo_root, base_sha, head_sha = git_repo
    pull = FakePullRequest(base_sha=base_sha, head_sha=head_sha)

    partial_chunk = Chunk(chunk_id="app.py:1-1", file="app.py", start_line=1, end_line=1, content="x", is_partial=True)
    normal_chunk = Chunk(chunk_id="app.py:2-2", file="app.py", start_line=2, end_line=2, content="y", is_partial=False)
    monkeypatch.setattr("ai_review.pipeline.chunk_diff", lambda diff_text, **kwargs: [partial_chunk, normal_chunk])

    def fake_review_chunk(self, c):
        is_failed = c.chunk_id == normal_chunk.chunk_id
        return ChunkReviewResult(chunk_id=c.chunk_id, findings=[], is_failed=is_failed)

    monkeypatch.setattr("ai_review.llm.client.LLMClient.review_chunk", fake_review_chunk)

    result = run(pull, repo_root, dry_run=True)

    assert result.chunks_reviewed == 2
    assert result.chunks_skipped == 2  # one partial + one failed


def test_carry_forward_reappears_when_file_untouched_by_new_diff(git_repo):
    repo_root, base_sha, head_sha = git_repo
    prior = make_finding(file="app.py", line_start=1, line_end=1, code_snippet="def existing():")
    comment = FakeComment(build_summary_body(make_result(commit_sha=head_sha, findings=[prior])))
    pull = FakePullRequest(existing_comments=[comment], base_sha=base_sha, head_sha=head_sha)

    # Third commit touches an unrelated file, never touching app.py.
    (repo_root / "other.py").write_text("def other():\n    return 3\n")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-q", "-m", "third")
    pull.head.sha = _git(repo_root, "rev-parse", "HEAD")

    result = run(pull, repo_root, dry_run=True)

    carried = [f for f in result.findings if f.file == "app.py" and f.line_start == 1]
    assert len(carried) == 1
    assert any(f.file == "other.py" for f in result.findings)


def test_carry_forward_drops_finding_whose_file_was_deleted(git_repo):
    repo_root, base_sha, head_sha = git_repo
    prior = make_finding(file="app.py", line_start=1, line_end=1, code_snippet="def existing():")
    comment = FakeComment(build_summary_body(make_result(commit_sha=head_sha, findings=[prior])))
    pull = FakePullRequest(existing_comments=[comment], base_sha=base_sha, head_sha=head_sha)

    _git(repo_root, "rm", "-q", "app.py")
    _git(repo_root, "commit", "-q", "-m", "delete app.py")
    pull.head.sha = _git(repo_root, "rev-parse", "HEAD")

    result = run(pull, repo_root, dry_run=True)

    assert result.findings == []


def test_dedup_prefers_new_finding_over_carried_duplicate(git_repo, monkeypatch):
    repo_root, base_sha, head_sha = git_repo
    prior = make_finding(file="app.py", line_start=1, line_end=1, code_snippet="def existing():", message="stale")
    comment = FakeComment(build_summary_body(make_result(commit_sha=head_sha, findings=[prior])))
    pull = FakePullRequest(existing_comments=[comment], base_sha=base_sha, head_sha=head_sha)

    chunk = Chunk(chunk_id="app.py:1-1", file="app.py", start_line=1, end_line=1, content="def existing():\n    pass\n")
    monkeypatch.setattr("ai_review.pipeline.chunk_diff", lambda diff_text, **kwargs: [chunk])
    monkeypatch.setattr(
        "ai_review.llm.client.LLMClient.review_chunk",
        lambda self, c: ChunkReviewResult(
            chunk_id=c.chunk_id,
            findings=[make_finding(file="app.py", line_start=1, line_end=1, code_snippet="def existing():", message="fresh")],
        ),
    )

    # Bump the head so the diff isn't empty and the main (non-early-return)
    # branch runs chunk_diff/review_chunk at all.
    (repo_root / "other.py").write_text("def other():\n    return 3\n")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-q", "-m", "third")
    pull.head.sha = _git(repo_root, "rev-parse", "HEAD")

    result = run(pull, repo_root, dry_run=True)

    app_py_findings = [f for f in result.findings if f.file == "app.py" and f.line_start == 1]
    assert len(app_py_findings) == 1
    assert app_py_findings[0].message == "fresh"
