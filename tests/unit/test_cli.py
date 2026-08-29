import json
from pathlib import Path

from typer.testing import CliRunner

from ai_review.cli import app
from ai_review.schema.models import Severity
from conftest import make_finding, make_result

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FINDINGS = REPO_ROOT / "examples" / "sample-findings.json"


# --- local ------------------------------------------------------------


def test_local_valid_json_exits_zero_and_prints_findings():
    result = runner.invoke(app, ["local", str(SAMPLE_FINDINGS)])
    assert result.exit_code == 0
    assert "CRITICAL" in result.output
    # default min-severity is "suggestion", so the sample's one "info"
    # finding (unused import) is filtered out: 2 of 3 shown.
    assert "2 finding(s) shown, 3 total." in result.output


def test_local_min_severity_filters_output():
    result = runner.invoke(app, ["local", str(SAMPLE_FINDINGS), "--min-severity", "critical"])
    assert result.exit_code == 0
    assert "1 finding(s) shown, 3 total." in result.output


def test_local_invalid_json_exits_one_with_schema_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"pr_number": 1}))  # missing required fields
    result = runner.invoke(app, ["local", str(bad)])
    assert result.exit_code == 1
    assert "Schema validation failed" in result.output


# --- dry-run ------------------------------------------------------------


def test_dry_run_exits_zero_with_placeholder_finding():
    target = REPO_ROOT / "src" / "ai_review" / "cli.py"
    result = runner.invoke(app, ["dry-run", str(target)])
    assert result.exit_code == 0
    assert "[DRY RUN]" in result.output
    assert "1 finding(s), all fabricated by dry-run mode." in result.output


# --- pr ------------------------------------------------------------


def _fake_github_factory(pull):
    class _FakeRepo:
        def get_pull(self, pr_number):
            return pull

    class _FakeGithub:
        def __init__(self, *args, **kwargs):
            pass

        def get_repo(self, repo):
            return _FakeRepo()

    return _FakeGithub


def test_pr_requires_github_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = runner.invoke(app, ["pr", "--repo", "owner/name", "--pr-number", "1"])
    assert result.exit_code == 1
    assert "GITHUB_TOKEN not set." in result.output


def test_pr_calls_pipeline_with_correct_kwargs_and_prints_summary(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

    fake_pull = object()
    monkeypatch.setattr("github.Github", _fake_github_factory(fake_pull))

    calls = {}

    def fake_run(pull, repo_root, *, model, min_severity, dry_run, post):
        calls["kwargs"] = dict(
            pull=pull, repo_root=repo_root, model=model, min_severity=min_severity, dry_run=dry_run, post=post
        )
        return make_result(
            pr_number=42,
            commit_sha="deadbeefcafe",
            chunks_reviewed=3,
            chunks_skipped=1,
            findings=[make_finding(message="something to fix")],
        )

    monkeypatch.setattr("ai_review.pipeline.run", fake_run)

    result = runner.invoke(
        app,
        ["pr", "--repo", "owner/name", "--pr-number", "7", "--min-severity", "warning", "--dry-run", "--no-post"],
    )

    assert result.exit_code == 0
    assert "Reviewed PR #42 @ deadbeef" in result.output
    assert "1 finding(s), 3 chunk(s) reviewed, 1 partial" in result.output
    assert "not posted" in result.output
    assert "something to fix" in result.output

    assert calls["kwargs"]["pull"] is fake_pull
    assert calls["kwargs"]["min_severity"] == Severity.WARNING
    assert calls["kwargs"]["dry_run"] is True
    assert calls["kwargs"]["post"] is False
