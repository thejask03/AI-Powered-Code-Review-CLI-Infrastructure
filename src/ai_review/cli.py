"""CLI entry point.

Three commands:
- `ai-review local` — validate/print a hand-written or fixture
  ReviewResult JSON. No LLM, no GitHub. Good for testing schema and
  report formatting.
- `ai-review dry-run` — exercise the LLM client's full call path with
  zero network calls. Good for validating wiring before spending
  tokens.
- `ai-review pr` — the real thing: reviews an actual PR and posts a
  summary comment. Needs GITHUB_TOKEN and (unless --dry-run)
  ANTHROPIC_API_KEY. This is what action.yml invokes in CI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from pydantic import ValidationError

from ai_review.diff.chunker import Chunk
from ai_review.llm.client import DEFAULT_MODEL, LLMClient
from ai_review.schema.models import ReviewResult, Severity

app = typer.Typer(add_completion=False)


@app.command()
def local(
    findings_json: Path = typer.Argument(
        ..., help="Path to a JSON file matching the ReviewResult schema"
    ),
    min_severity: Severity = typer.Option(
        Severity.SUGGESTION, help="Only show findings at or above this severity"
    ),
) -> None:
    """Validate and print a ReviewResult from a local JSON file.

    Useful for testing the schema and report formatting without
    calling an LLM: point this at a hand-written or fixture JSON file.
    """
    try:
        raw = json.loads(findings_json.read_text())
        result = ReviewResult.model_validate(raw)
    except ValidationError as e:
        typer.secho(f"Schema validation failed:\n{e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    findings = result.by_severity(min_severity)
    if not findings:
        typer.secho("No findings at or above threshold.", fg=typer.colors.GREEN)
        return

    for f in findings:
        color = {
            Severity.INFO: typer.colors.BLUE,
            Severity.SUGGESTION: typer.colors.CYAN,
            Severity.WARNING: typer.colors.YELLOW,
            Severity.CRITICAL: typer.colors.RED,
        }[f.severity]
        typer.secho(f"[{f.severity.value.upper()}] {f.file}:{f.line_start}", fg=color, bold=True)
        typer.echo(f"  {f.message}")
        if f.suggested_fix:
            typer.echo(f"  suggested fix: {f.suggested_fix}")
        typer.echo(f"  confidence: {f.confidence:.2f}")
        typer.echo("")

    typer.echo(f"{len(findings)} finding(s) shown, {len(result.findings)} total.")


@app.command(name="dry-run")
def dry_run(
    file: Path = typer.Argument(..., help="A source file to send through the LLM client in dry-run mode"),
) -> None:
    """Exercise the LLM client's full call path — prompt construction,
    schema passing, response parsing — with zero network calls and no
    API key required. Returns a fabricated but schema-valid response.

    This validates wiring, not review quality: it does not tell you
    whether the model produces good findings, only that the pipeline
    from chunk to validated ChunkReviewResult doesn't break. Use the
    eval suite (tests/evals) once you're ready to test against a real
    or recorded model.
    """
    content = file.read_text()
    chunk = Chunk(
        chunk_id="dry-run-0",
        file=str(file),
        start_line=1,
        end_line=max(len(content.splitlines()), 1),
        content=content,
    )
    client = LLMClient(dry_run=True)
    result = client.review_chunk(chunk)

    typer.secho(f"[DRY RUN] chunk_id={result.chunk_id}  (no API call made)", fg=typer.colors.MAGENTA, bold=True)
    for f in result.findings:
        typer.echo(f"  {f.file}:{f.line_start} [{f.severity.value}] {f.message}")
    typer.echo(f"\n{len(result.findings)} finding(s), all fabricated by dry-run mode.")


@app.command(name="pr")
def review_pr(
    repo: str = typer.Option(..., help="owner/repo"),
    pr_number: int = typer.Option(..., "--pr-number"),
    repo_root: Path = typer.Option(Path("."), help="Checkout root — must be at the PR's head commit"),
    min_severity: Severity = typer.Option(Severity.SUGGESTION, help="Filter applied to the posted comment only"),
    model: str = typer.Option(DEFAULT_MODEL),
    dry_run: bool = typer.Option(False, help="Skip real LLM calls; fabricated findings only"),
    post: bool = typer.Option(True, "--post/--no-post", help="Post the summary comment. Use --no-post to preview against a real PR without writing to it."),
) -> None:
    """Review a real PR and (by default) post the results as a summary
    comment.

    Requires GITHUB_TOKEN in the environment always, and
    ANTHROPIC_API_KEY unless --dry-run is set. First time pointing this
    at a real PR, run with `--dry-run --no-post` to validate the whole
    pipeline — fetch, chunk, fabricated review, ground — without
    spending tokens or writing a comment; then `--no-post` alone once
    you're ready for real findings but not ready to post; then neither
    flag for the real thing.
    """
    import os

    from github import Auth, Github  # lazy: `local`/`dry-run` shouldn't require PyGithub at import time

    from ai_review.pipeline import run as run_pipeline

    # Only configured here, not at module import time — importing ai_review.cli
    # as a library must not clobber a caller's own logging setup, and
    # `local`/`dry-run` output should stay clean for scripting.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        typer.secho("GITHUB_TOKEN not set.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    gh_repo = Github(auth=Auth.Token(token)).get_repo(repo)
    pull = gh_repo.get_pull(pr_number)

    result = run_pipeline(
        pull, repo_root=repo_root, model=model, min_severity=min_severity, dry_run=dry_run, post=post
    )

    typer.secho(f"Reviewed PR #{result.pr_number} @ {result.commit_sha[:8]}", fg=typer.colors.GREEN, bold=True)
    summary = f"{len(result.findings)} finding(s), {result.chunks_reviewed} chunk(s) reviewed"
    if result.chunks_skipped:
        summary += f", {result.chunks_skipped} partial"
    if not post:
        summary += "  [not posted — pass without --no-post to write the comment]"
    typer.echo(summary)

    for f in result.findings:
        typer.echo(f"  {f.file}:{f.line_start} [{f.severity.value}] {f.message}")


if __name__ == "__main__":
    app()
