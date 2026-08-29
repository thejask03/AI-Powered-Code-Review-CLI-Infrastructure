# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ai-review` — a schema-enforced AI code review CLI for GitHub PRs. It fetches a PR diff, chunks it at function/class boundaries, sends each chunk to Claude for review, verifies each finding with a second LLM pass, grounds findings against the actual repo content, and posts a single idempotent summary comment.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run unit + contract tests with coverage (what CI runs)
pytest tests/unit tests/contract -v --cov=ai_review --cov-report=term-missing

# Run a single test file / test
pytest tests/unit/test_chunker.py -v
pytest tests/unit/test_chunker.py::test_name -v

# Validate a hand-written findings JSON against the schema — no LLM, no GitHub
ai-review local examples/sample-findings.json --min-severity warning

# Exercise the full LLM client call path with zero API calls / no key required
ai-review dry-run src/ai_review/cli.py

# Review a real PR and post a summary comment (needs GITHUB_TOKEN + ANTHROPIC_API_KEY)
ai-review pr --repo owner/name --pr-number 42

# Preview a real PR's review without posting anything
ai-review pr --repo owner/name --pr-number 42 --no-post

# Recommended first-time validation against a real PR: fabricated findings, no post
ai-review pr --repo owner/name --pr-number 42 --dry-run --no-post
```

`tests/evals/` (curated diffs with expected finding categories, graded not pass/fail) and parts of `tests/contract/` are scaffolded but not wired into CI — they run separately since they call a real or recorded model.

## Architecture

Linear pipeline, orchestrated by [pipeline.py](src/ai_review/pipeline.py) — the only module that knows about all the others. Every other module builds and is tested in isolation:

```
diff/fetcher.py → diff/chunker.py → llm/client.py → verify/grounding.py → github/poster.py
```

1. **Fetch** ([diff/fetcher.py](src/ai_review/diff/fetcher.py)) — shells out to `git diff`, not the GitHub REST API, because the chunker needs the full checked-out working tree anyway. Scoped to `since_sha..head` on re-runs (only new commits reviewed), or `base..head` on a first review. Requires a non-shallow checkout (`fetch-depth: 0`) — shallow clones fail with "unknown revision".

2. **Chunk** ([diff/chunker.py](src/ai_review/diff/chunker.py)) — parses the *full file* at HEAD (not just the diff hunk) so the model never sees a function fragment with no signature. Two tiers: tree-sitter boundary detection (expands each changed line to its smallest enclosing function/class node) falling back to a blank-line/indentation heuristic when no parser is available for the language. Oversized nodes are split into child boundary nodes where possible, else truncated with `is_partial=True` so the prompt can warn the model it's seeing an incomplete view. The unified-diff parser itself (`_parse_unified_diff`) is pure text, no filesystem or third-party deps — fully unit-testable everywhere.

3. **Review** ([llm/client.py](src/ai_review/llm/client.py)) — one Claude call per chunk using the Messages API's structured outputs (`output_config.format`) so the response is schema-constrained server-side against `ChunkReviewResult`. 3 retries with exponential backoff on rate limits / 5xx; non-retryable errors (bad request, auth) raise immediately. A chunk that exhausts retries fails soft — empty findings, `is_failed=True` — rather than aborting the whole PR review.

4. **Verify** ([llm/client.py](src/ai_review/llm/client.py) `verify_finding`) — a second, cheaper LLM call per finding asking "is this actually correct given this code?" **Fail-closed**: if verification can't complete after retries, the finding is rejected, not passed through.

5. **Ground** ([verify/grounding.py](src/ai_review/verify/grounding.py)) — the primary defense against confidently-wrong output; schema validity says nothing about truth. Per finding, in order: file exists? → line in range? → `code_snippet` actually found near that location (small tolerance window; mismatch downweights confidence rather than dropping, since paraphrasing isn't proof of a wrong finding)? → optional verifier hook rejects outright. The verifier is injected as a callable from `pipeline.py`, not implemented in this module — grounding is pure content/location checking, not something that should own a live model call.

6. **Post** ([github/poster.py](src/ai_review/github/poster.py)) — one summary comment per PR, identified by a hidden HTML marker (`SUMMARY_MARKER`); found-and-edited on every run rather than duplicated. The reviewed commit SHA is embedded behind its own hidden marker so the next run's fetch stage knows where to scope `since_sha` from. A third marker embeds the full (unfiltered) findings list as base64-encoded JSON — base64 because a finding's free-text `message` could itself contain `-->` and corrupt the comment if embedded raw — so `pipeline.py` can re-ground prior findings on the next run (see carry-forward below). Talks to GitHub through a duck-typed `pull` object (`get_issue_comments()`, `create_issue_comment()`, comment `.body`/`.edit()`) rather than importing PyGithub types directly, so it's testable with a fake PR object and no network. Inline per-line comments are a deliberate v2 gap — they'd need their own dedup story.

**Since-SHA finding carry-forward**: on a since-SHA-scoped re-run, `pipeline.py` reads the prior run's findings back off the existing summary comment (`extract_prior_findings`) and re-grounds them against the *current* repo state via the same `ground_findings` call used for this run's new findings — a prior finding whose file/line no longer checks out is dropped like any other. Deduped on `(file, line_start)`: a new finding at the same location wins over a carried one. This runs even when the scoped diff is empty (an unrelated push that touches none of the previously-flagged files no longer silently wipes them from the summary) — the only case that skips it is a first review, where there's nothing to carry.

Schema contract ([schema/models.py](src/ai_review/schema/models.py)): `Finding` → `ChunkReviewResult` (one LLM call's output) → `ReviewResult` (final aggregated PR output). Every module imports these shapes from here rather than redefining them. `Finding.code_snippet` and `.confidence` are load-bearing for grounding, not optional extras — the prompts in [llm/prompts.py](src/ai_review/llm/prompts.py) ask for them explicitly.

### Known gaps (see README "Known limitations" and inline module docstrings for full detail)

- `tests/contract/` and `tests/evals/` are scaffolded, largely empty.
- Fork PRs need `pull_request_target` with an explicit checkout `ref:` — see README's "Reviewing PRs from forks".

### Security note

Diff content in [llm/prompts.py](src/ai_review/llm/prompts.py) is attacker-controlled for external PRs. It's kept strictly in a data role (delimited, with explicit instructions not to follow imperative text inside it) — preserve this when touching prompt construction.

### Lazy imports are deliberate

`anthropic` and `github` (PyGithub) are imported lazily inside functions, not at module top level — so that `ai-review local` and `ai-review dry-run` never require those packages or their API keys to be installed/set. Keep new heavy/optional dependencies lazy the same way unless a command genuinely always needs them.
