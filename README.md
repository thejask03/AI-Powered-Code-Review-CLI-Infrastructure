# ai-review

AI-powered, schema-enforced code review CLI for GitHub PRs.

## What it does

Point it at a GitHub PR and it:
1. **Fetches** the diff and scoped to new commits if it's a re-run
2. **Chunks** the diff at function/class boundaries using tree-sitter (falls back to a heuristic for unsupported languages)
3. **Reviews** each chunk with Claude LLM, producing schema-constrained findings
4. **Verifies** each finding with a second cheaper LLM pass — fail-closed, so unverifiable findings are rejected
5. **Grounds** every finding against the actual repo (file exists? line in range? snippet matches?) — drops or downweights anything that doesn't check out.
On a re-run, also re-grounds findings carried forward from the prior review so they don't silently disappear once their file scrolls out of the new diff
6. **Posts** a single, independent summary comment on the PR

## Architecture

```
diff/fetcher.py → diff/chunker.py → llm/client.py → verify/grounding.py → github/poster.py
     │                  │                  │                      │
     │           tree-sitter AST       review call +           file/line/snippet
     │           boundary detection    verify call for            checks + LLM
     │           + heuristic fallback   matching structured output   verification
     │
  git diff (scoped to since-last-review SHA when available)
```

### Pipeline stages

| Stage | Module | What it does |
|-------|--------|-------------|
| Fetch | `diff/fetcher.py` | `git diff` between base and head, scoped to new commits on re-runs |
| Chunk | `diff/chunker.py` | Tree-sitter boundary detection (tier 1) → blank-line heuristic (tier 2) → sub-chunking for oversized nodes |
| Review | `llm/client.py` | One Claude call per chunk, schema-constrained output, 3 retries with exponential backoff |
| Verify | `llm/client.py` | Second LLM call per finding — "is this actually valid?" Fail-closed after 3 retries |
| Ground | `verify/grounding.py` | File exists? Line in range? Snippet matches? Verifier approves? |
| Post | `github/poster.py` | Idempotent summary comment via hidden HTML markers |

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run tests (what CI runs)
pytest tests/unit tests/contract -v --cov=ai_review --cov-report=term-missing

# Validate a hand-written findings JSON
ai-review local examples/sample-findings.json --min-severity warning

# Exercise the full pipeline with zero API calls
ai-review dry-run src/ai_review/cli.py

# Review a real PR (needs GITHUB_TOKEN + ANTHROPIC_API_KEY)
ai-review pr --repo owner/name --pr-number 42

# Preview without posting
ai-review pr --repo owner/name --pr-number 42 --no-post
```

## GitHub Action

Add to any repo's `.github/workflows/`:

```yaml
name: AI Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: thejask03/AI-Powered-Code-Review-CLI-Infrastructure@v0.1.0
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

Add `ANTHROPIC_API_KEY` to your repo's **Settings → Secrets → Actions**. `GITHUB_TOKEN` is provided automatically.

### Reviewing PRs from forks

`pull_request` doesn't give a workflow access to secrets for PRs opened from
a fork, so the default snippet above won't have `ANTHROPIC_API_KEY` available
on fork PRs. Use `pull_request_target` instead, with an explicit `ref:` on
the checkout step:

```yaml
name: AI Code Review
on:
  pull_request_target:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}   # REQUIRED under pull_request_target
          fetch-depth: 0                                    # still required, see diff/fetcher.py
      - uses: thejask03/AI-Powered-Code-Review-CLI-Infrastructure@v0.1.0
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

The `ref:` override is required, not optional: `pull_request_target` runs
using the *base* repo's workflow file and checks out the *base* ref by
default. Without pointing `actions/checkout` explicitly at
`github.event.pull_request.head.sha`, it silently checks out the target
branch instead of the fork's changes, and the review runs against the wrong
code with no error raised.

Security note: `pull_request_target` makes secrets available even for PRs
from forks, which is exactly why the explicit-ref checkout above matters.
It's safe to point this action at untrusted fork content because nothing in
this pipeline executes checked-out file content — `diff/fetcher.py` only
shells out to `git diff` (read-only), `diff/chunker.py` only parses with
tree-sitter, and `verify/grounding.py` only reads file text.

## Known limitations

- **No contract or eval tests yet.** `tests/contract/` and `tests/evals/` are scaffolded and empty.

## Testing strategy

- `tests/unit/` — 68 tests, deterministic, no LLM calls
- `tests/contract/` — pipeline tested against recorded LLM responses (planned)
- `tests/evals/` — curated diffs with expected finding categories; graded, not pass/fail (planned)
