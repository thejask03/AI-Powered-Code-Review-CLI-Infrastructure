# Contributing

See [CLAUDE.md](CLAUDE.md) for architecture and the full command reference.

## Setup

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/unit -v
pytest tests/unit tests/contract -v   # what CI runs
```

## Testing changes against a real PR without spending tokens or posting

Point `ai-review pr` at a real PR in three stages, each one adding a bit more
risk than the last:

```bash
# 1. Fabricated findings, nothing written to the PR — validates the whole
#    pipeline wiring (fetch, chunk, ground) with zero API calls.
ai-review pr --repo owner/name --pr-number 42 --dry-run --no-post

# 2. Real LLM calls, still nothing written — once you're ready to see
#    real findings but not ready to comment on the PR.
ai-review pr --repo owner/name --pr-number 42 --no-post

# 3. The real thing.
ai-review pr --repo owner/name --pr-number 42
```

Requires `GITHUB_TOKEN` always, and `ANTHROPIC_API_KEY` unless `--dry-run` is
set — see [.env.example](.env.example).
