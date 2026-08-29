"""LLM client — one call per chunk, schema-constrained output.

Uses the Claude API's structured outputs (output_config.format), now GA:
the response is constrained to ChunkReviewResult's JSON schema by
constrained decoding server-side, not by hoping the prompt worked.

Check https://docs.claude.com/en/docs/build-with-claude/structured-outputs
and https://docs.claude.com/en/docs/about-claude/models before shipping —
both the request shape and which models support structured outputs have
changed before and will again; DEFAULT_MODEL below is a starting point,
not a guarantee.

Remember: schema validity is not correctness. This client's job ends at
"the shape is right" — verify/grounding.py is what checks the content is
actually true of the diff.

Dry-run mode: LLMClient(dry_run=True) skips the network call entirely and
returns a fabricated-but-schema-valid ChunkReviewResult, while still
exercising build_prompt() and the full return path. Use this to validate
wiring — prompt construction, schema passing, response parsing — before
spending real tokens or requiring an API key. It does NOT validate that
the model actually produces good reviews; that's what the eval suite in
tests/evals is for, against a real or recorded model.
"""

from __future__ import annotations

import json
import logging
import os
import time

from pydantic import ValidationError

from ai_review.diff.chunker import Chunk
from ai_review.llm.prompts import SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT, build_prompt, build_verify_prompt
from ai_review.schema.models import Category, ChunkReviewResult, Finding, Severity

logger = logging.getLogger(__name__)

# Verify against https://docs.claude.com/en/docs/about-claude/models before
# using this in production — pin an exact model you've tested against.
DEFAULT_MODEL = "claude-sonnet-5"
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0
CHUNK_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_TOKENS = 2048
VERIFY_MAX_OUTPUT_TOKENS = 64


class LLMClient:
    """One instance per pipeline run; reused across all chunks so we don't
    pay connection setup cost per chunk.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dry_run: bool = False,
    ) -> None:
        self.model = model
        self.dry_run = dry_run
        self._client = None
        if not dry_run:
            # Imported lazily so dry_run=True never requires the anthropic
            # package to be importable, let alone an API key.
            import anthropic

            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set and no api_key passed. "
                    "Use dry_run=True to test without a key."
                )
            self._client = anthropic.Anthropic(api_key=resolved_key)

    def review_chunk(self, chunk: Chunk) -> ChunkReviewResult:
        if self.dry_run:
            return self._dry_run_response(chunk)
        return self._live_review_chunk(chunk)

    def _live_review_chunk(self, chunk: Chunk) -> ChunkReviewResult:
        import anthropic  # see note above on lazy import

        schema = ChunkReviewResult.model_json_schema()
        prompt = build_prompt(chunk)

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                    timeout=CHUNK_TIMEOUT_SECONDS,
                )
                raw_text = response.content[0].text
                return ChunkReviewResult.model_validate_json(raw_text)

            except anthropic.RateLimitError as e:
                last_error = e
                self._backoff(attempt, chunk.chunk_id, "rate limited")

            except anthropic.APIStatusError as e:
                # Retry server-side failures; anything else (bad request,
                # auth, etc.) won't be fixed by retrying, so re-raise.
                if 500 <= e.status_code < 600:
                    last_error = e
                    self._backoff(attempt, chunk.chunk_id, f"server error {e.status_code}")
                else:
                    raise

            except (ValidationError, json.JSONDecodeError) as e:
                # Structured outputs should make this rare, but a chunk that
                # confuses the model into truncating or hitting stop_reason
                # limits can still fail here — worth one retry.
                last_error = e
                logger.warning("Chunk %s: response failed schema validation: %s", chunk.chunk_id, e)

        logger.error(
            "Giving up on chunk %s after %d attempts: %s", chunk.chunk_id, MAX_RETRIES, last_error
        )
        # Fail soft: an empty result for one chunk shouldn't take down the
        # whole PR review. pipeline.py counts is_failed in chunks_skipped.
        return ChunkReviewResult(chunk_id=chunk.chunk_id, findings=[], is_failed=True)

    def _backoff(self, attempt: int, chunk_id: str, reason: str) -> None:
        delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
        logger.warning(
            "Chunk %s: %s, retrying in %.1fs (attempt %d/%d)",
            chunk_id, reason, delay, attempt, MAX_RETRIES,
        )
        time.sleep(delay)

    def verify_finding(self, finding: Finding, context: str) -> bool:
        """Re-check a single finding against its surrounding code.

        Returns True if the finding is valid, False if it should be
        rejected.  Fail-closed: if verification itself fails after
        retries the finding is rejected (returns False) and the caller
        should log a warning rather than silently post an unverified
        finding.
        """
        if self.dry_run:
            return True
        return self._live_verify_finding(finding, context)

    def _live_verify_finding(self, finding: Finding, context: str) -> bool:
        import anthropic  # lazy import, same pattern as _live_review_chunk

        prompt = build_verify_prompt(finding, context)

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=VERIFY_MAX_OUTPUT_TOKENS,
                    system=VERIFY_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=CHUNK_TIMEOUT_SECONDS,
                )
                raw_text = response.content[0].text
                parsed = json.loads(raw_text)
                return bool(parsed.get("is_valid", False))

            except anthropic.RateLimitError as e:
                last_error = e
                self._backoff(attempt, f"verify:{finding.file}:{finding.line_start}", "rate limited")

            except anthropic.APIStatusError as e:
                if 500 <= e.status_code < 600:
                    last_error = e
                    self._backoff(attempt, f"verify:{finding.file}:{finding.line_start}", f"server error {e.status_code}")
                else:
                    raise

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = e
                logger.warning(
                    "Verify %s:%d: response parse failed: %s",
                    finding.file, finding.line_start, e,
                )

        logger.warning(
            "Verify %s:%d: giving up after %d attempts, rejecting finding (fail-closed): %s",
            finding.file, finding.line_start, MAX_RETRIES, last_error,
        )
        # Fail-closed: unverifiable findings are rejected.
        return False

    def _dry_run_response(self, chunk: Chunk) -> ChunkReviewResult:
        # Still exercise prompt construction so a broken template shows up
        # even in dry-run mode.
        _ = build_prompt(chunk)
        first_line = chunk.content.splitlines()[0] if chunk.content else ""
        return ChunkReviewResult(
            chunk_id=chunk.chunk_id,
            findings=[
                Finding(
                    file=chunk.file,
                    line_start=chunk.start_line,
                    line_end=chunk.start_line,
                    severity=Severity.INFO,
                    category=Category.STYLE,
                    message="[dry run] placeholder finding — no model call was made",
                    code_snippet=first_line,
                    confidence=1.0,
                )
            ],
        )
