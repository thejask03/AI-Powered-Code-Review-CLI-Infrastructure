"""Schema-enforced data contract for AI code review output.

Every LLM call in this project must produce output conforming to
`ReviewResult`. This is the single source of truth for what a "finding"
looks like — the LLM client, the grounding verifier, and the GitHub
poster all import from here rather than redefining shapes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """How serious a finding is. Used for filtering/sorting, not for
    blocking merges by default — that's a config-level policy decision,
    not a schema-level one.
    """

    INFO = "info"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    CRITICAL = "critical"


class Category(str, Enum):
    """Coarse categories, kept small and stable so eval grading and
    dashboards don't churn every time a prompt is tweaked.
    """

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    DESIGN = "design"
    TEST_COVERAGE = "test_coverage"
    STYLE = "style"


class Finding(BaseModel):
    """One reviewer comment, tied to an exact location in the diff.

    `code_snippet` exists specifically for grounding: after the LLM
    responds, verify/grounding.py checks that this snippet actually
    appears at (file, line_start) in the diff we sent. A finding that
    fails that check gets dropped or downweighted rather than posted —
    schema validity does not imply the finding is real.
    """

    file: str = Field(..., description="Path relative to repo root")
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    severity: Severity
    category: Category
    message: str = Field(..., min_length=1, max_length=1000)
    suggested_fix: str | None = Field(
        default=None, description="Optional concrete fix, e.g. a code suggestion"
    )
    code_snippet: str = Field(
        ..., description="Exact excerpt from the diff this finding refers to, for grounding"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's self-reported confidence, 0-1"
    )

    @field_validator("line_end")
    @classmethod
    def end_not_before_start(cls, v: int, info) -> int:
        start = info.data.get("line_start")
        if start is not None and v < start:
            raise ValueError("line_end must be >= line_start")
        return v


class ChunkReviewResult(BaseModel):
    """Output of a single LLM call, scoped to one chunk (one function,
    class, or bounded diff region). The pipeline aggregates many of
    these into a ReviewResult.
    """

    chunk_id: str
    findings: list[Finding] = Field(default_factory=list)
    is_failed: bool = Field(default=False, description="True when the LLM exhausted all retries")


class ReviewResult(BaseModel):
    """Final aggregated output for one PR review run, after all chunks
    have been reviewed, grounded, and deduplicated.
    """

    pr_number: int
    commit_sha: str
    findings: list[Finding] = Field(default_factory=list)
    chunks_reviewed: int = 0
    chunks_skipped: int = Field(
        default=0, description="e.g. chunks too large for the token budget"
    )

    def by_severity(self, minimum: Severity) -> list[Finding]:
        """Filter findings at or above a severity threshold."""
        order = list(Severity)
        cutoff = order.index(minimum)
        return [f for f in self.findings if order.index(f.severity) >= cutoff]
