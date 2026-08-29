import pytest
from pydantic import ValidationError

from ai_review.schema.models import (
    Category,
    Finding,
    ReviewResult,
    Severity,
)


def make_finding(**overrides) -> Finding:
    defaults = dict(
        file="src/app.py",
        line_start=10,
        line_end=10,
        severity=Severity.WARNING,
        category=Category.BUG,
        message="Possible off-by-one error",
        code_snippet="for i in range(len(items) + 1):",
        confidence=0.8,
    )
    defaults.update(overrides)
    return Finding(**defaults)


def test_valid_finding_constructs():
    f = make_finding()
    assert f.severity == Severity.WARNING
    assert f.suggested_fix is None


def test_line_end_before_start_rejected():
    with pytest.raises(ValidationError):
        make_finding(line_start=20, line_end=10)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        make_finding(confidence=1.5)


def test_empty_message_rejected():
    with pytest.raises(ValidationError):
        make_finding(message="")


def test_review_result_filters_by_severity():
    result = ReviewResult(
        pr_number=42,
        commit_sha="abc123",
        findings=[
            make_finding(severity=Severity.INFO),
            make_finding(severity=Severity.CRITICAL),
            make_finding(severity=Severity.SUGGESTION),
        ],
    )
    critical_and_above = result.by_severity(Severity.CRITICAL)
    assert len(critical_and_above) == 1
    assert critical_and_above[0].severity == Severity.CRITICAL

    warning_and_above = result.by_severity(Severity.WARNING)
    assert len(warning_and_above) == 1  # only CRITICAL qualifies here


def test_review_result_defaults():
    result = ReviewResult(pr_number=1, commit_sha="deadbeef")
    assert result.findings == []
    assert result.chunks_reviewed == 0
    assert result.chunks_skipped == 0
