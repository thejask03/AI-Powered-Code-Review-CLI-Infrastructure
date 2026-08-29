from ai_review.diff.chunker import Chunk
from ai_review.llm.client import LLMClient
from ai_review.schema.models import Category, ChunkReviewResult, Finding, Severity


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="c1",
        file="src/app.py",
        start_line=1,
        end_line=3,
        content="def add(a, b):\n    return a + b\n",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def make_finding(**overrides) -> Finding:
    defaults = dict(
        file="src/app.py",
        line_start=2,
        line_end=2,
        severity=Severity.WARNING,
        category=Category.BUG,
        message="test message",
        code_snippet="return a + b",
        confidence=0.9,
    )
    defaults.update(overrides)
    return Finding(**defaults)


def test_dry_run_returns_valid_schema():
    client = LLMClient(dry_run=True)
    result = client.review_chunk(make_chunk())
    assert isinstance(result, ChunkReviewResult)
    assert result.chunk_id == "c1"
    assert len(result.findings) >= 1


def test_dry_run_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Constructing with dry_run=True must not raise even with no key set
    # and no api_key passed — that's the whole point of dry-run mode.
    client = LLMClient(dry_run=True)
    result = client.review_chunk(make_chunk(chunk_id="c2"))
    assert result.chunk_id == "c2"


def test_live_mode_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        LLMClient(dry_run=False)
        assert False, "expected ValueError for missing API key"
    except ValueError as e:
        assert "ANTHROPIC_API_KEY" in str(e)


def test_dry_run_finding_references_chunk_location():
    chunk = make_chunk(file="src/orders.py", start_line=42)
    client = LLMClient(dry_run=True)
    result = client.review_chunk(chunk)
    finding = result.findings[0]
    assert finding.file == "src/orders.py"
    assert finding.line_start == 42


def test_dry_run_verify_finding_returns_true():
    """In dry-run mode the verifier should accept all findings so it
    doesn't interfere with pipeline testing."""
    client = LLMClient(dry_run=True)
    finding = make_finding()
    assert client.verify_finding(finding, "def add(a, b):\n    return a + b\n") is True


def test_is_failed_flag_defaults_false():
    """Normal ChunkReviewResult should not be marked as failed."""
    result = ChunkReviewResult(chunk_id="c1", findings=[])
    assert result.is_failed is False


def test_is_failed_flag_can_be_set():
    """When LLM exhausts retries, the result should be marked failed."""
    result = ChunkReviewResult(chunk_id="c1", findings=[], is_failed=True)
    assert result.is_failed is True

