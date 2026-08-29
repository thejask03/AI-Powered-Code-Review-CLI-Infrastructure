from pathlib import Path

from ai_review.schema.models import Category, Finding, Severity
from ai_review.verify.grounding import ground_findings


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


def write_file(tmp_path: Path, rel_path: str, content: str) -> None:
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_valid_finding_survives_unchanged(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    report = ground_findings([make_finding()], repo_root=tmp_path)
    assert len(report.grounded) == 1
    assert report.grounded[0].confidence == 0.9
    assert report.dropped == []
    assert report.downweighted == []


def test_missing_file_dropped(tmp_path):
    report = ground_findings([make_finding(file="does/not/exist.py")], repo_root=tmp_path)
    assert report.grounded == []
    assert report.dropped[0][1] == "file_not_found"


def test_line_out_of_range_dropped(tmp_path):
    write_file(tmp_path, "src/app.py", "one line only\n")
    report = ground_findings([make_finding(line_start=50, line_end=50)], repo_root=tmp_path)
    assert report.grounded == []
    assert report.dropped[0][1] == "line_out_of_range"


def test_snippet_mismatch_downweighted_not_dropped(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    finding = make_finding(code_snippet="this text is nowhere in the file", confidence=0.8)
    report = ground_findings([finding], repo_root=tmp_path)
    assert len(report.grounded) == 1
    assert report.grounded[0].confidence < 0.8
    assert report.dropped == []
    assert len(report.downweighted) == 1


def test_line_tolerance_allows_off_by_one(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    # model reports line 1; snippet actually lives on line 2 — within LINE_TOLERANCE
    report = ground_findings([make_finding(line_start=1, line_end=1)], repo_root=tmp_path)
    assert len(report.grounded) == 1
    assert report.grounded[0].confidence == 0.9  # found within tolerance, not downweighted
    assert report.downweighted == []


def test_whitespace_differences_tolerated(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return   a + b\n")
    report = ground_findings([make_finding()], repo_root=tmp_path)
    assert len(report.grounded) == 1
    assert report.dropped == []
    assert report.downweighted == []


def test_verifier_hook_can_reject(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    report = ground_findings([make_finding()], repo_root=tmp_path, verifier=lambda f, ctx: False)
    assert report.grounded == []
    assert report.dropped[0][1] == "verifier_rejected"


def test_verifier_hook_can_accept(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    report = ground_findings([make_finding()], repo_root=tmp_path, verifier=lambda f, ctx: True)
    assert len(report.grounded) == 1


def test_multiple_findings_mixed_outcomes(tmp_path):
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    findings = [
        make_finding(code_snippet="return a + b"),  # survives
        make_finding(file="missing.py"),  # dropped
        make_finding(code_snippet="nonsense text"),  # downweighted
    ]
    report = ground_findings(findings, repo_root=tmp_path)
    assert len(report.grounded) == 2
    assert len(report.dropped) == 1
    assert len(report.downweighted) == 1


def test_verifier_receives_correct_context(tmp_path):
    """The context passed to the verifier must be the actual file content
    within the LINE_TOLERANCE window, not an empty or arbitrary string."""
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    received_contexts: list[str] = []

    def capture_verifier(finding, context):
        received_contexts.append(context)
        return True

    ground_findings([make_finding()], repo_root=tmp_path, verifier=capture_verifier)
    assert len(received_contexts) == 1
    assert "return a + b" in received_contexts[0]


def test_verifier_rejection_tracked_with_reason(tmp_path):
    """When the verifier rejects a finding that passed all other checks,
    the report should track the reason as 'verifier_rejected'."""
    write_file(tmp_path, "src/app.py", "def add(a, b):\n    return a + b\n")
    report = ground_findings([make_finding()], repo_root=tmp_path, verifier=lambda f, ctx: False)
    assert len(report.dropped) == 1
    dropped_finding, reason = report.dropped[0]
    assert reason == "verifier_rejected"
    assert dropped_finding.file == "src/app.py"
    assert report.grounded == []

