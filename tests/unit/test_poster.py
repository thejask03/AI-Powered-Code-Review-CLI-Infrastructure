from ai_review.github.poster import (
    build_summary_body,
    extract_last_reviewed_sha,
    extract_prior_findings,
    find_summary_comment,
    post_summary,
)
from ai_review.schema.models import Severity
from conftest import FakeComment, FakePullRequest, make_finding, make_result


def test_build_summary_body_no_findings():
    body = build_summary_body(make_result())
    assert "No issues found." in body
    assert body.startswith("<!-- ai-review:summary -->")
    assert "<!-- ai-review:last-reviewed-sha:abc123 -->" in body


def test_build_summary_body_orders_critical_first():
    result = make_result(
        findings=[
            make_finding(severity=Severity.INFO, message="minor"),
            make_finding(severity=Severity.CRITICAL, message="serious"),
        ]
    )
    body = build_summary_body(result)
    assert body.index("### Critical") < body.index("### Info")


def test_extract_last_reviewed_sha_round_trip():
    body = build_summary_body(make_result(commit_sha="deadbeef"))
    assert extract_last_reviewed_sha(body) == "deadbeef"


def test_extract_last_reviewed_sha_missing_returns_none():
    assert extract_last_reviewed_sha("just a regular comment") is None


def test_find_summary_comment_ignores_unrelated_comments():
    pr = FakePullRequest(existing_comments=[FakeComment("unrelated human comment")])
    assert find_summary_comment(pr) is None


def test_post_summary_creates_when_none_exists():
    pr = FakePullRequest()
    post_summary(pr, make_result())
    assert len(pr.created_bodies) == 1
    assert len(pr.get_issue_comments()) == 1


def test_post_summary_is_idempotent_on_rerun():
    pr = FakePullRequest()
    post_summary(pr, make_result(commit_sha="aaa111"))
    post_summary(pr, make_result(commit_sha="bbb222"))

    # still exactly one comment, not two
    assert len(pr.get_issue_comments()) == 1
    assert len(pr.created_bodies) == 1  # only the first call created anything
    # the existing comment was edited to reflect the second run
    comment = pr.get_issue_comments()[0]
    assert extract_last_reviewed_sha(comment.body) == "bbb222"


def test_post_summary_leaves_unrelated_comments_alone():
    human_comment = FakeComment("looks good to me")
    pr = FakePullRequest(existing_comments=[human_comment])
    post_summary(pr, make_result())
    assert human_comment.edit_calls == []
    assert len(pr.get_issue_comments()) == 2


def test_extract_prior_findings_round_trip():
    findings = [make_finding(file="a.py", line_start=1), make_finding(file="b.py", line_start=2)]
    body = build_summary_body(make_result(findings=findings))
    round_tripped = extract_prior_findings(body)
    assert [(f.file, f.line_start) for f in round_tripped] == [("a.py", 1), ("b.py", 2)]


def test_extract_prior_findings_survives_html_comment_terminator_in_message():
    # A message containing the literal "-->" would prematurely close a raw
    # (non-base64) HTML comment marker and corrupt the rest of the body —
    # this is exactly what the base64 encoding guards against.
    finding = make_finding(message="watch out for --> this closes tags, plus unicode: café 日本語")
    body = build_summary_body(make_result(findings=[finding]))
    round_tripped = extract_prior_findings(body)
    assert len(round_tripped) == 1
    assert round_tripped[0].message == finding.message
    # and the SHA marker (which comes right before the findings marker)
    # must still be intact, i.e. not swallowed by a corrupted comment
    assert extract_last_reviewed_sha(body) is not None


def test_extract_prior_findings_missing_returns_empty_list():
    assert extract_prior_findings("just a regular comment") == []


def test_extract_prior_findings_no_marker_match_returns_empty_list():
    assert extract_prior_findings("<!-- ai-review:findings:not-valid-base64!! -->") == []


def test_extract_prior_findings_valid_base64_but_invalid_json_returns_empty_list():
    # "aGVsbG8=" is valid base64 (decodes to "hello"), but "hello" isn't
    # valid JSON — exercises the try/except path, not just the regex miss.
    assert extract_prior_findings("<!-- ai-review:findings:aGVsbG8= -->") == []


def test_extract_prior_findings_no_findings_round_trips_empty():
    body = build_summary_body(make_result(findings=[]))
    assert extract_prior_findings(body) == []
