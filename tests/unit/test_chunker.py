import textwrap

from ai_review.diff.chunker import _merge_ranges, _parse_unified_diff, chunk_diff

SAMPLE_DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,2 +1,4 @@\n"
    " def existing():\n"
    "     pass\n"
    "+def new_helper():\n"
    "+    return 42\n"
)


def test_parse_unified_diff_extracts_added_lines():
    files = _parse_unified_diff(SAMPLE_DIFF)
    assert len(files) == 1
    assert files[0].path == "src/app.py"
    assert files[0].changed_lines == {3, 4}


def test_parse_unified_diff_skips_deleted_files():
    deleted = (
        "diff --git a/gone.py b/gone.py\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-x = 1\n"
        "-y = 2\n"
    )
    assert _parse_unified_diff(deleted) == []


def test_parse_unified_diff_ignores_removed_lines():
    diff_text = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,3 +1,2 @@\n"
        " keep\n"
        "-remove_me\n"
        " also_keep\n"
    )
    files = _parse_unified_diff(diff_text)
    # No '+' lines at all, so nothing counts as "changed" in the new file
    assert files == []


def test_merge_ranges_combines_overlapping():
    assert _merge_ranges([(1, 5), (4, 10), (20, 25)]) == [(1, 10), (20, 25)]


def test_merge_ranges_combines_adjacent():
    assert _merge_ranges([(1, 5), (6, 10)]) == [(1, 10)]


def test_merge_ranges_empty():
    assert _merge_ranges([]) == []


def test_chunk_diff_heuristic_expands_to_blank_lines(tmp_path):
    # .txt has no tree-sitter language mapping, so this deterministically
    # exercises the heuristic fallback regardless of what's installed.
    source = textwrap.dedent(
        """\
        block one line one
        block one line two

        block two line one
        block two line two
        block two line three
        """
    )
    (tmp_path / "notes.txt").write_text(source)

    diff_text = (
        "diff --git a/notes.txt b/notes.txt\n"
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -4,3 +4,3 @@\n"
        " block two line one\n"
        "+block two line two\n"
        " block two line three\n"
    )

    chunks = chunk_diff(diff_text, repo_root=tmp_path)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.file == "notes.txt"
    assert chunk.start_line == 4
    assert chunk.end_line == 6
    assert "block two line one" in chunk.content
    assert "block two line three" in chunk.content
    assert not chunk.is_partial


def test_chunk_diff_skips_missing_file(tmp_path):
    diff_text = (
        "diff --git a/missing.py b/missing.py\n"
        "--- a/missing.py\n"
        "+++ b/missing.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert chunk_diff(diff_text, repo_root=tmp_path) == []


def test_chunk_diff_marks_oversized_chunk_partial(tmp_path):
    # One huge unbroken block (no blank lines) with a token_budget small
    # enough that even the heuristic-expanded chunk exceeds it.
    huge_line_count = 500
    source = "\n".join(f"x_{i} = {i}" for i in range(huge_line_count)) + "\n"
    (tmp_path / "big.txt").write_text(source)

    diff_text = (
        "diff --git a/big.txt b/big.txt\n"
        "--- a/big.txt\n"
        "+++ b/big.txt\n"
        "@@ -1,1 +250,1 @@\n"
        "+x_249 = 249\n"
    )

    chunks = chunk_diff(diff_text, repo_root=tmp_path, token_budget=10)
    assert len(chunks) == 1
    assert chunks[0].is_partial


def test_oversized_chunk_truncated_at_line_boundary(tmp_path):
    """Truncation must happen at a line boundary, never mid-line."""
    source = "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n"
    (tmp_path / "big.txt").write_text(source)

    diff_text = (
        "diff --git a/big.txt b/big.txt\n"
        "--- a/big.txt\n"
        "+++ b/big.txt\n"
        "@@ -1,1 +100,1 @@\n"
        "+line_99 = 99\n"
    )

    chunks = chunk_diff(diff_text, repo_root=tmp_path, token_budget=10)
    assert len(chunks) == 1
    assert chunks[0].is_partial
    # Every line in the truncated content must be complete — no partial lines
    content = chunks[0].content
    for line in content.splitlines():
        # Each line should match the pattern "line_N = N" exactly
        assert "line_" in line or line.strip() == ""


def test_module_level_change_gets_wider_context(tmp_path):
    """Module-level code (no enclosing function/class) should get a ±10
    line context window, not the old ±2."""
    # 25 import-only lines, no functions/classes — tree-sitter sees no
    # enclosing boundary node so the ±10 window kicks in.
    lines_list = [f"import mod_{i}" for i in range(25)]
    source = "\n".join(lines_list) + "\n"
    (tmp_path / "imports.py").write_text(source)

    diff_text = (
        "diff --git a/imports.py b/imports.py\n"
        "--- a/imports.py\n"
        "+++ b/imports.py\n"
        "@@ -12,1 +12,1 @@\n"
        "+import mod_11\n"
    )

    chunks = chunk_diff(diff_text, repo_root=tmp_path)
    assert len(chunks) == 1
    # Changed line is 12; ±10 gives lines 2-22 (1-indexed, clamped).
    # The old ±2 window would have given only lines 10-14.
    assert chunks[0].start_line == 2
    assert chunks[0].end_line == 22
    assert "import mod_1" in chunks[0].content
    assert "import mod_20" in chunks[0].content

