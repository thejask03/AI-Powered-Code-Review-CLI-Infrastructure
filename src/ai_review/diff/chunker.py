"""Boundary-aware diff chunking.

Design decision (do not naive-split on lines): a hunk showing a fragment
of a function with no signature or surrounding body produces confidently
-wrong reviews. Instead we parse the *full file* at the PR's head commit
(not just the diff hunk) with tree-sitter, and expand every changed line
range out to its smallest enclosing function/class node.

That needs filesystem access to the full file, which is fine in CI (the
repo is checked out at the PR head) but means this won't work against a
bare diff with no working tree.

Two tiers, on purpose:

1. Tree-sitter boundary detection — accurate, needs tree-sitter-language-
   pack installed and a real parser for the file's language.
2. Blank-line/indentation heuristic — no dependencies, works for any
   text file, less precise. Used whenever tier 1 isn't available (no
   language mapping, package not installed, or the grammar rejects the
   language name), so the pipeline degrades gracefully instead of
   failing outright.

The diff-parsing logic itself (_parse_unified_diff) has zero third-party
dependencies and needs no filesystem access — it's pure text parsing, so
it's fully unit-testable in any environment regardless of whether
tree-sitter is installed.

If a single enclosing node exceeds the token budget on its own, we don't
silently drop context: we truncate the sent content but mark the chunk
`is_partial=True` so the prompt can tell the model it's seeing an
incomplete view (see llm/prompts.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    file: str
    start_line: int
    end_line: int
    content: str
    is_partial: bool = False


@dataclass
class _DiffFileChanges:
    path: str
    changed_lines: set[int]  # 1-indexed line numbers in the NEW version of the file


# ---------------------------------------------------------------------------
# Tier 0: unified diff parsing. Pure text, no dependencies, no filesystem.
# ---------------------------------------------------------------------------

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_FILE_HEADER = re.compile(r"^\+\+\+ (.+)$")


def _parse_unified_diff(diff_text: str) -> list[_DiffFileChanges]:
    """Parse a unified diff (what `git diff` / GitHub's diff API produce)
    into, per file, the set of line numbers that changed in the NEW
    version of the file. Renamed/deleted files are skipped — there's no
    new-version file to chunk against.
    """
    files: list[_DiffFileChanges] = []
    current_path: str | None = None
    current_lines: set[int] = set()
    new_line_no = 0

    def flush() -> None:
        if current_path is not None and current_lines:
            files.append(_DiffFileChanges(path=current_path, changed_lines=set(current_lines)))

    for line in diff_text.splitlines():
        file_match = _FILE_HEADER.match(line)
        if file_match:
            flush()
            raw_path = file_match.group(1)
            if raw_path == "/dev/null":
                current_path = None  # file was deleted; nothing to chunk
            else:
                current_path = raw_path[2:] if raw_path.startswith(("a/", "b/")) else raw_path
            current_lines = set()
            continue

        hunk_match = _HUNK_HEADER.match(line)
        if hunk_match:
            new_line_no = int(hunk_match.group(1))
            continue

        if current_path is None:
            continue

        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            current_lines.add(new_line_no)
            new_line_no += 1
        elif line.startswith("-"):
            pass  # removed line; doesn't exist in the new file, no line number to track
        elif line.startswith(" ") or line == "":
            new_line_no += 1
        # other lines ("diff --git", "index ...", "\ No newline at end of
        # file") carry no line-number information; ignore them

    flush()
    return files


# ---------------------------------------------------------------------------
# Tier 1: tree-sitter boundary detection, with graceful fallback.
# ---------------------------------------------------------------------------

_EXTENSION_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
}

# Node type names considered a "reviewable unit" boundary, per language.
# Not exhaustive — add to this as you hit languages where the generic set
# below picks the wrong boundary.
_BOUNDARY_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "method_definition", "class_declaration", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "class_declaration", "arrow_function", "interface_declaration"},
    "tsx": {"function_declaration", "method_definition", "class_declaration", "arrow_function", "interface_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item", "impl_item", "struct_item"},
    "java": {"method_declaration", "class_declaration", "constructor_declaration"},
    "ruby": {"method", "class", "module"},
}
_GENERIC_BOUNDARY_TYPES = {
    "function_definition", "function_declaration", "method_definition",
    "class_definition", "class_declaration",
}


def _language_for_path(path: str) -> str | None:
    return _EXTENSION_LANGUAGE.get(Path(path).suffix)


def _get_ts_parser(language: str):
    """Returns a tree-sitter parser for `language`, or None if the
    package isn't installed or doesn't support it. Never raises — the
    whole point is that callers fall back to the heuristic chunker
    instead of crashing the pipeline.
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None
    try:
        return get_parser(language)
    except Exception:
        return None


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent 1-indexed inclusive (start, end)
    ranges so we never send the same lines to the model twice.
    """
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is a reasonable rough heuristic for code+English
    # mixes. Good enough for a budget check; not a billing-accurate count.
    return max(1, len(text) // 4)


def _build_chunks(
    path: str, lines: list[str], ranges: list[tuple[int, int]], token_budget: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for start, end in ranges:
        content = "\n".join(lines[start - 1 : end])
        is_partial = False
        if _estimate_tokens(content) > token_budget:
            # Truncate at line boundaries instead of mid-line/mid-token
            budget_chars = token_budget * 4
            truncated_lines: list[str] = []
            char_count = 0
            for line in content.splitlines():
                if char_count + len(line) + 1 > budget_chars:
                    break
                truncated_lines.append(line)
                char_count += len(line) + 1
            content = "\n".join(truncated_lines)
            is_partial = True
        chunks.append(
            Chunk(
                chunk_id=f"{path}:{start}-{end}",
                file=path,
                start_line=start,
                end_line=end,
                content=content,
                is_partial=is_partial,
            )
        )
    return chunks


def _chunk_with_heuristic(
    path: str, source: str, changed_lines: set[int], token_budget: int
) -> list[Chunk]:
    """No parser available for this file. Expand each changed line to the
    nearest blank lines above and below — a rough, dependency-free proxy
    for "the surrounding logical block".
    """
    lines = source.splitlines()
    ranges: list[tuple[int, int]] = []
    for line_no in sorted(changed_lines):
        row = line_no - 1
        if row < 0 or row >= len(lines):
            continue
        start = row
        while start > 0 and lines[start - 1].strip() != "":
            start -= 1
        end = row
        while end < len(lines) - 1 and lines[end + 1].strip() != "":
            end += 1
        ranges.append((start + 1, end + 1))
    return _build_chunks(path, lines, _merge_ranges(ranges), token_budget)


def _split_oversized_node(
    node, boundary_types: set[str], changed_lines: set[int]
) -> list[tuple[int, int]]:
    """When a boundary node exceeds the token budget, try to split it
    into its child boundary nodes that contain changed lines.
    Falls back to the original node range if no children are boundaries.
    """
    child_ranges: list[tuple[int, int]] = []
    for child in node.children:
        if child.type in boundary_types:
            child_start = child.start_point[0] + 1  # 1-indexed
            child_end = child.end_point[0] + 1
            if any(child_start <= ln <= child_end for ln in changed_lines):
                child_ranges.append((child_start, child_end))
    return child_ranges if child_ranges else [(node.start_point[0] + 1, node.end_point[0] + 1)]


def _chunk_with_tree_sitter(
    path: str, source: str, changed_lines: set[int], language: str, token_budget: int
) -> list[Chunk]:
    parser = _get_ts_parser(language)
    if parser is None:
        return _chunk_with_heuristic(path, source, changed_lines, token_budget)

    tree = parser.parse(source.encode("utf-8"))
    boundary_types = _BOUNDARY_TYPES.get(language, _GENERIC_BOUNDARY_TYPES)
    lines = source.splitlines()

    ranges: list[tuple[int, int]] = []
    for line_no in sorted(changed_lines):
        row = line_no - 1
        if row < 0 or row >= len(lines):
            continue
        col = max(len(lines[row]) - 1, 0)
        node = tree.root_node.named_descendant_for_point_range((row, 0), (row, col))
        boundary = node
        while boundary is not None and boundary.type not in boundary_types:
            boundary = boundary.parent
        if boundary is None:
            # No enclosing function/class (e.g. a bare module-level
            # statement) — send a wider fixed window so the model has
            # enough context for imports, constants, and top-level logic.
            start, end = max(row - 10, 0), min(row + 10, len(lines) - 1)
        else:
            start, end = boundary.start_point[0], boundary.end_point[0]
            # If the enclosing node is too large, try splitting into
            # its child boundary nodes (e.g. methods inside a class)
            # before falling back to truncation in _build_chunks.
            node_content = "\n".join(lines[start:end + 1])
            if _estimate_tokens(node_content) > token_budget:
                sub_ranges = _split_oversized_node(boundary, boundary_types, changed_lines)
                ranges.extend(sub_ranges)
                continue
        ranges.append((start + 1, end + 1))

    return _build_chunks(path, lines, _merge_ranges(ranges), token_budget)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def chunk_diff(diff_text: str, repo_root: Path | str, token_budget: int = 6000) -> list[Chunk]:
    """Turn a unified diff plus a checked-out working tree into
    boundary-aware review chunks.

    `repo_root` must be the checkout at the diff's NEW (head) commit —
    that's what makes line numbers and tree-sitter parsing line up.
    """
    repo_root = Path(repo_root)
    chunks: list[Chunk] = []

    for diff_file in _parse_unified_diff(diff_text):
        file_path = repo_root / diff_file.path
        if not file_path.exists() or not diff_file.changed_lines:
            continue  # deleted, renamed away, or a hunk with no real content

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # unreadable (permissions, race with a real deletion, etc.)

        language = _language_for_path(diff_file.path)
        if language:
            chunks.extend(
                _chunk_with_tree_sitter(diff_file.path, source, diff_file.changed_lines, language, token_budget)
            )
        else:
            chunks.extend(_chunk_with_heuristic(diff_file.path, source, diff_file.changed_lines, token_budget))

    return chunks
