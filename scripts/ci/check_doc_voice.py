#!/usr/bin/env python3
"""Check user-facing prose docs for promotional voice and unpinned "typical" numbers.

Enforces `.claude/rules/doc-integrity.md` §5 (sober reference register — no
marketing/hype voice) and §2/§4 (no hand-frozen "typical"/"about N" numbers)
across `README.md` and `CONTRIBUTING.md`. (The former `book/src/` scan root
was retired with book/ — mdBook decommission; the unified docs site at
docs.novomodelo.invalid owns prose voice for user-facing content now.)

Scans PROSE only: fenced code blocks (``` / ~~~), inline `code` spans, and HTML
comments are blanked before matching, so code samples, identifiers, and config
defaults are never flagged. A line carrying an inline `<!-- doc-voice-ok: reason -->`
marker is exempted — use it sparingly, only for a genuine domain term that trips a
pattern.

Two checks:
  1. Hype phrases — a curated, high-signal list of marketing superlatives and
     contrasting-affirmative constructs (§5 banned constructs).
  2. Unpinned typical-numbers — a hedge word (typically/roughly/usually/
     approximately/commonly) followed by a magnitude, "N or more", "common in
     practice", or "on a modern/typical ..." (§2: state the invariant, not the
     transient number). Structural references ("stage 0", "1-based") are excluded.

Usage:
    python3 scripts/ci/check_doc_voice.py
    python3 scripts/ci/check_doc_voice.py --root /path/to/repo

Exit code 0 only if no prose violation is found; 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Files scanned, relative to the repo root. `.claude/rules/*` is intentionally
# NOT scanned: doc-integrity.md quotes every banned phrase as part of its own
# rule.
_TOP_LEVEL_DOCS: tuple[str, ...] = ("README.md", "CONTRIBUTING.md")

# §5 banned constructs. Each entry is (label, regex). Kept deliberately tight:
# only phrases that have no sober-reference use, so a hit is almost always real.
_HYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "hype-superlative",
        re.compile(
            r"(?i)\b("
            r"blazing[- ]?fast|blazingly|lightning[- ]?fast|world[- ]class|"
            r"best[- ]in[- ]class|cutting[- ]edge|state[- ]of[- ]the[- ]art|"
            r"game[- ]chang(?:ing|er)|revolutionar(?:y|ize)|revolutioniz\w*|"
            r"seamless(?:ly)?|effortless(?:ly)?|production[- ]grade|"
            r"battle[- ]tested|turnkey|supercharg\w*|unleash\w*|high[- ]fidelity|"
            r"dramatically|zero[- ]cost|zero[- ]overhead"
            r")\b"
        ),
    ),
    (
        "hype-contrasting-affirmative",
        re.compile(
            r"(?i)("
            r"more than just\b|not just an?\b|isn't just\b|is not just\b|"
            r"isn't merely\b|is not merely\b|not merely\b"
            r")"
        ),
    ),
]

# §2 unpinned-number checks.
_HEDGE = re.compile(r"(?i)\b(typically|roughly|usually|approximately|commonly)\b")
_OR_MORE = re.compile(r"(?i)\b\d[\d,]*\s+or\s+more\b")
_COMMON_IN_PRACTICE = re.compile(r"(?i)\bcommon in practice\b")
_ON_A_MODERN = re.compile(r"(?i)\bon a (?:modern|typical|standard|decent|recent)\b")

# A number found shortly after a hedge word, capturing the word right before it.
_HEDGE_NUM = re.compile(r"(?i)([A-Za-z][\w-]*\s+)?(~?\d[\d.,]*\S*)")
# Words that legitimately precede a number (structural index, not a magnitude).
_STRUCTURAL_NOUNS = frozenset(
    {
        "stage",
        "stages",
        "rank",
        "ranks",
        "id",
        "ids",
        "index",
        "indices",
        "step",
        "steps",
        "phase",
        "phases",
        "bus",
        "buses",
        "line",
        "lines",
        "node",
        "nodes",
        "figure",
        "section",
        "chapter",
        "version",
        "table",
        "day",
        "days",
        "month",
        "months",
        "week",
        "weeks",
        "year",
        "years",
        "dimension",
        "dimensions",
        "order",
        "tier",
        "tiers",
        "block",
        "blocks",
        "level",
        "levels",
        "row",
        "rows",
        "column",
        "columns",
        "page",
    }
)
# A digit that is really part of a token like "1-based" / "8-bit" / "2-D".
_STRUCTURAL_SUFFIX = re.compile(
    r"(?i)^\d[\d.,]*-?(based|indexed|bit|byte|dimensional|d)\b"
)

_OK_MARKER = re.compile(r"<!--\s*doc-voice-ok")
_INLINE_CODE = re.compile(r"`[^`]*`")
_HTML_COMMENT = re.compile(r"<!--.*?-->")


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """Return (lineno, prose) for each line, with code and comments blanked.

    Fenced code blocks are dropped wholesale; inline code spans and HTML comments
    are blanked in place so that line numbers are preserved for reporting.
    """
    out: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        prose = _INLINE_CODE.sub(" ", line)
        prose = _HTML_COMMENT.sub(" ", prose)
        out.append((lineno, prose))
    return out


def _typical_number_hit(prose: str) -> str | None:
    """Return a short description of the first unpinned-number hit, else None."""
    or_more = _OR_MORE.search(prose)
    if or_more:
        return or_more.group(0)
    if _COMMON_IN_PRACTICE.search(prose):
        return "common in practice"
    on_a = _ON_A_MODERN.search(prose)
    if on_a:
        return on_a.group(0)
    for hedge in _HEDGE.finditer(prose):
        tail = prose[hedge.end() : hedge.end() + 30]
        m = _HEDGE_NUM.search(tail)
        if not m:
            continue
        preceding = (m.group(1) or "").strip().lower()
        if preceding in _STRUCTURAL_NOUNS:
            continue
        num_part = m.group(2).lstrip()
        if num_part.startswith("~"):
            num_part = num_part[1:]
        if _STRUCTURAL_SUFFIX.match(num_part):
            continue
        return f"{hedge.group(0)} ... {num_part[:12].strip()}"
    return None


def scan(root: Path) -> tuple[list[tuple[Path, int, str, str]], int]:
    """Return (violations, lines_checked).

    Each violation is (path, lineno, rule_label, matched_text).
    """
    targets: list[Path] = []
    for name in _TOP_LEVEL_DOCS:
        p = root / name
        if p.exists():
            targets.append(p)

    violations: list[tuple[Path, int, str, str]] = []
    lines_checked = 0
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as err:
            print(f"WARNING: could not read {path}: {err}", file=sys.stderr)
            violations.append((path, 0, "unreadable", str(err)))
            continue
        raw_lines = text.splitlines()
        for lineno, prose in _prose_lines(text):
            if not prose.strip():
                continue
            lines_checked += 1
            # The OK marker is an HTML comment, blanked out of `prose`; check the
            # raw source line so a genuine exception can opt out.
            if _OK_MARKER.search(raw_lines[lineno - 1]):
                continue
            for label, pat in _HYPE_PATTERNS:
                m = pat.search(prose)
                if m:
                    violations.append((path, lineno, label, m.group(0).strip()))
            hit = _typical_number_hit(prose)
            if hit is not None:
                violations.append((path, lineno, "unpinned-number", hit))
    return violations, lines_checked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check README/CONTRIBUTING prose for hype and unpinned numbers."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    violations, lines_checked = scan(args.root)
    if not violations:
        print(
            f"OK: {lines_checked} prose lines scanned; no promotional voice or "
            f"unpinned 'typical' numbers found (doc-integrity §5/§2)."
        )
        sys.exit(0)

    for path, lineno, label, text in violations:
        rel = path.relative_to(args.root) if path.is_absolute() else path
        print(f"VIOLATION [{label}]: {rel}:{lineno}: {text!r}")
    print(
        f"FAIL: {len(violations)} prose violation(s). Rewrite per "
        f".claude/rules/doc-integrity.md §5/§2, or mark a genuine exception with "
        f"an inline '<!-- doc-voice-ok: reason -->'."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
