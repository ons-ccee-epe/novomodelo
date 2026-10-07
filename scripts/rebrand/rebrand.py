#!/usr/bin/env python3
"""Rename the product across the repository, driven by one name map.

Every rebrand run reads `scripts/rebrand/map.toml`: the fork's first rename,
each replay of an upstream commit, and the later swap to the final name (a
second map whose `from` is the placeholder, passed with `--map`).

Subcommands, in the order a rename runs them:

  paths   `git mv` every tracked path component that carries the name, deepest
          first. Moves only, so git records each file as a 100% rename.
  tokens  Rewrite file contents: the map's address rules first, then the
          case-preserving name rule, in one pass. Refuses, writing nothing,
          while an `open` rule matches (unless `--rehearsal`) or when the name
          touches a letter somewhere `adjacent_ok` does not cover, or when a
          reference to a kept file (`[paths] keep`) would be renamed.
  check   The residual guard: every occurrence of the old name, in any case,
          outside kept files, kept spans and `keep` rules. `manual` matches
          are listed apart. Exit 1 if anything is found.

Files are tracked regular files only, processed as bytes (encodings and line
endings survive unchanged); files with a NUL byte in their first 8 KiB are
treated as binary and skipped.

Usage:
    python3 scripts/rebrand/rebrand.py paths  [--root DIR] [--map FILE] [--dry-run]
    python3 scripts/rebrand/rebrand.py tokens [--root DIR] [--map FILE] [--dry-run] [--rehearsal]
    python3 scripts/rebrand/rebrand.py check  [--root DIR] [--map FILE]

Exit code 0 on success; 1 when `tokens` refuses or `check` finds the name.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_MAP = Path(__file__).with_name("map.toml")
_BINARY_PROBE = 8192
_NAME_GROUP = "name"


class MapError(ValueError):
    """The name map is malformed."""


@dataclass(frozen=True)
class Rule:
    label: str
    to: bytes | None
    keep: bool
    manual: str | None
    open: str | None


@dataclass(frozen=True)
class KeepSpan:
    files: tuple[str, ...]
    pattern: re.Pattern[bytes]


@dataclass(frozen=True)
class NameMap:
    old: str
    forms: dict[bytes, bytes]
    adjacent_ok: tuple[re.Pattern[bytes], ...]
    keep_paths: tuple[str, ...]
    keep_files: tuple[str, ...]
    keep_spans: tuple[KeepSpan, ...]
    rules: tuple[Rule, ...]
    combined: re.Pattern[bytes]
    any_case: re.Pattern[bytes]


@dataclass
class Rewrite:
    data: bytes
    counts: Counter[str]
    open_hits: list[str]
    adjacent: list[int]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    text: str


def _compile(pattern: str, where: str) -> re.Pattern[bytes]:
    try:
        compiled = re.compile(pattern.encode())
    except re.error as err:
        raise MapError(f"{where}: invalid pattern {pattern!r}: {err}") from err
    if compiled.groupindex:
        raise MapError(f"{where}: named groups are reserved: {pattern!r}")
    return compiled


def _table(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise MapError(f"[{key}] must be a table")
    return value


def _tables(raw: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise MapError(f"[[{key}]] must be an array of tables")
    return value


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MapError(f"{where} must be a list of strings")
    return tuple(value)


def _optional_string(entry: Mapping[str, object], key: str, where: str) -> str | None:
    value = entry.get(key)
    if value is not None and not isinstance(value, str):
        raise MapError(f"{where}: `{key}` must be a string")
    return value


def _rule(entry: Mapping[str, object], index: int, old: str) -> tuple[Rule, str]:
    where = f"rule {index + 1}"
    pattern = entry.get("pattern")
    if not isinstance(pattern, str):
        raise MapError(f"{where}: `pattern` is required")
    _compile(pattern, where)
    actions = [key for key in ("to", "keep", "manual") if key in entry]
    if len(actions) != 1:
        raise MapError(f"{where}: exactly one of `to`, `keep`, `manual`, got {actions}")
    if "keep" in entry and entry["keep"] is not True:
        raise MapError(f"{where}: `keep` must be true")
    to = _optional_string(entry, "to", where)
    manual = _optional_string(entry, "manual", where)
    open_choice = _optional_string(entry, "open", where)
    if open_choice is not None and to is None:
        raise MapError(f"{where}: `open` needs the proposed `to`")
    if to is not None and old.lower() in to.lower():
        raise MapError(f"{where}: `to` still carries {old!r}: {to!r}")
    rule = Rule(
        label=pattern,
        to=to.encode() if to is not None else None,
        keep="keep" in entry,
        manual=manual,
        open=open_choice,
    )
    return rule, pattern


def load_map(path: Path) -> NameMap:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    name = _table(raw, "name")
    old = name.get("from")
    forms = name.get("forms")
    if not isinstance(old, str) or not old or not isinstance(forms, dict) or not forms:
        raise MapError("[name] needs `from` and a non-empty `forms` table")
    for source, target in forms.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise MapError("[name] forms map strings to strings")
        if source.lower() != old.lower():
            raise MapError(f"form {source!r} is not a spelling of {old!r}")
        if old.lower() in target.lower():
            raise MapError(
                f"form {source!r} maps to {target!r}, which still carries {old!r}"
            )

    rules: list[Rule] = []
    patterns: list[bytes] = []
    for index, entry in enumerate(_tables(raw, "rule")):
        rule, pattern = _rule(entry, index, old)
        rules.append(rule)
        patterns.append(pattern.encode())

    alternatives = [b"(?P<r%d>%s)" % (i, p) for i, p in enumerate(patterns)]
    spellings = sorted(forms, key=len, reverse=True)
    alternatives.append(
        b"(?P<%s>%s)"
        % (_NAME_GROUP.encode(), b"|".join(re.escape(s.encode()) for s in spellings))
    )
    try:
        combined = re.compile(b"|".join(alternatives))
    except re.error as err:
        raise MapError(f"rules do not combine into one pattern: {err}") from err

    spans: list[KeepSpan] = []
    for index, entry in enumerate(_tables(raw, "keep_span")):
        where = f"keep_span {index + 1}"
        span_pattern = entry.get("pattern")
        if not isinstance(span_pattern, str):
            raise MapError(f"{where}: `pattern` is required")
        try:
            compiled = re.compile(span_pattern.encode())
        except re.error as err:
            raise MapError(f"{where}: invalid pattern {span_pattern!r}: {err}") from err
        spans.append(
            KeepSpan(_strings(entry.get("files"), f"{where} `files`"), compiled)
        )

    adjacent_ok = _strings(name.get("adjacent_ok", []), "[name] adjacent_ok")
    return NameMap(
        old=old,
        forms={s.encode(): t.encode() for s, t in forms.items()},
        adjacent_ok=tuple(_compile(p, "[name] adjacent_ok") for p in adjacent_ok),
        keep_paths=_strings(_table(raw, "paths").get("keep", []), "[paths] keep"),
        keep_files=_strings(
            _table(raw, "content").get("keep_files", []), "[content] keep_files"
        ),
        keep_spans=tuple(spans),
        rules=tuple(rules),
        combined=combined,
        any_case=re.compile(re.escape(old.encode()), re.IGNORECASE),
    )


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, glob) for glob in globs)


def _kept_ranges(nm: NameMap, rel: str, data: bytes) -> list[tuple[int, int]]:
    ranges = [
        m.span()
        for span in nm.keep_spans
        if _matches(rel, span.files)
        for m in span.pattern.finditer(data)
    ]
    return sorted(ranges)


def _free_ranges(kept: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    free, cursor = [], 0
    for start, end in kept:
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        free.append((cursor, length))
    return free


def _matched_rule(nm: NameMap, m: re.Match[bytes]) -> Rule | None:
    """The address rule behind a combined-pattern match; None for the name rule.

    Every alternative is wrapped in its own named group and inner groups are
    unnamed (`_compile`), so `lastgroup` is always the wrapper.
    """
    group = m.lastgroup
    if group is None:
        raise RuntimeError(f"match {m.group()!r} came from no alternative")
    return None if group == _NAME_GROUP else nm.rules[int(group[1:])]


def _touches_letter(data: bytes, start: int, end: int) -> bool:
    before = data[start - 1 : start]
    after = data[end : end + 1]
    return before.isalpha() or after.islower()


def _scan(
    nm: NameMap, rel: str, data: bytes
) -> Iterator[tuple[re.Match[bytes], Rule | None]]:
    """Yield (match, rule or None) for every map match outside kept spans."""
    for lo, hi in _free_ranges(_kept_ranges(nm, rel, data), len(data)):
        for m in nm.combined.finditer(data, lo, hi):
            yield m, _matched_rule(nm, m)


def rewrite(nm: NameMap, rel: str, data: bytes, rehearsal: bool = False) -> Rewrite:
    pieces: list[bytes] = []
    counts: Counter[str] = Counter()
    open_hits: list[str] = []
    adjacent: list[int] = []
    allowed = [m.span() for p in nm.adjacent_ok for m in p.finditer(data)]
    cursor = 0
    for m, rule in _scan(nm, rel, data):
        start, end = m.span()
        pieces.append(data[cursor:start])
        cursor = end
        if rule is None:
            counts[_NAME_GROUP] += 1
            covered = any(lo <= start and end <= hi for lo, hi in allowed)
            if _touches_letter(data, start, end) and not covered:
                adjacent.append(start)
            pieces.append(nm.forms[m.group()])
            continue
        counts[rule.label] += 1
        if rule.to is None:
            pieces.append(m.group())
            continue
        if rule.open and not rehearsal:
            open_hits.append(rule.open)
        pieces.append(rule.to)
    pieces.append(data[cursor:])
    return Rewrite(b"".join(pieces), counts, open_hits, adjacent)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def tracked(root: Path) -> list[str]:
    return [p.decode() for p in _git(root, "ls-files", "-z").split(b"\0") if p]


def _read_text(root: Path, rel: str) -> bytes | None:
    path = root / rel
    if path.is_symlink() or not path.is_file():
        return None
    data = path.read_bytes()
    return None if b"\0" in data[:_BINARY_PROBE] else data


def _line_of(data: bytes, offset: int) -> tuple[int, str]:
    start = data.rfind(b"\n", 0, offset) + 1
    end = data.find(b"\n", offset)
    line = data[start : end if end != -1 else len(data)]
    return data.count(b"\n", 0, offset) + 1, line.decode("utf-8", "replace").strip()


def plan_moves(nm: NameMap, paths: list[str]) -> list[tuple[str, str]]:
    prefixes = {
        "/".join(p.split("/")[:i]) for p in paths for i in range(1, p.count("/") + 2)
    }
    spellings = re.compile(
        b"|".join(re.escape(s) for s in sorted(nm.forms, key=len, reverse=True))
    )
    moves = []
    for prefix in prefixes:
        parent, _, leaf = prefix.rpartition("/")
        if not spellings.search(leaf.encode()) or _matches(prefix, nm.keep_paths):
            continue
        renamed = spellings.sub(lambda m: nm.forms[m.group()], leaf.encode()).decode()
        moves.append((prefix, f"{parent}/{renamed}" if parent else renamed))
    moves.sort(key=lambda move: (-move[0].count("/"), move[0]))
    return moves


def cmd_paths(nm: NameMap, root: Path, dry_run: bool) -> int:
    moves = plan_moves(nm, tracked(root))
    for old, new in moves:
        print(f"{old} -> {new}")
        if not dry_run:
            _git(root, "mv", "--", old, new)
    print(f"{len(moves)} path(s) {'to move' if dry_run else 'moved'}")
    return 0


def kept_names(nm: NameMap, paths: list[str]) -> set[bytes]:
    """File names that carry the old name but are kept: references must keep them too."""
    spellings = re.compile(
        b"|".join(re.escape(s) for s in sorted(nm.forms, key=len, reverse=True))
    )
    return {
        Path(p).name.encode()
        for p in paths
        if _matches(p, nm.keep_paths) and spellings.search(Path(p).name.encode())
    }


def cmd_tokens(nm: NameMap, root: Path, dry_run: bool, rehearsal: bool) -> int:
    changed: dict[str, bytes] = {}
    totals: Counter[str] = Counter()
    blocked: Counter[str] = Counter()
    adjacent: list[Finding] = []
    dangling: list[str] = []
    paths = tracked(root)
    kept = kept_names(nm, paths)
    for rel in paths:
        if _matches(rel, nm.keep_files):
            continue
        data = _read_text(root, rel)
        if data is None:
            continue
        result = rewrite(nm, rel, data, rehearsal)
        totals.update(result.counts)
        blocked.update(result.open_hits)
        for offset in result.adjacent:
            line, text = _line_of(data, offset)
            adjacent.append(Finding(rel, line, "adjacent", text))
        if result.data != data:
            changed[rel] = result.data
        dangling.extend(
            f"{rel}: {name.decode()}"
            for name in sorted(kept)
            if result.data.count(name) < data.count(name)
        )

    for label, count in sorted(
        totals.items(), key=lambda item: (item[0] != _NAME_GROUP, item[0])
    ):
        print(f"{count:6d}  {label}")
    for finding in adjacent:
        print(
            f"letter-adjacent: {finding.path}:{finding.line}: {finding.text}",
            file=sys.stderr,
        )
    for reason, count in sorted(blocked.items()):
        print(f"open rule matched {count} time(s): {reason}", file=sys.stderr)
    for ref in dangling:
        print(f"reference to a kept file would be renamed: {ref}", file=sys.stderr)
    if adjacent or blocked or dangling:
        print("tokens: refused; nothing written", file=sys.stderr)
        return 1
    if not dry_run:
        for rel, data in changed.items():
            (root / rel).write_bytes(data)
    print(f"{len(changed)} file(s) {'to change' if dry_run else 'changed'}")
    return 0


def find_residuals(nm: NameMap, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel in tracked(root):
        if _matches(rel, nm.keep_paths):
            continue
        if nm.any_case.search(rel.encode()):
            findings.append(Finding(rel, 0, "path", rel))
        if _matches(rel, nm.keep_files):
            continue
        data = _read_text(root, rel)
        if data is None:
            continue
        excused: list[tuple[int, int, str]] = []
        for m, rule in _scan(nm, rel, data):
            if rule is not None and rule.to is None:
                excused.append((*m.span(), "keep" if rule.keep else "manual"))
        kept = _kept_ranges(nm, rel, data)
        for m in nm.any_case.finditer(data):
            start, end = m.span()
            if any(lo <= start and end <= hi for lo, hi in kept):
                continue
            kind = next(
                (k for lo, hi, k in excused if lo <= start and end <= hi), "residual"
            )
            if kind == "keep":
                continue
            line, text = _line_of(data, start)
            findings.append(Finding(rel, line, kind, text))
    return findings


def cmd_check(nm: NameMap, root: Path) -> int:
    findings = find_residuals(nm, root)
    per_line = Counter(findings)
    for kind in ("path", "residual", "manual"):
        for f, count in per_line.items():
            if f.kind == kind:
                times = f" (x{count})" if count > 1 else ""
                print(f"{kind}: {f.path}:{f.line}{times}: {f.text}")
    summary = Counter(f.kind for f in findings)
    print(f"check: {sum(summary.values())} finding(s) {dict(sorted(summary.items()))}")
    return 1 if findings else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=("paths", "tokens", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--map", type=Path, default=_DEFAULT_MAP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args(argv)
    try:
        nm = load_map(args.map)
    except MapError as err:
        print(f"map {args.map}: {err}", file=sys.stderr)
        return 1
    root = args.root.resolve()
    if args.command == "paths":
        return cmd_paths(nm, root, args.dry_run)
    if args.command == "tokens":
        return cmd_tokens(nm, root, args.dry_run, args.rehearsal)
    return cmd_check(nm, root)


if __name__ == "__main__":
    sys.exit(main())
