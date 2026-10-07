"""Tests for rebrand.py: map validation, the one-pass rewrite, moves and the guard."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import rebrand

_REAL_MAP = Path(__file__).with_name("map.toml")

_BASE_MAP = """
[name]
from = "cobre"
forms = { cobre = "novomodelo", Cobre = "Novomodelo", COBRE = "NOVOMODELO" }
adjacent_ok = ['253mCOBRE']

[paths]
keep = ["*/vendor/*", "assets/cobre-*.svg"]

[content]
keep_files = ["LICENSE", "*/vendor/*"]

[[keep_span]]
files = ["README.md"]
pattern = '(?s)```bibtex\\n@software\\{cobre,.*?\\n```'

[[rule]]
pattern = 'github\\.com/cobre-rs/cobre/(issues|pull)/[0-9]+'
keep = true

[[rule]]
pattern = 'github\\.com/cobre-rs/cobre-docs'
to = 'github.com/ons/novomodelo-docs'
open = "docs repository"

[[rule]]
pattern = 'github\\.com/cobre-rs/cobre'
to = 'github.com/ons/novomodelo'

[[rule]]
pattern = 'github\\.com/cobre-rs/ferrompi'
keep = true

[[rule]]
pattern = 'crates\\.io/crates/cobre'
manual = "registry page"
"""

MapFactory = Callable[[str], rebrand.NameMap]


@pytest.fixture
def make_map(tmp_path: Path) -> MapFactory:
    def build(text: str) -> rebrand.NameMap:
        path = tmp_path / "map.toml"
        path.write_text(text, encoding="utf-8")
        return rebrand.load_map(path)

    return build


@pytest.fixture
def base_map(make_map: MapFactory) -> rebrand.NameMap:
    return make_map(_BASE_MAP)


def _repo(root: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def test_load_map_real_map_loads_with_rules() -> None:
    nm = rebrand.load_map(_REAL_MAP)
    assert nm.old == "cobre"
    assert nm.forms[b"Cobre"] == b"Novomodelo"
    assert any(rule.keep for rule in nm.rules)


def test_load_map_target_still_carrying_old_name_raises(make_map: MapFactory) -> None:
    text = _BASE_MAP + "\n[[rule]]\npattern = 'x'\nto = 'cobre-x'\n"
    with pytest.raises(rebrand.MapError, match="still carries"):
        make_map(text)


def test_load_map_rule_with_two_actions_raises(make_map: MapFactory) -> None:
    text = _BASE_MAP + "\n[[rule]]\npattern = 'x'\nto = 'y'\nkeep = true\n"
    with pytest.raises(rebrand.MapError, match="exactly one of"):
        make_map(text)


def test_load_map_open_without_target_raises(make_map: MapFactory) -> None:
    text = _BASE_MAP + "\n[[rule]]\npattern = 'x'\nmanual = 'm'\nopen = 'o'\n"
    with pytest.raises(rebrand.MapError, match="needs the proposed"):
        make_map(text)


def test_load_map_named_group_in_pattern_raises(make_map: MapFactory) -> None:
    text = _BASE_MAP + "\n[[rule]]\npattern = '(?P<x>a)'\nkeep = true\n"
    with pytest.raises(rebrand.MapError, match="named groups"):
        make_map(text)


def test_rewrite_identifiers_in_each_case_keep_their_case(
    base_map: rebrand.NameMap,
) -> None:
    data = b"use cobre_core::X; CobreError; COBRE_BIN; cobre-sddp\n"
    result = rebrand.rewrite(base_map, "a.rs", data)
    assert (
        result.data
        == b"use novomodelo_core::X; NovomodeloError; NOVOMODELO_BIN; novomodelo-sddp\n"
    )
    assert result.counts["name"] == 4


def test_rewrite_specific_address_before_general_one_wins(
    base_map: rebrand.NameMap,
) -> None:
    data = b"https://github.com/cobre-rs/cobre-docs and https://github.com/cobre-rs/cobre/blob/x\n"
    result = rebrand.rewrite(base_map, "a.md", data, rehearsal=True)
    assert result.data == (
        b"https://github.com/ons/novomodelo-docs and https://github.com/ons/novomodelo/blob/x\n"
    )


def test_rewrite_keep_rule_leaves_upstream_address_whole(
    base_map: rebrand.NameMap,
) -> None:
    data = b"[ferrompi](https://github.com/cobre-rs/ferrompi) and issue github.com/cobre-rs/cobre/issues/12\n"
    result = rebrand.rewrite(base_map, "a.md", data)
    assert result.data == data


def test_rewrite_tracker_without_number_points_at_fork(
    base_map: rebrand.NameMap,
) -> None:
    result = rebrand.rewrite(
        base_map, "a.rs", b"report at https://github.com/cobre-rs/cobre/issues\n"
    )
    assert result.data == b"report at https://github.com/ons/novomodelo/issues\n"


def test_rewrite_keep_span_leaves_citation_verbatim(base_map: rebrand.NameMap) -> None:
    block = b"```bibtex\n@software{cobre,\n  title = {Cobre},\n  url = {https://github.com/cobre-rs/cobre}\n}\n```"
    data = b"Cobre is here.\n" + block + b"\nAfter Cobre.\n"
    result = rebrand.rewrite(base_map, "README.md", data)
    assert result.data == b"Novomodelo is here.\n" + block + b"\nAfter Novomodelo.\n"


def test_rewrite_keep_span_applies_only_to_its_files(base_map: rebrand.NameMap) -> None:
    data = b"```bibtex\n@software{cobre,\n}\n```"
    result = rebrand.rewrite(base_map, "OTHER.md", data)
    assert b"novomodelo" in result.data


def test_rewrite_open_rule_reports_hit_unless_rehearsal(
    base_map: rebrand.NameMap,
) -> None:
    data = b"github.com/cobre-rs/cobre-docs\n"
    assert rebrand.rewrite(base_map, "a.md", data).open_hits == ["docs repository"]
    assert rebrand.rewrite(base_map, "a.md", data, rehearsal=True).open_hits == []


def test_rewrite_portuguese_words_are_flagged(base_map: rebrand.NameMap) -> None:
    data = b"Ele descobre o fio; os fios cobrem o cobre.\n"
    result = rebrand.rewrite(base_map, "guia.html", data)
    assert len(result.adjacent) == 2


def test_rewrite_banner_escape_is_not_flagged(base_map: rebrand.NameMap) -> None:
    data = b'format!("\\x1b[1;38;5;253mCOBRE v{version}\\x1b[0m")\n'
    result = rebrand.rewrite(base_map, "banner.rs", data)
    assert result.adjacent == []
    assert b"253mNOVOMODELO v" in result.data


def test_rewrite_manual_rule_leaves_match(base_map: rebrand.NameMap) -> None:
    data = b"https://crates.io/crates/cobre-cli\n"
    assert rebrand.rewrite(base_map, "README.md", data).data == data


def test_plan_moves_order_is_deepest_first_and_kept_paths_stay(
    base_map: rebrand.NameMap,
) -> None:
    paths = [
        "crates/cobre-python/python/cobre/__init__.py",
        "crates/cobre-python/tests/_cobre_cli.py",
        "crates/cobre-solver/vendor/HiGHS",
        "assets/cobre-logo.svg",
    ]
    moves = rebrand.plan_moves(base_map, paths)
    assert moves == [
        ("crates/cobre-python/python/cobre", "crates/cobre-python/python/novomodelo"),
        (
            "crates/cobre-python/tests/_cobre_cli.py",
            "crates/cobre-python/tests/_novomodelo_cli.py",
        ),
        ("crates/cobre-python", "crates/novomodelo-python"),
        ("crates/cobre-solver", "crates/novomodelo-solver"),
    ]


def test_cmd_paths_records_pure_renames(
    base_map: rebrand.NameMap, tmp_path: Path
) -> None:
    root = _repo(
        tmp_path / "repo",
        {
            "crates/cobre-io/src/lib.rs": "pub fn f() {}\n",
            "crates/cobre-io/Cargo.toml": "[package]\n",
        },
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "x",
        ],
        check=True,
    )
    assert rebrand.cmd_paths(base_map, root, dry_run=False) == 0
    status = subprocess.run(
        ["git", "-C", str(root), "diff", "--cached", "-M", "--name-status"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    assert sorted(line.split("\t")[0] for line in status if line) == ["R100", "R100"]


def test_cmd_tokens_open_rule_refuses_and_writes_nothing(
    base_map: rebrand.NameMap, tmp_path: Path
) -> None:
    original = "Cobre docs: https://github.com/cobre-rs/cobre-docs\n"
    root = _repo(tmp_path / "repo", {"README.md": original})
    assert rebrand.cmd_tokens(base_map, root, dry_run=False, rehearsal=False) == 1
    assert (root / "README.md").read_text(encoding="utf-8") == original


def test_cmd_tokens_skips_kept_files(base_map: rebrand.NameMap, tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo", {"LICENSE": "Copyright Cobre\n", "a.md": "Cobre\n"})
    assert rebrand.cmd_tokens(base_map, root, dry_run=False, rehearsal=False) == 0
    assert (root / "LICENSE").read_text(encoding="utf-8") == "Copyright Cobre\n"
    assert (root / "a.md").read_text(encoding="utf-8") == "Novomodelo\n"


def test_find_residuals_sorts_manual_from_residual_and_accepts_keep(
    base_map: rebrand.NameMap, tmp_path: Path
) -> None:
    root = _repo(
        tmp_path / "repo",
        {
            "a.md": "see crates.io/crates/cobre-cli\nleft: cOBRE\n",
            "b.md": "github.com/cobre-rs/ferrompi\n",
            "LICENSE": "Cobre\n",
        },
    )
    findings = rebrand.find_residuals(base_map, root)
    assert [(f.path, f.line, f.kind) for f in findings] == [
        ("a.md", 1, "manual"),
        ("a.md", 2, "residual"),
    ]


def test_find_residuals_reports_name_in_path(
    base_map: rebrand.NameMap, tmp_path: Path
) -> None:
    root = _repo(tmp_path / "repo", {"docs/cobre-guide.md": "fine\n"})
    assert [f.kind for f in rebrand.find_residuals(base_map, root)] == ["path"]


def test_rewrite_manual_rule_with_whole_address_renames_none_of_it(
    make_map: MapFactory,
) -> None:
    text = _BASE_MAP + (
        "\n[[rule]]\npattern = '''docs\\.rs/cobre[^\\s)>\"']*'''\nmanual = \"api docs\"\n"
    )
    nm = make_map(text)
    data = b"| <https://docs.rs/cobre-comm/latest/cobre_comm/> |\n"
    assert rebrand.rewrite(nm, "README.md", data).data == data


def test_load_map_real_map_manual_rules_cover_whole_addresses() -> None:
    nm = rebrand.load_map(_REAL_MAP)
    data = b"[x](https://docs.rs/cobre-io/latest/cobre_io/) https://pypi.org/project/cobre-python/\n"
    assert rebrand.rewrite(nm, "README.md", data).data == data


def test_rewrite_real_map_path_fragment_matches_full_url() -> None:
    nm = rebrand.load_map(_REAL_MAP)
    data = (
        b'"https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/a.json"\n'
        b'content.replace("/cobre-rs/cobre/refs/heads/main/schemas/", "/cobre-rs/cobre/refs/tags/v")\n'
    )
    out = rebrand.rewrite(nm, "init.rs", data, rehearsal=True).data
    assert out.count(b"/ons-ccee-epe/novomodelo/refs/heads/main/schemas/") == 2
    assert b"/ons-ccee-epe/novomodelo/refs/tags/v" in out
    assert b"novomodelo-rs" not in out


def test_rewrite_real_map_leaves_other_upstream_org_mentions_for_hand_edits(
    tmp_path: Path,
) -> None:
    nm = rebrand.load_map(_REAL_MAP)
    data = b"Part of the cobre-rs organization.\n"
    assert rebrand.rewrite(nm, "a.md", data).data == data
    root = _repo(tmp_path / "repo", {"a.md": data.decode()})
    assert [f.kind for f in rebrand.find_residuals(nm, root)] == ["manual"]


def test_cmd_tokens_renaming_a_reference_to_a_kept_file_refuses(
    base_map: rebrand.NameMap, tmp_path: Path
) -> None:
    original = '<img src="assets/cobre-logo.svg"/>\n'
    root = _repo(
        tmp_path / "repo", {"assets/cobre-logo.svg": "<svg/>\n", "README.md": original}
    )
    assert rebrand.cmd_tokens(base_map, root, dry_run=False, rehearsal=False) == 1
    assert (root / "README.md").read_text(encoding="utf-8") == original


def test_rewrite_real_map_keeps_references_to_kept_logos() -> None:
    nm = rebrand.load_map(_REAL_MAP)
    data = b'<img src="assets/cobre-logo-dark.svg" alt="Cobre"/>\n'
    out = rebrand.rewrite(nm, "README.md", data).data
    assert b"assets/cobre-logo-dark.svg" in out
    assert b'alt="Novomodelo"' in out
