"""Tests that validate the README's public-surface claims."""

import re
import sys
from pathlib import Path

import pytest


def test_module_census_matches_registered_submodules() -> None:
    """The README's `## Modules` section lists all public submodules."""
    import cobre  # noqa: F401 -- registers the cobre.* submodules in sys.modules

    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    readme_modules = set(
        re.findall(r"^- \*\*`cobre\.([a-z_]+)`\*\*", readme_text, re.MULTILINE)
    )

    installed_modules = {
        name.removeprefix("cobre.")
        for name in sys.modules
        if name.startswith("cobre.")
        and "." not in name[len("cobre.") :]
        and not name.startswith("cobre._")
    }

    assert readme_modules == installed_modules, (
        f"README module census disagrees with registered submodules.\n"
        f"  README: {sorted(readme_modules)}\n"
        f"  Installed: {sorted(installed_modules)}\n"
        f"  Missing from README: {sorted(installed_modules - readme_modules)}\n"
        f"  Extra in README: {sorted(readme_modules - installed_modules)}"
    )


def test_wheel_platform_tags_match_release_workflow() -> None:
    """The README's wheel-platform list matches the release matrix."""
    workflow_path = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "workflows"
        / "release-python.yml"
    )

    if not workflow_path.exists():
        pytest.skip("Workflow file not present (installed wheel)")

    workflow_text = workflow_path.read_text(encoding="utf-8")

    raw_tags = re.findall(r"manylinux:\s*\"?([A-Za-z0-9_]+)\"?", workflow_text)
    parsed_tags = {
        tag if tag.startswith("musllinux") else f"manylinux_{tag}" for tag in raw_tags
    }

    assert parsed_tags, "No manylinux tags found in workflow — parse may have failed"

    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    missing_tags = [tag for tag in parsed_tags if tag not in readme_text]

    assert not missing_tags, (
        f"Wheel platform tags from the release workflow are missing from the README.\n"
        f"  Expected tags: {sorted(parsed_tags)}\n"
        f"  Missing: {sorted(missing_tags)}"
    )
