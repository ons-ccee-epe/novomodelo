"""Smoke tests for the novomodelo Python extension module foundation.

These tests verify that the PyO3 extension module loads correctly and that
the top-level module and its empty sub-modules are importable. They are
intended to be run after `maturin develop --uv` installs the extension.

Run with:
    pytest crates/novomodelo-python/tests/
"""


def test_import_novomodelo() -> None:
    import novomodelo  # noqa: F401, PLC0415


def test_version() -> None:
    import novomodelo  # noqa: PLC0415

    assert isinstance(novomodelo.__version__, str)
    assert len(novomodelo.__version__) > 0


def test_submodules_exist() -> None:
    import novomodelo.io  # noqa: F401, PLC0415
    import novomodelo.model  # noqa: F401, PLC0415
    import novomodelo.results  # noqa: F401, PLC0415
    import novomodelo.run  # noqa: F401, PLC0415


def test_native_module_importable() -> None:
    import novomodelo._native  # noqa: F401, PLC0415


def test_public_submodules_are_the_native_modules() -> None:
    """The public novomodelo.* submodules ARE the _native compiled modules.

    The package re-exports the compiled submodules rather than copying them, so
    identity must hold for all compiled-only modules.
    """
    import novomodelo  # noqa: PLC0415
    import novomodelo._native  # noqa: PLC0415

    assert novomodelo.run is novomodelo._native.run
    assert novomodelo.io is novomodelo._native.io
    assert novomodelo.model is novomodelo._native.model
    assert novomodelo.results is novomodelo._native.results
    assert novomodelo.schema is novomodelo._native.schema


def test_version_matches_native() -> None:
    import novomodelo  # noqa: PLC0415
    import novomodelo._native  # noqa: PLC0415

    assert novomodelo.__version__ == novomodelo._native.__version__


def test_submodule_docstrings_are_the_installed_one_liners() -> None:
    """Each novomodelo.* submodule __doc__ is the lib.rs override, not the //! module doc."""
    import novomodelo  # noqa: PLC0415

    expected = {
        "model": "Data model types for the Novomodelo power systems solver.",
        "io": "I/O helpers for loading Novomodelo case directories.",
        "run": "Solver execution entry points for training and simulation.",
        "results": "Result loading and inspection functions for Novomodelo output artifacts.",
        "errors": "Structured exception hierarchy for Novomodelo errors.",
        "schema": "JSON Schema export helpers for Novomodelo case directory input types.",
    }

    for module_name, expected_doc in expected.items():
        module = getattr(novomodelo, module_name)
        assert module.__doc__ == expected_doc
