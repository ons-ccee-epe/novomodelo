"""Guard that the `run.pyi` TypedDict retype did not break the runtime module.

The `_types.pyi` stub and the compiled `cobre.run` module are independent
(stubs are erased at runtime), but this test documents the intent: the
`from ._types import RunResult` edit in `run.pyi` is typing-only and must not
affect the importability or callability of `cobre.run.run`.
"""

from cobre.run import run


def test_run_result_typeddict_importable():
    """`cobre.run.run` stays importable and callable after the stub retype."""
    assert callable(run)
    assert run.__doc__ is not None
    assert run.__doc__.strip() != ""
