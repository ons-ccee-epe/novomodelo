"""Type stubs for the `cobre.errors` exception hierarchy.

Every leaf subclasses both `CobreError` and the matching builtin, so existing
`except OSError` / `except ValueError` / `except RuntimeError` code keeps
catching while new code can catch the typed class or the common `CobreError`
base. The qualified name of every class is `cobre.errors.<Name>`.
"""

class CobreError(Exception):
    """Base class for every Cobre exception."""

class ValidationError(CobreError, ValueError):
    """Case data or configuration failed validation."""

class PolicyIncompatibleError(CobreError, ValueError):
    """A warm-start policy is incompatible with the current system."""

class CaseIoError(CobreError, OSError):
    """A filesystem read or write failure while loading or writing a case."""

class OutputError(CobreError, OSError):
    """An output serialization, schema, or manifest failure while writing results."""

class SolverError(CobreError, RuntimeError):
    """A training or solver failure.

    For an infeasible subproblem, the `stage`, `iteration`, and `scenario`
    attributes are the integer coordinates of the infeasibility; otherwise they
    are `None`.
    """

    stage: int | None
    iteration: int | None
    scenario: int | None

class SimulationError(CobreError, RuntimeError):
    """A simulation-phase failure."""

class InternalError(CobreError, RuntimeError):
    """An internal software or environment fault."""
