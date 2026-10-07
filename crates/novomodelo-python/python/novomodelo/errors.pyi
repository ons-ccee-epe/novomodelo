"""Type stubs for the `novomodelo.errors` exception hierarchy.

Every leaf subclasses both `NovomodeloError` and the matching builtin, so existing
`except OSError` / `except ValueError` / `except RuntimeError` code keeps
catching while new code can catch the typed class or the common `NovomodeloError`
base. The qualified name of every class is `novomodelo.errors.<Name>`.
"""

class NovomodeloError(Exception):
    """Base class for every Novomodelo exception."""

class ValidationError(NovomodeloError, ValueError):
    """Case data or configuration failed validation."""

class PolicyIncompatibleError(NovomodeloError, ValueError):
    """A warm-start policy is incompatible with the current system."""

class CaseIoError(NovomodeloError, OSError):
    """A filesystem read or write failure while loading or writing a case."""

class OutputError(NovomodeloError, OSError):
    """An output serialization, schema, or manifest failure while writing results."""

class SolverError(NovomodeloError, RuntimeError):
    """A training or solver failure.

    For an infeasible subproblem, the `stage`, `iteration`, and `scenario`
    attributes are the integer coordinates of the infeasibility; otherwise they
    are `None`.
    """

    stage: int | None
    iteration: int | None
    scenario: int | None

class SimulationError(NovomodeloError, RuntimeError):
    """A simulation-phase failure."""

class InternalError(NovomodeloError, RuntimeError):
    """An internal software or environment fault."""
