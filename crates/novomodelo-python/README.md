# novomodelo

Python bindings for the [Novomodelo](https://github.com/ons-ccee-epe/novomodelo) power systems solver.

Novomodelo is a high-performance SDDP (Stochastic Dual Dynamic Programming) solver for hydrothermal dispatch, written in Rust. This package provides Python access to case loading, validation, training, simulation, and result inspection.

## Installation

```bash
pip install novomodelo-python
```

Pre-built wheels are available for:

- Linux x86_64 (manylinux_2_34)
- Linux aarch64 (manylinux_2_28)
- Linux musl x86_64 (musllinux_1_2)
- macOS Apple Silicon (aarch64) and Intel (x86_64)
- Windows x86_64
- Python 3.12+

## Quick Start

```python
import novomodelo

# Load and validate a case
system = novomodelo.io.load_case("path/to/case")
print(
    f"System: {system.n_buses} buses, {system.n_hydros} hydros, {system.n_thermals} thermals"
)

# Run training + simulation
result = novomodelo.run.run("path/to/case", output_dir="output/")
print(f"Converged: {result['converged']}, LB: {result['lower_bound']:.2f}")

convergence = novomodelo.results.load_convergence("output/")
print(f"Iterations: {len(convergence)}")

simulation = novomodelo.results.load_simulation("output/")
print(f"Cost records: {len(simulation['costs'])}")

policy = novomodelo.results.load_policy("output/")
print(f"Iterations completed: {policy['metadata']['producer']['completed_iterations']}")
```

The same workflow, driven step by step:

```python
# The Study lifecycle — construct, train, simulate
study = novomodelo.Study("path/to/case", output_dir="output/")
policy = study.train()
summary = study.simulate(policy)
print(f"Scenarios: {summary['n_scenarios']}, Completed: {summary['completed']}")

# Inspect the study before running any phase
print(f"Stages: {study.stochastic['n_stages']}, Seed: {study.stochastic['seed']}")
```

## Modules

- **`novomodelo.io`** — Load and validate case directories
- **`novomodelo.model`** — Data model classes (System, Bus, Line, Thermal, Hydro, etc.)
- **`novomodelo.run`** — Execute SDDP training and simulation
- **`novomodelo.results`** — Load and inspect output artifacts, including convergence
  history, Parquet simulation outputs, and FlatBuffers policy (FCF) checkpoints
- **`novomodelo.schema`** — JSON Schema export for case-directory input types
- **`novomodelo.errors`** — Typed exception hierarchy for case-loading and solver errors

The package also exports `Study`, `Policy`, `write_policy_checkpoint`, and `version_info` at the top level.

## Requirements

- Python >= 3.12
- No runtime dependencies (the Rust solver is statically linked)

## License

Apache-2.0 — see [LICENSE](https://github.com/ons-ccee-epe/novomodelo/blob/main/LICENSE).

## Links

- [Repository](https://github.com/ons-ccee-epe/novomodelo)
- [Documentation](https://docs.novomodelo.invalid/)
- [Bug Tracker](https://github.com/ons-ccee-epe/novomodelo/issues)
