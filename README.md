<p align="center">
  <img src="assets/cobre-logo-dark.svg" width="360" alt="Novomodelo — Power Systems in Rust"/>
</p>

<p align="center">
  <strong>Open infrastructure for power system computation</strong>
</p>

<p align="center">
  <a href="https://github.com/ons-ccee-epe/novomodelo/actions/workflows/ci.yml"><img src="https://github.com/ons-ccee-epe/novomodelo/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://codecov.io/gh/cobre-rs/cobre"><img src="https://codecov.io/gh/cobre-rs/cobre/branch/main/graph/badge.svg" alt="Coverage"/></a>
  <a href="https://crates.io/crates/cobre"><img src="https://img.shields.io/crates/v/cobre.svg" alt="crates.io"/></a>
  <a href="https://pypi.org/project/cobre-python/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/cobre-python"></a>
  <a href="https://pypi.org/project/cobre-python/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/cobre-python"></a>
  <a href="https://docs.rs/cobre-sddp"><img src="https://docs.rs/cobre-sddp/badge.svg" alt="docs.rs"/></a>
  <a href="https://github.com/ons-ccee-epe/novomodelo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"/></a>
</p>

---

**Novomodelo** is a Rust ecosystem for power system optimization. It ships a distributed SDDP solver for hydrothermal dispatch with a CLI and Python bindings. The name comes from the Portuguese word for **copper**.

## Why Novomodelo?

- **Production performance** -- Rust gives C/C++-level speed with memory safety. For software that dispatches national power grids, both matter.
- **Reproducibility** -- Declaration-order invariance guarantees bit-for-bit identical results regardless of input entity ordering.
- **Modularity** -- Pick the crates you need. Use `novomodelo-core` for data modeling alone, or `novomodelo-sddp` for the full solver.
- **Interoperability** -- JSON/Parquet input, Python bindings for Jupyter workflows, MCP server for AI agents (reserved; not yet implemented).

## Install

```bash
# Rust CLI (requires Rust 1.88+ and HiGHS)
cargo install novomodelo-cli

# Python bindings (3.12 / 3.13 / 3.14)
pip install novomodelo-python
```

## Quick Links

| Resource      | Link                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| Documentation | [docs.novomodelo.invalid](https://docs.novomodelo.invalid/)                         |
| API Docs      | [docs.rs/cobre-sddp](https://docs.rs/cobre-sddp)                        |
| PyPI          | [pypi.org/project/cobre-python](https://pypi.org/project/cobre-python/) |

## Getting Started

- **Coming from other software?** -- See the [novomodelo-bridge guide](https://docs.novomodelo.invalid/guide/novomodelo-bridge.html)
- **New to SDDP?** -- Read [What Novomodelo Solves](https://docs.novomodelo.invalid/tutorial/what-novomodelo-solves.html)
- **Python user?** -- Try the [Python Quickstart](https://docs.novomodelo.invalid/guide/python-quickstart.html)

## Current Status

Novomodelo is alpha software with a fully functional SDDP solver. The pipeline covers case loading, stochastic scenario generation, training, simulation, policy checkpointing, and output writing. See the [CHANGELOG](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under [Apache-2.0](LICENSE).

```bibtex
@software{cobre,
  author = {Alves, Rogerio J. M.},
  title = {Cobre: Open Infrastructure for Power System Computation},
  url = {https://github.com/cobre-rs/cobre},
  license = {Apache-2.0}
}
```
