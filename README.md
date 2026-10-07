<h1 align="center">Novomodelo</h1>

<p align="center">
  <strong>Open infrastructure for power system computation</strong>
</p>

<p align="center">
  <a href="https://github.com/ons-ccee-epe/novomodelo/actions/workflows/ci.yml"><img src="https://github.com/ons-ccee-epe/novomodelo/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://github.com/ons-ccee-epe/novomodelo/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"/></a>
</p>

---

**Novomodelo** is a Rust ecosystem for power system optimization. It ships a distributed SDDP solver for hydrothermal dispatch with a CLI and Python bindings.

## Why Novomodelo?

- **Production performance** -- Rust gives C/C++-level speed with memory safety. For software that dispatches national power grids, both matter.
- **Reproducibility** -- Declaration-order invariance guarantees bit-for-bit identical results regardless of input entity ordering.
- **Modularity** -- Pick the crates you need. Use `novomodelo-core` for data modeling alone, or `novomodelo-sddp` for the full solver.
- **Interoperability** -- JSON/Parquet input, Python bindings for Jupyter workflows, MCP server for AI agents (reserved; not yet implemented).

## Install

Nothing is published to crates.io or PyPI under this name. Build from a clone;
the solvers are vendored as git submodules:

```bash
git clone --recurse-submodules https://github.com/ons-ccee-epe/novomodelo.git
cd novomodelo

# Rust CLI (requires Rust 1.88+)
cargo install --locked --path crates/novomodelo-cli

# Python bindings (3.12 / 3.13 / 3.14)
pip install ./crates/novomodelo-python
```

Each tagged release also attaches prebuilt CLI archives to the repository's
GitHub releases.

## Quick Links

| Resource      | Link                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| Documentation | [docs.novomodelo.invalid](https://docs.novomodelo.invalid/)                         |

## Getting Started

- **Coming from other software?** -- See the [novomodelo-bridge guide](https://docs.novomodelo.invalid/guide/novomodelo-bridge.html)
- **New to SDDP?** -- Read [What Novomodelo Solves](https://docs.novomodelo.invalid/tutorial/what-novomodelo-solves.html)
- **Python user?** -- Try the [Python Quickstart](https://docs.novomodelo.invalid/guide/python-quickstart.html)

## Current Status

Novomodelo is alpha software with a fully functional SDDP solver. The pipeline covers case loading, stochastic scenario generation, training, simulation, policy checkpointing, and output writing. See the [CHANGELOG](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Origin and credits

Novomodelo is developed by Operador Nacional do Sistema Elétrico - ONS as a
fork of [Cobre](https://github.com/cobre-rs/cobre), the open-source Rust
ecosystem for power system optimization created by Rogerio J. M. Alves and the
Cobre contributors. The fork was created from the Cobre v0.18.0 release
(commit `3ab1f748`, tagged `fork-point` in this repository); every commit up to
and including it is Cobre's and is preserved unchanged here. Changes taken
from Cobre after that release carry an `Upstream-Commit:` trailer. Issue and
PR numbers (`#NN`) in commit messages before the fork refer to
<https://github.com/cobre-rs/cobre>.

The Cobre name and logo belong to the Cobre project; [TRADEMARKS.md](TRADEMARKS.md)
states how they may be used. If you use Novomodelo in published work, please
also cite Cobre (see `CITATION.cff`).

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
