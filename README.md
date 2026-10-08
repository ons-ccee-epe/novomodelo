<h1 align="center">Novomodelo</h1>

<p align="center">
  <strong>Power system optimization in Rust</strong>
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

- **Coming from other software?** -- See the [novomodelo-bridge guide](https://docs.novomodelo.invalid/running/case-conversion/)
- **New to SDDP?** -- Read [What Novomodelo Solves](https://docs.novomodelo.invalid/overview/what-novomodelo-solves/)
- **Python user?** -- Try the [Python Quickstart](https://docs.novomodelo.invalid/getting-started/python-quickstart/)

## Current Status

Novomodelo is alpha software with a fully functional SDDP solver. The pipeline covers case loading, stochastic scenario generation, training, simulation, policy checkpointing, and output writing. See the [CHANGELOG](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Origin and credits

Novomodelo is developed by Operador Nacional do Sistema Elétrico - ONS, Câmara
de Comercialização de Energia Elétrica - CCEE and Empresa de Pesquisa
Energética - EPE, with other contributors, as a fork of
[Cobre](https://github.com/cobre-rs/cobre), the open-source Rust ecosystem for
power system optimization created by Rogerio J. M. Alves and the Cobre
contributors. The fork was created from the Cobre v0.18.0 release (commit
[`3ab1f748`](https://github.com/cobre-rs/cobre/tree/3ab1f748814a87ffd08585096ca9cbe810320cb6),
DOI [10.5281/zenodo.23202162](https://doi.org/10.5281/zenodo.23202162), tagged
`fork-point` in this repository); every commit up to and including it is
Cobre's and is preserved unchanged here.

References to "Cobre" in this repository describe the origin of the work only;
"Cobre" is not the name of this project.

Cite Cobre v0.18.0 as well when your work discusses the origin of Novomodelo,
the architecture or methodology it inherits, or results obtained with Cobre
itself. The DOI above resolves to a record that exports the citation as BibTeX,
APA and other formats.

## Citing

If you use Novomodelo in published work, please cite Novomodelo:

```bibtex
@software{novomodelo,
  author = {{Operador Nacional do Sistema Elétrico - ONS} and {Câmara de Comercialização de Energia Elétrica - CCEE} and {Empresa de Pesquisa Energética - EPE}},
  title = {Novomodelo},
  url = {https://github.com/ons-ccee-epe/novomodelo},
  license = {Apache-2.0}
}
```

### Upstream reference

Novomodelo is derived from Cobre. The reference below is provided for
information only; citing it is not required when using Novomodelo.
Attribution obligations under the Apache-2.0 license are covered in
`NOTICE` and `LICENSE`.

```bibtex
@software{cobre,
  author = {Alves, Rogerio J. M.},
  title = {Cobre: Open Infrastructure for Power System Computation},
  url = {https://github.com/cobre-rs/cobre},
  license = {Apache-2.0}
}
```

## License

Licensed under [Apache-2.0](LICENSE). The appendix of `LICENSE` carries the
copyright lines of the upstream project and of the fork's maintainers;
[`NOTICE`](NOTICE) carries the fork's attribution notice followed by the
upstream NOTICE verbatim. Redistributions must keep both files (Apache License
2.0, Section 4). Contributors must not remove or alter existing copyright or
attribution notices; new copyright lines are added alongside them.
