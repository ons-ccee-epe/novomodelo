# novomodelo

**Open infrastructure for power system computation.**

Novomodelo is an ecosystem of Rust crates for power system analysis and optimization.
This umbrella crate re-exports the individual components for convenience.

For most use cases, depend on the specific crates you need:

| Crate                                                           | Purpose                      |
| --------------------------------------------------------------- | ---------------------------- |
| [`novomodelo-core`](https://crates.io/crates/cobre-core)             | Power system data model      |
| [`novomodelo-io`](https://crates.io/crates/cobre-io)                 | File parsers and serializers |
| [`novomodelo-stochastic`](https://crates.io/crates/cobre-stochastic) | Stochastic process models    |
| [`novomodelo-solver`](https://crates.io/crates/cobre-solver)         | LP/MIP solver abstraction    |
| [`novomodelo-sddp`](https://crates.io/crates/cobre-sddp)             | SDDP algorithm               |
| [`novomodelo-cli`](https://crates.io/crates/cobre-cli)               | Command-line interface       |

## When to Use

Use the `novomodelo` umbrella crate only when you need re-exports from multiple
subcrates in a single dependency. For all other cases, add each subcrate you
actually need as a direct dependency — this keeps compile times lower and
makes the dependency graph explicit.

## Links

| Resource               | URL                                                        |
| ---------------------- | ---------------------------------------------------------- |
| Workspace map          | <https://github.com/ons-ccee-epe/novomodelo/blob/main/ARCHITECTURE.md> |
| Repository             | <https://github.com/ons-ccee-epe/novomodelo>                        |
| Changelog              | <https://github.com/ons-ccee-epe/novomodelo/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/ons-ccee-epe/novomodelo) for the current release.

## License

Apache-2.0
