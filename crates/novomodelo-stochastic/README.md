# cobre-stochastic

Stochastic process models for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

This crate provides the probabilistic building blocks used in scenario-based stochastic
optimization of power systems. It implements Periodic Autoregressive (PAR(p)) models
for inflow time series following the methodology used in the Brazilian power sector,
spectral-based spatial correlation for multi-variate scenario generation, and
deterministic communication-free noise generation via SipHash-1-3 seed derivation.
The `StochasticContext` bundles all precomputed parameters and the opening tree into
a single value ready for iterative optimization algorithms.

## When to Use

Depend on `cobre-stochastic` when you need to generate correlated stochastic scenarios
for a power system optimization algorithm. If you are implementing a new iterative
algorithm that draws inflow or load realisations at each iteration, `sample_forward`
and `StochasticContext` are the primary entry points. The crate is solver-agnostic
and carries no dependency on LP or MIP solvers.

## Key Types

- **`StochasticContext`** — Bundles precomputed PAR parameters, correlated factors, and the opening tree for use in iterative algorithms
- **`ForwardSampler`** — Composite struct holding one `ClassSampler` per entity class; entry point for forward-pass noise sampling
- **`ClassSampler`** — Per-entity-class noise source enum with variants InSample, OutOfSample, Historical, and External
- **`HistoricalScenarioLibrary`** — Pre-standardized historical inflow windows for historical replay sampling
- **`ExternalScenarioLibrary`** — Pre-standardized external scenarios for inflow, load, or NCS classes
- **`PrecomputedPar`** — Precomputed PAR(p) seasonal statistics and AR coefficients ready for fast evaluation
- **`OpeningTree`** — Scenario tree structure defining which openings are sampled at each stage
- **`SpectralFactor`** — Symmetric matrix square root via eigendecomposition used to apply spatial correlation to noise draws
- **`build_forward_sampler`** — Factory function that constructs a `ForwardSampler` from a `ForwardSamplerConfig`

## Module overview

| Module                    | Purpose                                                                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `par`                     | PAR(p) coefficient preprocessing: validation, original-unit conversion, and the `PrecomputedPar` cache                                                                                                                         |
| `par::evaluate`           | PAR model forward evaluation (`evaluate_par`) and inverse noise solving (`solve_par_noise`)                                                                                                                                    |
| `par::fitting`            | PAR model estimation: Levinson-Durbin recursion, seasonal statistics, AR coefficient and correlation estimation, PACF/AIC order selection                                                                                      |
| `noise`                   | Deterministic noise generation: SipHash-1-3 seed derivation (`seed`) and `Pcg64` RNG construction (`rng`)                                                                                                                      |
| `noise::quantile`         | Beasley-Springer-Moro inverse normal CDF (`norm_quantile`)                                                                                                                                                                     |
| `normal`                  | Normal noise precomputation for load demand modeling: `PrecomputedNormal` cache with stage-major layout                                                                                                                        |
| `correlation`             | Spectral spatial correlation: eigendecomposition (`spectral`) and profile resolution (`resolve`)                                                                                                                               |
| `tree`                    | Opening scenario tree: flat storage structure (`opening_tree`) and tree generation (`generate`)                                                                                                                                |
| `tree::lhs`               | Latin Hypercube Sampling: batch `generate_lhs` and point-wise `sample_lhs_point`                                                                                                                                               |
| `tree::qmc_sobol`         | Sobol QMC sequence generation with Joe-Kuo direction tables and Matousek scrambling                                                                                                                                            |
| `tree::qmc_halton`        | Halton QMC sequence generation with Owen-style digit scrambling and prime sieve                                                                                                                                                |
| `sampling`                | Forward-pass sampling abstraction: `ForwardSampler` struct (composite sampler), `ClassSampler` enum, `build_forward_sampler` factory, `SampleRequest` and `ForwardNoise` types; `insample` sub-module for tree-based selection |
| `sampling::out_of_sample` | Out-of-sample fresh noise generation dispatching over `NoiseMethod`                                                                                                                                                            |
| `sampling::historical`    | Historical inflow replay: `HistoricalScenarioLibrary` construction, window discovery, eta standardization, lag seeding, and forward-pass window selection                                                                      |
| `sampling::external`      | External scenario sources: `ExternalScenarioLibrary` construction, per-class standardization (PAR inversion for inflow, mean/std for load and NCS), and forward-pass scenario lookup                                           |
| `sampling::class_sampler` | Per-class noise source enum (`ClassSampler`): InSample tree segment copy, OutOfSample fresh noise, Historical window replay, and External library lookup                                                                       |
| `sampling::window`        | Historical window discovery: `discover_historical_windows` finds contiguous year spans covering the study period in `inflow_history.parquet`                                                                                   |
| `context`                 | `StochasticContext` integration type and `build_stochastic_context` pipeline entry point                                                                                                                                       |
| `error`                   | `StochasticError` — nine variants covering six failure domains of the stochastic layer                                                                                                                                         |

## Deterministic, communication-free noise generation

Each rank derives its own noise seeds independently from a shared `base_seed`
plus a context tuple via SipHash-1-3 (`noise::seed`) — chosen because it is a
fast, non-cryptographic hash producing high-quality 64-bit output suitable for
seeding a `Pcg64` RNG, with no system dependency. This avoids an MPI broadcast
on the hot path and gives identical results regardless of rank count:
`derive_forward_seed(base_seed, iteration, scenario, stage)` for per-opening
forward noise, `derive_forward_seed_grouped(base_seed, iteration, scenario,
group_id)` for noise shared across a stage group (e.g. weekly stages sharing
monthly PAR noise, with a leading `0x01` domain-separation byte),
`derive_stage_seed(base_seed, stage_id)` for batch methods (LHS, QMC) that need
all openings at a stage simultaneously, and `derive_opening_seed(base_seed,
opening_index, stage)` for opening-tree generation. SipHash-1-3 folds message
length into its state, so the differing wire-format byte lengths across these
functions already keep their outputs domain-separated without needing an
explicit prefix on every variant.

## Feature flags

`cobre-stochastic` has one reserved, currently-inert feature:

| Feature      | Default | Description                                                                                                       |
| ------------ | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `slow-tests` | off     | Reserved for workspace consistency with other crates' slow-test gating; there are no slow tests in this crate yet |

No external system libraries (HiGHS, MPI, etc.) are required to build or test
this crate.

## Testing

```
cargo test -p cobre-stochastic
```

No external dependencies or system libraries are required — everything
(siphasher, rand, rand_pcg, rand_distr, thiserror) is Cargo-managed. The suite
covers unit tests, a PAR(p) conformance suite (`tests/conformance.rs`, verified
against hand-computed AR(0)/AR(1) fixtures), a reproducibility suite
(`tests/reproducibility.rs`, covering seed determinism, opening-tree seed
sensitivity, declaration-order invariance, and an infrastructure-genericity
grep gate that fails CI if algorithm-specific references leak into this crate),
and doc-tests for all public types.

## Links

| Resource   | URL                                                       |
| ---------- | --------------------------------------------------------- |
| Docs site  | <https://docs.cobre-rs.dev/>                              |
| API Docs   | https://docs.rs/cobre-stochastic/latest/cobre_stochastic/ |
| Repository | https://github.com/cobre-rs/cobre                         |
| CHANGELOG  | https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md  |

## Status

Alpha — API is functional but not yet stable.

## License

Apache-2.0
