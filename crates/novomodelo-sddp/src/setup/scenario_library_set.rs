//! Nested scenario-library containers for training and simulation phases.
//!
//! [`ScenarioLibraries`] groups the per-phase sampling configuration and
//! optional pre-built libraries into two [`PhaseLibraries`] values — one for
//! training and one for simulation.

use cobre_core::scenario::SamplingScheme;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary};

/// Sampling schemes and optional pre-built libraries for a single execution
/// phase (training or simulation).
///
/// One [`SamplingScheme`] and one optional library per entity class (inflow,
/// load, NCS), plus the optional historical inflow library. Each library is
/// pre-standardized; its presence is keyed on the corresponding scheme.
#[derive(Debug)]
pub struct PhaseLibraries {
    /// Forward-pass noise source scheme for the inflow entity class.
    pub inflow_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the load entity class.
    pub load_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the NCS entity class.
    pub ncs_scheme: SamplingScheme,
    /// Historical inflow windows; `Some` iff `inflow_scheme == Historical`.
    pub historical: Option<HistoricalScenarioLibrary>,
    /// External inflow library; `Some` iff `inflow_scheme == External`.
    pub external_inflow: Option<ExternalScenarioLibrary>,
    /// External load library; `Some` iff `load_scheme == External`.
    pub external_load: Option<ExternalScenarioLibrary>,
    /// External NCS library; `Some` iff `ncs_scheme == External`.
    pub external_ncs: Option<ExternalScenarioLibrary>,
}

/// Training and simulation [`PhaseLibraries`] grouped as a pair.
///
/// For each entity class, a `simulation` library is stored as `None` when its
/// scheme matches training's; the simulation context then falls back to the
/// training library. A `None` therefore means "shares training", not "no
/// library".
#[derive(Debug)]
pub struct ScenarioLibraries {
    /// Libraries and schemes used during the training (backward-pass) phase.
    pub training: PhaseLibraries,
    /// Libraries and schemes used during the simulation phase.
    pub simulation: PhaseLibraries,
}
