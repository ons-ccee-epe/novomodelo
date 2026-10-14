//! Post-training simulation configuration types for `config.json → simulation`.

use serde::{Deserialize, Deserializer, Serialize, de};

use super::scenario_source::RawScenarioSourceConfig;
use super::training::PhaseSolverProfileConfig;

/// Default scenario count when `simulation.selection` is absent. Sole owner of
/// the value; the count resolver reads it.
pub(crate) const DEFAULT_NUM_SCENARIOS: u32 = 2000;

/// Post-training simulation settings (`config.json → simulation`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SimulationConfig {
    /// Enable post-training simulation.
    pub enabled: bool,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    pub io_channel_capacity: u32,

    /// Scenario source configuration for the post-training simulation forward pass.
    /// When absent, falls back to the training scenario source.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,

    /// Simulation solver profile. Absent leaves the phase's built-in
    /// tuned profile.
    #[serde(default)]
    pub solver: Option<PhaseSolverProfileConfig>,

    /// Phase-level scenario selection. Absent resolves to the default sampled
    /// count.
    #[serde(default)]
    pub selection: Option<SimulationSelection>,

    /// Which rows of each stage's row pool the simulation loads into the stage
    /// LP. `all` loads every row a training run added to the pool, including
    /// rows the selection method deactivated and rows `max_active_per_stage`
    /// evicted, so a simulation LP can be larger than the training LPs.
    /// `active` loads only the rows active when training ended.
    ///
    /// A run with training disabled loads each row of the policy with the role
    /// it had in the run that wrote it, so `all` loads what that run's `all`
    /// held and `active` the rows active when it ended; a `resume` does the
    /// same. Under `warm_start` and for `policy.boundary` rows, `all` loads a
    /// row the policy records as `scored` (its writing run stored the trial
    /// state the row was computed at, under `exports.states` or `level1`)
    /// whatever its active flag, and any other loaded row only when that flag
    /// is set. The exported `is_active` flag records the last selection.
    ///
    /// A run with `training.cut_selection.selection.method` set to `dynamic`
    /// that simulates under `all` is refused at load when it would hold a row
    /// that `all` includes and that is not active, in the policy it loads under
    /// `warm_start` or `resume` or with training disabled, or in the rows of
    /// `policy.boundary`; set `active`, or use the `level1` method.
    #[serde(deserialize_with = "deserialize_cut_set")]
    pub cut_set: SimulationRowSet,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            io_channel_capacity: 64,
            scenario_source: None,
            solver: None,
            selection: None,
            cut_set: SimulationRowSet::All,
        }
    }
}

/// Prefixes the derived error with the key, keeping serde's `unknown variant`
/// text: `parse_config` classifies a schema error by it and takes the first
/// backticked word as the field.
fn deserialize_cut_set<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<SimulationRowSet, D::Error> {
    SimulationRowSet::deserialize(deserializer)
        .map_err(|e| de::Error::custom(format_args!("`simulation.cut_set`: {e}")))
}

/// The rows of a stage's row pool a simulation loads
/// (`config.json → simulation.cut_set`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SimulationRowSet {
    /// Every row a training run added, whatever its active flag.
    All,
    /// Only the rows active when training ended.
    Active,
}

/// Post-training scenario selection and its method-specific parameters
/// (`config.json → simulation.selection`).
///
/// The `method` key selects how scenarios are chosen: `sampled` draws
/// `num_scenarios` trajectories; `enumerated` walks the scenario set
/// exhaustively. Each method accepts only its own parameters, so pairing a
/// count with `enumerated` is a parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SimulationSelection {
    /// Sampled scenarios: draw `num_scenarios` trajectories.
    Sampled {
        /// Number of simulation trajectories to draw.
        num_scenarios: u32,
    },
    /// Exhaustive enumeration of the scenario set.
    // A braced variant, not a unit one: serde enforces `deny_unknown_fields`
    // only for braced variants of an internally tagged enum, and this variant
    // must reject a stray `num_scenarios`.
    Enumerated {},
}

/// Effective simulation scenario-count resolution
/// ([`Config::resolve_num_scenarios`](super::Config::resolve_num_scenarios)):
/// either a concrete sampled count or a signal that the count is derived from
/// the policy graph downstream, since config load holds no graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumScenariosResolution {
    /// `num_scenarios` sampled simulation trajectories.
    Sampled(u32),
    /// Exhaustive enumeration; the count is derived from the policy graph.
    Enumerated,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{SimulationConfig, SimulationRowSet, SimulationSelection};
    use crate::config::training::{PriceStrategy, ScaleStrategy};

    #[test]
    fn simulation_cut_set_defaults_to_all() {
        let absent: SimulationConfig = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert_eq!(absent.cut_set, SimulationRowSet::All);

        let active: SimulationConfig =
            serde_json::from_str(r#"{"enabled": true, "cut_set": "active"}"#).unwrap();
        assert_eq!(active.cut_set, SimulationRowSet::Active);

        assert!(
            serde_json::from_str::<SimulationConfig>(r#"{"enabled": true, "cut_set": "some"}"#)
                .is_err()
        );
    }

    /// A `sampled` phase selection round-trips into the `Sampled` variant.
    #[test]
    fn sampled_selection_round_trips() {
        let json = r#"{"enabled": true, "selection": {"method": "sampled", "num_scenarios": 500}}"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.selection,
            Some(SimulationSelection::Sampled { num_scenarios: 500 })
        );
    }

    /// The removed flat `num_scenarios` alias is an unknown-field deserialize
    /// error under `deny_unknown_fields`; the count lives solely in the
    /// `selection.sampled` arm.
    #[test]
    fn flat_num_scenarios_alias_is_deserialize_error() {
        let alias = r#"{"enabled": true, "num_scenarios": 500}"#;
        assert!(
            serde_json::from_str::<SimulationConfig>(alias).is_err(),
            "flat num_scenarios must be rejected as an unknown field"
        );

        let arm = r#"{"enabled": true, "selection": {"method": "sampled", "num_scenarios": 500}}"#;
        let cfg: SimulationConfig = serde_json::from_str(arm).unwrap();
        assert_eq!(
            cfg.selection,
            Some(SimulationSelection::Sampled { num_scenarios: 500 })
        );
    }

    /// A count under `enumerated` is unrepresentable — `deny_unknown_fields` on
    /// the braced variant rejects it at parse time.
    #[test]
    fn enumerated_selection_with_count_is_deserialize_error() {
        let json =
            r#"{"enabled": true, "selection": {"method": "enumerated", "num_scenarios": 500}}"#;
        let result = serde_json::from_str::<SimulationConfig>(json);
        assert!(
            result.is_err(),
            "a count under enumerated must be rejected as unrepresentable"
        );
    }

    #[test]
    fn simulation_solver_profile_block_round_trips() {
        let json = r#"{
            "enabled": true,
            "selection": { "method": "sampled", "num_scenarios": 500 },
            "solver": {
                "scale": "off",
                "price": "row"
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.scale, Some(ScaleStrategy::Off));
        assert_eq!(solver.price, Some(PriceStrategy::Row));
    }

    #[test]
    fn simulation_solver_profile_absent_is_none() {
        assert!(SimulationConfig::default().solver.is_none());
    }

    #[test]
    fn simulation_solver_profile_steepest_edge_fallback_threshold_round_trips() {
        let json = r#"{
            "enabled": true,
            "selection": { "method": "sampled", "num_scenarios": 500 },
            "solver": {
                "steepest_edge_devex_fallback_threshold": 12.5
            }
        }"#;
        let cfg: SimulationConfig = serde_json::from_str(json).unwrap();
        let solver = cfg.solver.as_ref().expect("solver present");
        assert_eq!(solver.steepest_edge_devex_fallback_threshold, Some(12.5));
    }
}
