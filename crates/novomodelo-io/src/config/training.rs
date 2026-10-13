//! Training-phase configuration types for `config.json → training`.

use std::fmt;
use std::num::{NonZeroU32, NonZeroUsize};

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use super::scenario_source::RawScenarioSourceConfig;

/// Training parameters (`config.json → training`).
///
/// `selection` and `stopping_rules` are required; the loader rejects a
/// configuration that omits either.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainingConfig {
    /// Enable the training phase. When `false`, skip directly to simulation.
    #[serde(default = "TrainingConfig::default_enabled")]
    pub enabled: bool,

    /// Random seed for the opening scenario tree (reproducible training).
    #[serde(default)]
    pub tree_seed: Option<i64>,

    /// List of stopping rule configurations.
    ///
    /// **Mandatory** — no default. Must contain at least one `iteration_limit` rule.
    #[cfg_attr(
        feature = "schema",
        schemars(
            required,
            extend("contains" = {"required": ["type"], "properties": {"type": {"const": "iteration_limit"}}})
        )
    )]
    pub stopping_rules: Option<Vec<StoppingRuleConfig>>,

    /// How multiple stopping rules combine: `any` (OR) or `all` (AND over every
    /// rule except `iteration_limit`, which caps the run).
    #[serde(default)]
    pub stopping_mode: StoppingMode,

    /// Row-selection settings.
    // Rationale: the type stays algorithm-neutral (`RowSelectionConfig`) per the
    // infrastructure genericity rule, while the serialized key uses the
    // domain-standard term every practitioner types. The key/type divergence is
    // deliberate, not an unfinished rename.
    #[serde(default)]
    pub cut_selection: RowSelectionConfig,

    /// LP solver retry settings and optional per-phase solver profiles.
    #[serde(default)]
    pub solver: TrainingSolverConfig,

    /// Parallel-execution settings.
    #[serde(default)]
    pub parallelism: ParallelismConfig,

    /// Scenario source configuration for the training forward pass.
    /// When absent, all classes default to `in_sample`.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,

    /// Phase-level scenario selection. The forward-pass count lives in the
    /// `sampled` arm; absent is a missing-count load error.
    #[cfg_attr(feature = "schema", schemars(required))]
    pub selection: Option<TrainingSelection>,
}

/// Training-phase scenario selection and its method-specific parameters
/// (`config.json → training.selection`).
///
/// The `method` key selects how scenarios are chosen: `sampled` runs
/// `forward_passes` trajectories per iteration; `enumerated` walks the scenario
/// openings exhaustively. Each method accepts only its own parameters, so
/// pairing a count with `enumerated` is a parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum TrainingSelection {
    /// Sampled forward passes: `forward_passes` trajectories per iteration.
    Sampled {
        /// Number of forward-pass trajectories per iteration.
        forward_passes: u32,
    },
    /// Exhaustive enumeration of the scenario openings.
    // A braced variant, not a unit one: serde enforces `deny_unknown_fields`
    // only for braced variants of an internally tagged enum, and this variant
    // must reject a stray `forward_passes`.
    Enumerated {},
}

/// Effective training forward-pass resolution
/// ([`Config::resolve_forward_passes`](super::Config::resolve_forward_passes)):
/// either a concrete sampled count or a signal that the count is derived from
/// the policy graph downstream, since config load holds no graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardPassesResolution {
    /// `forward_passes` sampled trajectories per iteration.
    Sampled(u32),
    /// Exhaustive enumeration; the count is derived from the policy graph.
    Enumerated,
}

impl TrainingConfig {
    pub(super) fn default_enabled() -> bool {
        true
    }
}

/// How multiple stopping rules combine into a single stop decision
/// (`config.json → training.stopping_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum StoppingMode {
    /// Stop when any configured rule triggers (OR).
    #[default]
    Any,
    /// Stop when every configured rule other than `iteration_limit` triggers at
    /// the same iteration (AND); the largest `iteration_limit` caps the run.
    All,
}

impl<'de> Deserialize<'de> for StoppingMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "any" => Ok(Self::Any),
            "all" => Ok(Self::All),
            other => Err(serde::de::Error::unknown_variant(other, &["any", "all"])),
        }
    }
}

impl fmt::Display for StoppingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => f.write_str("any"),
            Self::All => f.write_str("all"),
        }
    }
}

/// Row-selection settings (`config.json → training.cut_selection`).
///
/// `selection` chooses the method and carries only that method's parameters;
/// omitting it (the default) disables row selection.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RowSelectionConfig {
    /// Binding threshold for activity tracking: a constraint row counts as
    /// binding at a solution point when its dual multiplier is strictly greater
    /// than this value (a signed comparison; a multiplier equal to the
    /// threshold does not bind). In one solve the multipliers of a stage's pool
    /// rows are non-negative weights whose sum is at most that stage's discount
    /// factor, so the threshold is dimensionless. The recorded activity is read
    /// only by `max_active_per_stage` and by the `seed_window` of the `dynamic`
    /// method. Must be `>= 0`. A value of `0` counts a multiplier at solver
    /// rounding level as binding.
    pub row_activity_tolerance: f64,

    /// Cap on the active rows that the current run has added to a stage's row
    /// pool. Absent (default): no cap.
    ///
    /// The cap applies to the row pool of every stage strictly between the
    /// first and the last; the pools of the first stage and of the last stage
    /// are not capped. Where the policy graph has several nodes at such a
    /// stage, each node has its own pool and the cap applies to each pool
    /// separately. Rows loaded from a policy (`policy.mode` set to `warm_start`
    /// or `resume`, or `policy.boundary`) are never evicted and do not count
    /// toward the cap, so a pool can hold more active rows than the cap. Under
    /// `resume` every reloaded row is a loaded row, so the cap counts only the
    /// rows added after the resume.
    ///
    /// The cap is applied in every training iteration, after that iteration's
    /// rows have been added, whether or not a selection method is configured;
    /// in an iteration where the selection method runs its periodic pass, the
    /// cap is applied after that pass. Rows added in the current iteration
    /// count toward the cap and are never evicted, so the count stays above
    /// the cap when those rows alone exceed it.
    ///
    /// When the count exceeds the cap, the excess is evicted from the rows
    /// added in earlier iterations, in ascending order of the iteration in
    /// which a row was last binding; ties go to the row that was binding in
    /// fewer solves, then to the row at the lower position in the pool. A row
    /// that has never been binding ranks by the iteration that added it.
    /// `row_activity_tolerance` decides whether a row is binding.
    ///
    /// The cap is rejected at load when set together with
    /// `training.cut_selection.selection.method` = `dynamic` or with
    /// `training.selection.method` set to `enumerated`.
    ///
    /// Eviction deactivates a row and keeps it in the pool. With no selection
    /// method an evicted row does not become active again during training.
    /// With `level1`, a later selection pass may reactivate it, and the cap is
    /// then applied again in the same iteration.
    pub max_active_per_stage: Option<u32>,

    /// Active selection method and its parameters. Absent/`null` (default)
    /// disables row selection.
    #[serde(default, deserialize_with = "deserialize_selection_method")]
    pub selection: Option<SelectionMethod>,
}

impl Default for RowSelectionConfig {
    fn default() -> Self {
        Self {
            row_activity_tolerance: 1e-6,
            max_active_per_stage: None,
            selection: None,
        }
    }
}

/// Methods that left the config surface; naming one is a load error that names
/// `level1`, never a silent mapping onto it.
const REMOVED_SELECTION_METHODS: [&str; 2] = ["lml1", "domination"];

/// A `level1` key that left the config surface; setting it is a load error,
/// never a silently dropped key.
const REMOVED_LEVEL1_KEY: &str = "tie_tolerance";

/// Deserializes `selection`, rejecting a removed `method` or `level1` key
/// before the derived [`SelectionMethod`] implementation reports it as an
/// unknown variant or field.
fn deserialize_selection_method<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<SelectionMethod>, D::Error> {
    let Some(SelectionEntries(entries)) = Option::<SelectionEntries>::deserialize(deserializer)?
    else {
        return Ok(None);
    };
    let method = entries.get("method").and_then(Value::as_str);
    if let Some(removed) = method.filter(|method| REMOVED_SELECTION_METHODS.contains(method)) {
        return Err(de::Error::custom(format_args!(
            "the row-selection method `{removed}` was removed: use `level1` \
             (key `training.cut_selection.selection.method`)"
        )));
    }
    if method == Some("level1") && entries.contains_key(REMOVED_LEVEL1_KEY) {
        return Err(de::Error::custom(format_args!(
            "the key `{REMOVED_LEVEL1_KEY}` was removed: `level1` compares row values \
             exactly; delete the key (`training.cut_selection.selection`)"
        )));
    }
    SelectionMethod::deserialize(Value::Object(entries))
        .map(Some)
        .map_err(de::Error::custom)
}

/// The entries of a `selection` object, read one key at a time so that a
/// repeated key is serde's `duplicate field` error: collecting into a
/// [`Value`] would keep the last occurrence silently.
struct SelectionEntries(Map<String, Value>);

impl<'de> Deserialize<'de> for SelectionEntries {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EntriesVisitor;

        impl<'de> Visitor<'de> for EntriesVisitor {
            type Value = SelectionEntries;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("internally tagged enum SelectionMethod")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if entries.contains_key(&key) {
                        return Err(de::Error::custom(format_args!("duplicate field `{key}`")));
                    }
                    let value = map.next_value()?;
                    entries.insert(key, value);
                }
                Ok(SelectionEntries(entries))
            }
        }

        deserializer.deserialize_map(EntriesVisitor)
    }
}

/// Row-selection method and its method-specific parameters.
///
/// The `method` key selects the method, and each method accepts only its own
/// parameters: a parameter of another method, a misspelled `method` and a
/// removed `method` are load-time errors.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum SelectionMethod {
    /// Level 1: every row the run adds to the row pool of a stage strictly
    /// between the first and the last records the trial state it was computed
    /// at, and keeps it for the whole run or, with `lookback_iterations`, for
    /// the iterations that key sets. At each stored state, the row with the
    /// highest value there owns the state, among the rows the run added to
    /// that pool and the active rows loaded into it; of rows whose values there
    /// are exactly equal, the one at the lowest pool position owns it. Every
    /// `check_frequency` iterations, a row the run added to such a pool is
    /// deactivated when it owns no stored state and reactivated when it is
    /// inactive and owns one. Values are compared exactly, with no tolerance,
    /// and a row of the current iteration is judged like any other. A row
    /// loaded from a policy keeps the activity it was loaded with.
    Level1 {
        /// Iterations between periodic pruning checks. Default `5`.
        #[serde(
            default = "default_check_frequency",
            deserialize_with = "deserialize_check_frequency"
        )]
        check_frequency: NonZeroU32,
        /// Iterations of the current run whose rows keep their trial states.
        /// Absent (default): the trial state of every row the run added is
        /// kept, and the memory of a run grows with its row count. A value `w`
        /// keeps the trial states of the rows added in the last `w` iterations
        /// of the current run and drops the older states, never a row: a row
        /// whose state was dropped stays in the pool and still competes at the
        /// states that remain. The memory holding the trial states of a pool is
        /// then bounded by `(lookback_iterations + check_frequency)` times the
        /// most rows the run can add to that pool in one iteration, times the
        /// state dimension, times 8 bytes. Must be at least `check_frequency`.
        #[serde(default, deserialize_with = "deserialize_lookback_iterations")]
        lookback_iterations: Option<NonZeroU32>,
    },
    /// Dynamic: a per-solve lazy loop, this software's own lazy row generation.
    /// From `start_iteration` on, each training solve starts with the seed rows
    /// described under `seed_window`, adds the candidate rows that its solution
    /// violates, at most `max_added_per_round` per round, and solves again; it
    /// returns when no candidate is violated. After a fixed number of rounds
    /// the per-round limit is lifted. A simulation solve starts with every
    /// active row, so `seed_window` and `candidate_recency` apply to training
    /// only. The method never deactivates a row, and the pool keeps every row.
    Dynamic {
        /// First 1-based iteration at which the lazy loop becomes active. Default `2`.
        #[serde(
            default = "default_start_iteration",
            deserialize_with = "deserialize_start_iteration"
        )]
        start_iteration: NonZeroU32,
        /// A solve starts with the active rows whose last binding was at most
        /// `seed_window` iterations ago (a row that has not bound counts from
        /// the iteration that added it), plus the rows added in the current
        /// iteration. `row_activity_tolerance` decides whether a row binds. `0`
        /// seeds the rows of the current iteration and those that bound in it.
        /// A row loaded from a policy counts from the start of the run under
        /// `warm_start` and for `policy.boundary` rows, and from its recorded
        /// iteration under `resume`. Default `5`.
        #[serde(default = "default_seed_window")]
        seed_window: u32,
        /// Absent (default): every active row is a candidate and a solve returns
        /// the optimum over them. A value `n` restricts the candidates to the
        /// rows added in the last `n` iterations and loaded rows, and the result
        /// is then the optimum over those rows together with the seed rows.
        #[serde(default, deserialize_with = "deserialize_candidate_recency")]
        candidate_recency: Option<NonZeroU32>,
        /// Maximum rows added per lazy-solve round. Default `10`.
        #[serde(
            default = "default_max_added_per_round",
            deserialize_with = "deserialize_max_added_per_round"
        )]
        max_added_per_round: NonZeroU32,
        /// A relative factor: a candidate row is added when its value exceeds
        /// the future-cost value by more than the factor times the larger of
        /// the two magnitudes. Absent (default): each phase uses the primal
        /// feasibility tolerance of its solver profile. A value must be finite
        /// and greater than `0`. It must also be less than 1 and at least the
        /// primal feasibility tolerance of every phase's solver profile.
        #[serde(default)]
        violation_tolerance: Option<f64>,
    },
}

fn default_check_frequency() -> NonZeroU32 {
    const { NonZeroU32::new(5).expect("5 is nonzero") }
}

fn default_start_iteration() -> NonZeroU32 {
    const { NonZeroU32::new(2).expect("2 is nonzero") }
}

fn default_seed_window() -> u32 {
    5
}

fn default_max_added_per_round() -> NonZeroU32 {
    const { NonZeroU32::new(10).expect("10 is nonzero") }
}

/// `value` as a count of at least 1, or the load error naming `key`, the full
/// dotted path of the config key.
fn positive_count<E: de::Error>(key: &str, value: u32) -> Result<NonZeroU32, E> {
    NonZeroU32::new(value)
        .ok_or_else(|| E::custom(format_args!("{key} must be >= 1 (got {value})")))
}

/// Deserializes a count of at least 1 through [`positive_count`].
fn required_positive_count<'de, D: Deserializer<'de>>(
    deserializer: D,
    key: &str,
) -> Result<NonZeroU32, D::Error> {
    positive_count(key, u32::deserialize(deserializer)?)
}

/// Deserializes an optional count of at least 1 through [`positive_count`].
fn optional_positive_count<'de, D: Deserializer<'de>>(
    deserializer: D,
    key: &str,
) -> Result<Option<NonZeroU32>, D::Error> {
    Option::<u32>::deserialize(deserializer)?
        .map(|value| positive_count(key, value))
        .transpose()
}

fn deserialize_check_frequency<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<NonZeroU32, D::Error> {
    required_positive_count(
        deserializer,
        "training.cut_selection.selection.check_frequency",
    )
}

fn deserialize_lookback_iterations<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NonZeroU32>, D::Error> {
    optional_positive_count(
        deserializer,
        "training.cut_selection.selection.lookback_iterations",
    )
}

fn deserialize_start_iteration<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<NonZeroU32, D::Error> {
    required_positive_count(
        deserializer,
        "training.cut_selection.selection.start_iteration",
    )
}

fn deserialize_candidate_recency<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NonZeroU32>, D::Error> {
    optional_positive_count(
        deserializer,
        "training.cut_selection.selection.candidate_recency",
    )
}

fn deserialize_max_added_per_round<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<NonZeroU32, D::Error> {
    required_positive_count(
        deserializer,
        "training.cut_selection.selection.max_added_per_round",
    )
}

/// LP solver settings (`config.json → training.solver`): retry policy plus
/// optional per-phase solver profiles.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainingSolverConfig {
    /// Maximum solver retry attempts before propagating a hard error.
    pub retry_max_attempts: u32,

    /// Total time budget in seconds across all retry attempts for one solve.
    pub retry_time_budget_seconds: f64,

    /// Backward-pass solver profile. Absent leaves the phase's built-in
    /// tuned profile.
    #[serde(default)]
    pub backward: Option<PhaseSolverProfileConfig>,

    /// Forward-pass solver profile. Absent leaves the phase's built-in
    /// tuned profile.
    #[serde(default)]
    pub forward: Option<PhaseSolverProfileConfig>,
}

impl Default for TrainingSolverConfig {
    fn default() -> Self {
        Self {
            retry_max_attempts: 5,
            retry_time_budget_seconds: 30.0,
            backward: None,
            forward: None,
        }
    }
}

/// Per-phase LP solver profile (`config.json → training.solver.backward` /
/// `.forward`, and `simulation.solver`).
///
/// Backend-agnostic. Every field is optional: an absent field leaves the
/// corresponding option at the phase's built-in tuned-profile value.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PhaseSolverProfileConfig {
    /// Dual simplex edge-weight strategy override.
    #[serde(default)]
    pub dual_edge_weight: Option<DualEdgeWeight>,

    /// Constraint-matrix scaling strategy override.
    #[serde(default)]
    pub scale: Option<ScaleStrategy>,

    /// Simplex pricing strategy override.
    #[serde(default)]
    pub price: Option<PriceStrategy>,

    /// Primal feasibility tolerance override.
    #[serde(default)]
    pub primal_feasibility_tolerance: Option<f64>,

    /// Dual feasibility tolerance override.
    #[serde(default)]
    pub dual_feasibility_tolerance: Option<f64>,

    /// Presolve mode override. Warm-started solves skip presolve regardless
    /// of this setting, so it affects only genuinely cold solves.
    #[serde(default)]
    pub presolve: Option<PresolveMode>,

    /// Simplex update-count limit override before a refactorization.
    #[serde(default)]
    pub simplex_update_limit: Option<u32>,

    /// Dual simplex cost-perturbation multiplier override.
    #[serde(default)]
    pub cost_perturbation: Option<f64>,

    /// Refactorization solution-error tolerance override.
    #[serde(default)]
    pub refactor_error_tolerance: Option<f64>,

    /// Matrix factorization pivot-threshold override.
    #[serde(default)]
    pub factor_pivot_threshold: Option<f64>,

    /// Warm-start override. This is a diagnostic setting: disabling it forces
    /// every solve cold.
    #[serde(default)]
    pub use_warm_start: Option<bool>,

    /// Dual steepest-edge weight log-error threshold override above which the
    /// solver falls back to Devex pricing.
    #[serde(default)]
    pub steepest_edge_devex_fallback_threshold: Option<f64>,
}

/// Presolve mode for a solver profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum PresolveMode {
    /// Presolve enabled.
    On,
    /// Presolve disabled.
    Off,
    /// Solver decides whether to presolve.
    Choose,
}

/// Parallel-execution settings (`config.json → training.parallelism`).
///
/// Groups the result-preserving knobs that shape how training work is
/// scheduled across workers, apart from the algorithm-semantics fields at the
/// `training` root. Thread count itself stays a CLI concern (`--threads`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ParallelismConfig {
    /// Backward-pass scheduler selection.
    pub backward_scheduler: BackwardScheduler,
}

/// Backward-pass scheduler and its scheduler-specific parameters
/// (`config.json → training.parallelism.backward_scheduler`).
///
/// The `method` key selects the scheduler, and each scheduler accepts only its
/// own parameters: a parameter of another scheduler is a load-time error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum BackwardScheduler {
    /// By-scenario backward scheduling (the default): each parallel work
    /// unit claims one whole trial point (a row of the backward work
    /// rectangle).
    // A braced variant, not a unit one: serde enforces `deny_unknown_fields`
    // only for braced variants of an internally tagged enum, and this variant
    // must reject `block_size`.
    ByScenario {},
    /// By-node backward scheduling: each parallel work unit claims one
    /// (trial point, opening-block) tile of the backward work rectangle.
    ByNode {
        /// Openings per block. Absent resolves per stage to `⌈|Ω_s|/2⌉` (half
        /// the openings, rounded up); a set value is clamped to
        /// `min(|Ω_s|, block_size)`.
        #[serde(default)]
        block_size: Option<NonZeroUsize>,
    },
}

impl Default for BackwardScheduler {
    fn default() -> Self {
        Self::ByScenario {}
    }
}

/// Dual simplex edge-weight (pricing) strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum DualEdgeWeight {
    /// Devex approximate edge weights.
    Devex,
    /// Exact steepest-edge weights.
    SteepestEdge,
    /// Dantzig most-negative-reduced-cost rule.
    Dantzig,
}

/// LP constraint-matrix scaling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ScaleStrategy {
    /// No scaling.
    Off,
    /// Solver-managed scaling.
    SolverScaling,
}

/// Simplex pricing (column-selection) strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum PriceStrategy {
    /// Row-wise pricing.
    Row,
    /// Row-wise pricing with hyper-sparse updates.
    RowHyperSparse,
}

/// Deserialized configuration for one entry in `training.stopping_rules[]`.
///
/// Uses a `"type"` discriminator field (internally tagged) with `snake_case`
/// variant names matching the JSON schema. `deny_unknown_fields` makes a
/// parameter belonging to a different rule type a load-time error rather
/// than a silently ignored key.
///
/// # Examples
///
/// ```
/// use novomodelo_io::config::StoppingRuleConfig;
///
/// let json = r#"{"type": "iteration_limit", "limit": 100}"#;
/// let rule: StoppingRuleConfig = serde_json::from_str(json).unwrap();
/// assert!(matches!(rule, StoppingRuleConfig::IterationLimit { limit: 100 }));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schema",
    schemars(
        description = "One entry of `training.stopping_rules`. The `type` key selects the rule; a key that belongs to another rule type is a load-time error."
    )
)]
pub enum StoppingRuleConfig {
    /// Stop after a fixed number of iterations. **Mandatory** — every rule set must
    /// contain at least one `iteration_limit` rule.
    IterationLimit {
        /// Maximum iteration count.
        limit: u32,
    },
    /// Stop after a wall-clock time limit.
    TimeLimit {
        /// Time limit in seconds.
        seconds: f64,
    },
    /// Stop when the lower bound stalls (relative improvement falls below tolerance).
    BoundStalling {
        /// Window size (number of past iterations to compare).
        iterations: u32,
        /// Relative improvement threshold.
        tolerance: f64,
    },
    /// Stop when the exact upper bound is within tolerance of the lower
    /// bound. At least one of `tolerance` / `relative_tolerance` must be
    /// present (checked at `from_config`). Admissible only under enumerated
    /// forward selection, where the upper bound is the exact bound a gap rule
    /// requires rather than a statistical estimate.
    Gap {
        /// Absolute gap tolerance, canonical R$.
        #[serde(default)]
        tolerance: Option<f64>,
        /// Relative gap tolerance in percent (e.g. `0.01` means 0.01%), compared
        /// against `100·gap / max(1, |lower_bound|)` — the same convention the
        /// reported `gap_percent` uses.
        #[serde(default)]
        relative_tolerance: Option<f64>,
    },
}

/// Upper-bound evaluation settings (`config.json → upper_bound_evaluation`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UpperBoundEvaluationConfig {
    /// Enable vertex-based inner approximation for upper bound computation.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// First iteration to compute the upper bound.
    #[serde(default)]
    pub initial_iteration: Option<u32>,

    /// Iterations between upper-bound evaluations.
    #[serde(default)]
    pub interval_iterations: Option<u32>,

    /// Lipschitz constant settings.
    #[serde(default)]
    pub lipschitz: LipschitzConfig,
}

/// Lipschitz constant settings for inner approximation.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LipschitzConfig {
    /// Computation mode: `"auto"`.
    #[serde(default)]
    pub mode: Option<String>,

    /// Fallback value when automatic computation fails.
    #[serde(default)]
    pub fallback_value: Option<f64>,

    /// Multiplicative safety margin applied to computed Lipschitz constants.
    #[serde(default)]
    pub scale_factor: Option<f64>,
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::{
        BackwardScheduler, DualEdgeWeight, NonZeroU32, NonZeroUsize, PresolveMode, PriceStrategy,
        RowSelectionConfig, ScaleStrategy, SelectionMethod, StoppingRuleConfig, TrainingConfig,
        TrainingSelection,
    };
    use crate::config::tests::removed_selection_method_names;

    /// A `dynamic` selection block round-trips through the tagged enum, with
    /// every method-specific field landing in the `Dynamic` variant.
    #[test]
    fn dynamic_selection_block_round_trips() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {
                "row_activity_tolerance": 1e-6,
                "max_active_per_stage": 4000,
                "selection": {
                    "method": "dynamic",
                    "start_iteration": 5,
                    "seed_window": 0,
                    "candidate_recency": 20,
                    "max_added_per_round": 3,
                    "violation_tolerance": 1e-9
                }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let cs = &cfg.cut_selection;
        assert_eq!(cs.row_activity_tolerance, 1e-6);
        assert_eq!(cs.max_active_per_stage, Some(4000));
        match cs.selection.as_ref().expect("selection present") {
            SelectionMethod::Dynamic {
                start_iteration,
                seed_window,
                candidate_recency,
                max_added_per_round,
                violation_tolerance,
            } => {
                assert_eq!(start_iteration.get(), 5);
                assert_eq!(*seed_window, 0);
                assert_eq!(*candidate_recency, NonZeroU32::new(20));
                assert_eq!(max_added_per_round.get(), 3);
                assert_eq!(*violation_tolerance, Some(1e-9));
            }
            other @ SelectionMethod::Level1 { .. } => panic!("expected Dynamic, got {other:?}"),
        }
    }

    /// A `level1` selection block round-trips and fills the variant defaults
    /// when its fields are omitted.
    #[test]
    fn level1_selection_block_round_trips_with_defaults() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": { "selection": { "method": "level1" } }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        match cfg
            .cut_selection
            .selection
            .as_ref()
            .expect("selection present")
        {
            SelectionMethod::Level1 {
                check_frequency,
                lookback_iterations,
            } => {
                assert_eq!(check_frequency.get(), 5);
                assert_eq!(*lookback_iterations, None);
            }
            other @ SelectionMethod::Dynamic { .. } => panic!("expected Level1, got {other:?}"),
        }
    }

    #[test]
    fn a_dynamic_selection_without_a_violation_tolerance_leaves_it_absent() {
        let json = training_json_with_selection(r#"{ "method": "dynamic" }"#);
        let cfg: TrainingConfig = serde_json::from_str(&json).unwrap();
        match cfg.cut_selection.selection.expect("selection present") {
            SelectionMethod::Dynamic {
                violation_tolerance,
                ..
            } => assert_eq!(violation_tolerance, None),
            other @ SelectionMethod::Level1 { .. } => panic!("expected Dynamic, got {other:?}"),
        }
    }

    #[test]
    fn a_level1_lookback_window_round_trips() {
        let json = training_json_with_selection(
            r#"{ "method": "level1", "check_frequency": 2, "lookback_iterations": 7 }"#,
        );
        let cfg: TrainingConfig = serde_json::from_str(&json).unwrap();
        match cfg.cut_selection.selection.expect("selection present") {
            SelectionMethod::Level1 {
                check_frequency,
                lookback_iterations,
            } => {
                assert_eq!(check_frequency.get(), 2);
                assert_eq!(lookback_iterations, NonZeroU32::new(7));
            }
            other @ SelectionMethod::Dynamic { .. } => panic!("expected Level1, got {other:?}"),
        }
    }

    #[test]
    fn a_zero_lookback_window_names_its_key() {
        let err = selection_error(r#"{ "method": "level1", "lookback_iterations": 0 }"#);
        assert!(
            err.contains(
                "training.cut_selection.selection.lookback_iterations must be >= 1 (got 0)"
            ),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn row_selection_defaults_the_activity_tolerance_to_1e_6() {
        let loaded: RowSelectionConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(loaded.row_activity_tolerance, 1e-6);
        assert_eq!(RowSelectionConfig::default().row_activity_tolerance, 1e-6);
    }

    #[test]
    fn a_zero_count_names_its_key() {
        for (method, key) in [
            ("level1", "check_frequency"),
            ("dynamic", "start_iteration"),
            ("dynamic", "candidate_recency"),
            ("dynamic", "max_added_per_round"),
        ] {
            let selection = |value: u32| format!(r#"{{ "method": "{method}", "{key}": {value} }}"#);
            serde_json::from_str::<TrainingConfig>(&training_json_with_selection(&selection(1)))
                .unwrap_or_else(|err| panic!("{key} = 1 must load: {err}"));
            let err = selection_error(&selection(0));
            assert!(
                err.contains(&format!("training.cut_selection.selection.{key}"))
                    && err.contains(">= 1"),
                "unexpected message for {key}: {err}"
            );
        }
    }

    #[test]
    fn omitting_selection_disables_row_selection() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {}
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.cut_selection.selection.is_none());
    }

    /// A parameter that belongs to a different method is a deserialize error
    /// under `deny_unknown_fields` (here `max_added_per_round` under `level1`).
    #[test]
    fn wrong_method_field_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": {
                "selection": { "method": "level1", "max_added_per_round": 3 }
            }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "a Dynamic-only field under level1 must be rejected"
        );
    }

    /// A misspelled `method` is an unknown-variant deserialize error.
    #[test]
    fn bad_method_string_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "cut_selection": { "selection": { "method": "dynmic" } }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(result.is_err(), "an unknown method tag must be rejected");
    }

    fn training_json_with_selection(selection: &str) -> String {
        format!(
            r#"{{
                "selection": {{ "method": "sampled", "forward_passes": 4 }},
                "stopping_rules": [{{ "type": "iteration_limit", "limit": 100 }}],
                "cut_selection": {{ "selection": {selection} }}
            }}"#
        )
    }

    /// A removed method is rejected with the message that names it and
    /// `level1`, with or without the parameters it used to take.
    #[test]
    fn a_removed_selection_method_names_level1() {
        for removed in removed_selection_method_names() {
            for selection in [
                format!(r#"{{ "method": "{removed}" }}"#),
                format!(r#"{{ "method": "{removed}", "check_frequency": 2 }}"#),
            ] {
                let err = serde_json::from_str::<TrainingConfig>(&training_json_with_selection(
                    &selection,
                ))
                .expect_err("a removed method must be rejected")
                .to_string();
                assert!(
                    err.contains(&format!(
                        "the row-selection method `{removed}` was removed: use `level1` \
                         (key `training.cut_selection.selection.method`)"
                    )),
                    "unexpected message for {selection}: {err}"
                );
            }
        }
    }

    /// A misspelled method stays serde's unknown-variant error listing the
    /// accepted spellings, and is not reported as a removed method.
    #[test]
    fn a_misspelled_selection_method_is_still_an_unknown_variant() {
        let err = serde_json::from_str::<TrainingConfig>(&training_json_with_selection(
            r#"{ "method": "lml2" }"#,
        ))
        .expect_err("a misspelled method must be rejected")
        .to_string();
        assert!(
            err.contains("unknown variant `lml2`, expected `level1` or `dynamic`"),
            "unexpected message: {err}"
        );
        assert!(!err.contains("was removed"), "unexpected message: {err}");
    }

    fn selection_error(selection: &str) -> String {
        serde_json::from_str::<TrainingConfig>(&training_json_with_selection(selection))
            .expect_err("the selection object must be rejected")
            .to_string()
    }

    /// The removed key is rejected by name under `level1`, even when it comes
    /// before the method tag.
    #[test]
    fn the_removed_tie_tolerance_key_is_rejected_by_name() {
        let err = selection_error(
            r#"{ "tie_tolerance": 1e6, "check_frequency": 2, "method": "level1" }"#,
        );
        assert!(
            err.contains(
                "the key `tie_tolerance` was removed: `level1` compares row values \
                 exactly; delete the key (`training.cut_selection.selection`)"
            ),
            "unexpected message: {err}"
        );
    }

    /// Any other key under `level1` stays serde's unknown-field error, whose
    /// list of expected fields is the variant's own.
    #[test]
    fn a_foreign_level1_key_is_still_an_unknown_field() {
        let err = selection_error(r#"{ "method": "level1", "tie_tolrance": 1 }"#);
        assert!(
            err.contains("unknown field `tie_tolrance`, expected `check_frequency`"),
            "unexpected message: {err}"
        );
        assert!(!err.contains("was removed"), "unexpected message: {err}");
    }

    /// A key repeated inside the selection object is a duplicate-field error
    /// naming the key, the method tag included.
    #[test]
    fn a_repeated_selection_key_is_a_duplicate_field() {
        let [removed, _] = removed_selection_method_names();
        for (selection, key) in [
            (
                r#"{ "method": "level1", "check_frequency": 2, "check_frequency": 3 }"#.to_string(),
                "check_frequency",
            ),
            (
                format!(r#"{{ "method": "{removed}", "method": "level1" }}"#),
                "method",
            ),
        ] {
            let err = selection_error(&selection);
            assert!(
                err.contains(&format!("duplicate field `{key}`")),
                "unexpected message for {selection}: {err}"
            );
        }
    }

    /// A full `training.solver.backward` block round-trips: every per-field
    /// override lands in `PhaseSolverProfileConfig`, and the sibling `forward`
    /// phase stays absent.
    #[test]
    fn backward_solver_profile_block_round_trips() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "solver": {
                "backward": {
                    "dual_edge_weight": "steepest_edge",
                    "scale": "solver_scaling",
                    "price": "row",
                    "primal_feasibility_tolerance": 1e-7
                }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let backward = cfg.solver.backward.as_ref().expect("backward present");
        assert_eq!(
            backward.dual_edge_weight,
            Some(DualEdgeWeight::SteepestEdge)
        );
        assert_eq!(backward.scale, Some(ScaleStrategy::SolverScaling));
        assert_eq!(backward.price, Some(PriceStrategy::Row));
        assert_eq!(backward.primal_feasibility_tolerance, Some(1e-7));
        assert!(cfg.solver.forward.is_none());
    }

    /// A `training.solver.backward` block setting `presolve`, `use_warm_start`,
    /// and `factor_pivot_threshold` round-trips into the new fields.
    #[test]
    fn backward_solver_profile_new_fields_round_trip() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "solver": {
                "backward": {
                    "presolve": "off",
                    "use_warm_start": false,
                    "factor_pivot_threshold": 0.2
                }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let backward = cfg.solver.backward.as_ref().expect("backward present");
        assert_eq!(backward.presolve, Some(PresolveMode::Off));
        assert_eq!(backward.use_warm_start, Some(false));
        assert_eq!(backward.factor_pivot_threshold, Some(0.2));
        assert!(backward.dual_feasibility_tolerance.is_none());
        assert!(backward.simplex_update_limit.is_none());
        assert!(backward.cost_perturbation.is_none());
        assert!(backward.refactor_error_tolerance.is_none());
        assert!(backward.steepest_edge_devex_fallback_threshold.is_none());
    }

    /// A `training.solver.forward` block round-trips independently of `backward`.
    #[test]
    fn forward_solver_profile_block_round_trips() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "solver": {
                "forward": {
                    "price": "row_hyper_sparse",
                    "dual_edge_weight": "dantzig"
                }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let forward = cfg.solver.forward.as_ref().expect("forward present");
        assert_eq!(forward.price, Some(PriceStrategy::RowHyperSparse));
        assert_eq!(forward.dual_edge_weight, Some(DualEdgeWeight::Dantzig));
        assert!(cfg.solver.backward.is_none());
    }

    /// An unknown field under `backward` (here the misspelling `dual_edge_weght`)
    /// is a deserialize error under `deny_unknown_fields`.
    #[test]
    fn backward_solver_profile_unknown_field_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "solver": { "backward": { "dual_edge_weght": "devex" } }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "an unknown field under backward must be rejected"
        );
    }

    /// The misspelled `presolv` field under `backward` is a deserialize error
    /// under `deny_unknown_fields`.
    #[test]
    fn backward_solver_profile_presolv_typo_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "solver": { "backward": { "presolv": "off" } }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "the presolv typo under backward must be rejected"
        );
    }

    /// A misspelled enum value is an unknown-variant deserialize error; the
    /// informal `curtis_reid` spelling is not a valid `scale` value.
    #[test]
    fn backward_solver_profile_bad_enum_value_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "solver": { "backward": { "scale": "curtis_reid" } }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(result.is_err(), "an unknown scale value must be rejected");
    }

    #[test]
    fn backward_scheduler_defaults_to_by_scenario_when_absent() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }]
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.parallelism.backward_scheduler,
            BackwardScheduler::ByScenario {}
        );
    }

    /// `{"method": "by_node", "block_size": 4}` round-trips into the
    /// `ByNode` variant carrying `Some(4)`.
    #[test]
    fn by_node_scheduler_and_block_size_round_trip() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "parallelism": {
                "backward_scheduler": { "method": "by_node", "block_size": 4 }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.parallelism.backward_scheduler,
            BackwardScheduler::ByNode {
                block_size: NonZeroUsize::new(4)
            }
        );
    }

    /// A `by_node` scheduler without `block_size` round-trips into
    /// `block_size: None` (the per-stage `⌈|Ω_s|/2⌉` resolution).
    #[test]
    fn by_node_scheduler_without_block_size_round_trips() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "parallelism": {
                "backward_scheduler": { "method": "by_node" }
            }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.parallelism.backward_scheduler,
            BackwardScheduler::ByNode { block_size: None }
        );
    }

    /// `block_size` under `by_scenario` is a deserialize error under
    /// `deny_unknown_fields` — the invalid combination is unrepresentable, not
    /// warned-and-ignored.
    #[test]
    fn block_size_under_by_scenario_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "parallelism": {
                "backward_scheduler": { "method": "by_scenario", "block_size": 4 }
            }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "block_size under by_scenario must be rejected"
        );
    }

    /// The retired `trial_point` / `opening_block` spellings are unknown-variant
    /// deserialize errors — no `serde(alias)` accepts them (clean break, no
    /// deprecated-with-fallback).
    #[test]
    fn retired_scheduler_spellings_are_deserialize_error() {
        for method in ["trial_point", "opening_block"] {
            let json = format!(
                r#"{{
                    "selection": {{ "method": "sampled", "forward_passes": 4 }},
                    "stopping_rules": [{{ "type": "iteration_limit", "limit": 100 }}],
                    "parallelism": {{
                        "backward_scheduler": {{ "method": "{method}" }}
                    }}
                }}"#
            );
            let result = serde_json::from_str::<TrainingConfig>(&json);
            assert!(
                result.is_err(),
                "retired scheduler spelling '{method}' must be an unknown-variant error"
            );
        }
    }

    /// A misspelled scheduler `method` is an unknown-variant deserialize error.
    #[test]
    fn unknown_scheduler_method_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "parallelism": {
                "backward_scheduler": { "method": "by_nod" }
            }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(result.is_err(), "an unknown method tag must be rejected");
    }

    /// `block_size: 0` is an out-of-range deserialize error (`NonZeroUsize`).
    #[test]
    fn block_size_zero_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "parallelism": {
                "backward_scheduler": { "method": "by_node", "block_size": 0 }
            }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "block_size = 0 must be rejected by NonZeroUsize"
        );
    }

    /// The retired root-level scheduler keys are hard-rejected, never silently
    /// ignored: the scheduler lives under `training.parallelism`, and the
    /// backward solve order is intrinsic (no config field selects it).
    #[test]
    fn removed_root_scheduler_keys_are_rejected() {
        for stale in [
            r#""backward_scheduler": "opening_block""#,
            r#""opening_block_size": 4"#,
            r#""backward_opening_order": "sigma_key""#,
        ] {
            let json = format!(
                r#"{{
                    "selection": {{ "method": "sampled", "forward_passes": 4 }},
                    "stopping_rules": [{{ "type": "iteration_limit", "limit": 100 }}],
                    {stale}
                }}"#
            );
            let result = serde_json::from_str::<TrainingConfig>(&json);
            assert!(
                result.is_err(),
                "removed root key must be rejected, got Ok for: {stale}"
            );
        }
    }

    /// A `sampled` phase selection round-trips into the `Sampled` variant.
    #[test]
    fn sampled_selection_round_trips() {
        let json = r#"{
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "selection": { "method": "sampled", "forward_passes": 8 }
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.selection,
            Some(TrainingSelection::Sampled { forward_passes: 8 })
        );
    }

    /// The removed root `forward_passes` alias is an unknown-field deserialize
    /// error under `deny_unknown_fields`; the count lives solely in the
    /// `selection.sampled` arm.
    #[test]
    fn root_forward_passes_alias_is_deserialize_error() {
        let alias = r#"{
            "forward_passes": 4,
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }]
        }"#;
        assert!(
            serde_json::from_str::<TrainingConfig>(alias).is_err(),
            "root forward_passes must be rejected as an unknown field"
        );

        let arm = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }]
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(arm).unwrap();
        assert_eq!(
            cfg.selection,
            Some(TrainingSelection::Sampled { forward_passes: 4 })
        );
    }

    /// A count under `enumerated` is unrepresentable — `deny_unknown_fields` on
    /// the braced variant rejects it at parse time, so the invalid pairing is
    /// never gated at runtime.
    #[test]
    fn enumerated_selection_with_count_is_deserialize_error() {
        let json = r#"{
            "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
            "selection": { "method": "enumerated", "forward_passes": 8 }
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "a count under enumerated must be rejected as unrepresentable"
        );
    }

    /// A parameter belonging to a different stopping-rule type is a
    /// deserialize error under `deny_unknown_fields` (here `seconds` under
    /// `iteration_limit`), never a silently ignored key.
    #[test]
    fn wrong_stopping_rule_field_is_deserialize_error() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [
                { "type": "iteration_limit", "limit": 100, "seconds": 60.0 }
            ]
        }"#;
        let result = serde_json::from_str::<TrainingConfig>(json);
        assert!(
            result.is_err(),
            "a time_limit-only field under iteration_limit must be rejected"
        );
    }

    /// A `gap` rule with only `tolerance` set deserializes into
    /// `StoppingRuleConfig::Gap`.
    #[test]
    fn gap_stopping_rule_tolerance_only_round_trips() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "gap", "tolerance": 1000.0 }]
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let rules = cfg.stopping_rules.expect("stopping_rules present");
        assert!(matches!(
            rules[0],
            StoppingRuleConfig::Gap {
                tolerance: Some(t),
                relative_tolerance: None
            } if (t - 1000.0).abs() < f64::EPSILON
        ));
    }

    /// A `gap` rule with only `relative_tolerance` set deserializes into
    /// `StoppingRuleConfig::Gap`.
    #[test]
    fn gap_stopping_rule_relative_tolerance_only_round_trips() {
        let json = r#"{
            "selection": { "method": "sampled", "forward_passes": 4 },
            "stopping_rules": [{ "type": "gap", "relative_tolerance": 0.01 }]
        }"#;
        let cfg: TrainingConfig = serde_json::from_str(json).unwrap();
        let rules = cfg.stopping_rules.expect("stopping_rules present");
        assert!(matches!(
            rules[0],
            StoppingRuleConfig::Gap {
                tolerance: None,
                relative_tolerance: Some(rt)
            } if (rt - 0.01).abs() < f64::EPSILON
        ));
    }
}
