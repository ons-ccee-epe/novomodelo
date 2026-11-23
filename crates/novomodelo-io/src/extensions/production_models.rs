//! Parsing for `system/hydro_production_models.json` — per-hydro production model configuration.
//!
//! [`parse_production_models`] reads `system/hydro_production_models.json` and returns a sorted
//! `Vec<ProductionModelConfig>` describing the HPF model selection for each configured hydro.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/production_models.schema.json",
//!   "production_models": [
//!     {
//!       "hydro_id": 0,
//!       "selection_mode": "stage_ranges",
//!       "stage_ranges": [
//!         {
//!           "start_stage_id": 0, "end_stage_id": 24,
//!           "model": "fpha",
//!           "fpha_config": { "source": "computed" }
//!         }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Selection modes
//!
//! - **`stage_ranges`**: Each stage maps to a model via explicit `[start, end)` ranges.
//! - **`seasonal`**: Each stage maps to a model via its season index. Seasons not listed
//!   fall back to `default_model`.
//!
//! ## Output ordering
//!
//! Results are sorted by `hydro_id` ascending. Duplicate `hydro_id` values are rejected
//! as a `SchemaError`.
//!
//! ## Validation
//!
//! Per-entry constraints enforced by this parser:
//!
//! - No two entries share the same `hydro_id`.
//! - For `stage_ranges` mode: `start_stage_id <= end_stage_id` when `end_stage_id` is not null.
//! - In `fitting_window`: absolute bounds (`volume_min_hm3` / `volume_max_hm3`) and percentile
//!   bounds (`volume_min_percentile` / `volume_max_percentile`) are mutually exclusive.
//! - `productivity_mw_per_m3s` is **rejected** for `"fpha"` entries (both stage-range and
//!   seasonal); FPHA derives productivity from its hyperplane geometry.
//! - `productivity_mw_per_m3s` is **optional** for `"constant_productivity"` entries. When
//!   omitted or `null`, the value is expected to be supplied by
//!   `system/hydro_energy_productivity.parquet`; cross-file resolution is enforced by
//!   `validation::productivity_resolution`.
//! - When `productivity_mw_per_m3s` is present for a non-FPHA entry it must be finite and
//!   non-negative (`>= 0.0`). A value of `0.0` is accepted as a planned-outage marker.
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3.
//! - Cross-validation that `source: "precomputed"` hydros have FPHA hyperplanes — Layer 3/5.
//! - That exactly one source (JSON or parquet) provides `productivity_mw_per_m3s` for each
//!   `(hydro, stage)` pair — `validation::productivity_resolution`.

use novomodelo_core::{EntityId, HydroGenerationModel, Stage};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;
use std::slice;

use crate::LoadError;

/// Production model configuration for one hydro plant.
///
/// Loaded from `system/hydro_production_models.json`. Specifies how the hydro
/// production function (HPF) model is selected across stages or seasons.
///
/// # Examples
///
/// ```
/// use novomodelo_io::extensions::{ProductionModelConfig, SelectionMode};
/// use novomodelo_core::EntityId;
///
/// let config = ProductionModelConfig {
///     hydro_id: EntityId::from(0),
///     selection_mode: SelectionMode::StageRanges {
///         ranges: vec![],
///     },
/// };
/// assert_eq!(config.hydro_id, EntityId::from(0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionModelConfig {
    /// Hydro plant this configuration applies to.
    pub hydro_id: EntityId,
    /// How the model variant is selected for each stage.
    pub selection_mode: SelectionMode,
}

/// Parsed contents of `system/hydro_production_models.json`.
///
/// Bundles the per-hydro production model configs with the optional file-level
/// FPHA plane-reduction block. `plane_reduction` is `None` when the file carries
/// no `fpha_plane_reduction` key (the off-by-default case).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProductionModelFile {
    /// Per-hydro production model configurations, sorted by `hydro_id` ascending.
    pub configs: Vec<ProductionModelConfig>,
    /// File-level similar-hyperplane reduction config, applied uniformly to
    /// every plant. `None` ⇒ no reduction.
    pub plane_reduction: Option<PlaneReductionConfig>,
}

/// Model selection strategy for a hydro plant.
///
/// The two variants are mutually exclusive within a single hydro entry.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionMode {
    /// Models are selected by stage ID ranges.
    StageRanges {
        /// Ordered list of stage range descriptors.
        ranges: Vec<StageRange>,
    },
    /// Models are selected by season index, with a fallback default.
    Seasonal {
        /// Fallback model for seasons not listed in `seasons`.
        default_model: HydroGenerationModel,
        /// Season-specific overrides.
        seasons: Vec<SeasonConfig>,
    },
}

/// A stage range descriptor for the `stage_ranges` selection mode.
#[derive(Debug, Clone, PartialEq)]
pub struct StageRange {
    /// First stage (inclusive) to which this entry applies.
    pub start_stage_id: i32,
    /// Last stage (inclusive) to which this entry applies. `None` means "until end of horizon".
    pub end_stage_id: Option<i32>,
    /// Production model for this entry.
    pub model: HydroGenerationModel,
    /// FPHA configuration, required when `model == "fpha"`.
    pub fpha_config: Option<FphaColumnLayout>,
    /// Optional reference operating volume, a sibling of `fpha_config`. `None`
    /// when not declared; a default is applied later in resolution, not here.
    pub reference_volume: Option<ReferenceVolume>,
    /// Per-stage productivity coefficient [MW/(m³/s)]; see the module doc for the
    /// optional / `0.0`-outage / `None`-for-fpha rules.
    pub productivity_mw_per_m3s: Option<f64>,
}

/// A season-specific model descriptor for the `seasonal` selection mode.
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonConfig {
    /// Season index (0-based, matching `stages.json` season map).
    pub season_id: i32,
    /// Production model for this entry.
    pub model: HydroGenerationModel,
    /// FPHA configuration, required when `model == "fpha"`.
    pub fpha_config: Option<FphaColumnLayout>,
    /// Optional reference operating volume, a sibling of `fpha_config`. `None`
    /// when not declared; a default is applied later in resolution, not here.
    pub reference_volume: Option<ReferenceVolume>,
    /// Per-season productivity coefficient [MW/(m³/s)]; see the module doc for the
    /// optional / `0.0`-outage / `None`-for-fpha rules.
    pub productivity_mw_per_m3s: Option<f64>,
}

/// Configuration for the FPHA production function model.
#[derive(Debug, Clone, PartialEq)]
pub struct FphaColumnLayout {
    /// `"computed"` (fit from topology) or `"precomputed"` (from `fpha_hyperplanes.parquet`).
    pub source: String,
    /// Number of volume discretization points used when computing hyperplanes.
    pub volume_discretization_points: Option<i32>,
    /// Number of turbine flow discretization points used when computing hyperplanes.
    pub turbine_discretization_points: Option<i32>,
    /// Optional fitting window within the plant's storage range; absent = the
    /// whole range, `min_storage_hm3` to `max_storage_hm3`.
    pub fitting_window: Option<FittingWindow>,
    /// Upper end (m³/s) of the spillage range the spillage coefficient is fit over.
    /// Finite and `> 0.0` when present.
    pub spillage_fit_max_m3s: Option<f64>,
    /// Upper end (m³/s) of the turbined-flow range the planes are fit over.
    /// Finite and `> 0.0` when present.
    pub turbined_fit_max_m3s: Option<f64>,
}

/// Similar-hyperplane reduction configuration for FPHA planes.
///
/// Selects how near-parallel / near-coincident FPHA planes are merged into
/// their mean hyperplane to shrink the LP. The two variants are mutually
/// exclusive (the input picks one `method`); both are applied uniformly to
/// every plant. Absent in the input means no reduction.
#[derive(Debug, Clone, PartialEq)]
pub enum PlaneReductionConfig {
    /// Merge planes whose normal vectors lie within `tolerance_deg` of each
    /// other. `tolerance_deg` is an angle in degrees, in `[0.0, 90.0]`.
    Angle {
        /// Maximum angle (degrees) between plane normals to treat them as
        /// parallel.
        tolerance_deg: f64,
    },
    /// Merge planes whose mean-squared distance over `n_samples` sampled points
    /// stays within `tolerance_pct`.
    Distance {
        /// Maximum relative MSE distance (fraction) to treat two planes as
        /// coincident.
        tolerance_pct: f64,
        /// Number of sample points used to estimate the distance.
        n_samples: u32,
    },
}

/// Volume fitting window for computed FPHA hyperplanes.
///
/// Each bound is clamped to the plant's storage range, `min_storage_hm3` to
/// `max_storage_hm3`, and a bound left unset takes that range's endpoint.
///
/// Absolute bounds (`volume_min_hm3` / `volume_max_hm3`) and percentile bounds
/// (`volume_min_percentile` / `volume_max_percentile`) are mutually exclusive.
#[derive(Debug, Clone, PartialEq)]
pub struct FittingWindow {
    /// Explicit minimum volume for fitting (hm³). Mutually exclusive with `volume_min_percentile`.
    pub volume_min_hm3: Option<f64>,
    /// Explicit maximum volume for fitting (hm³). Mutually exclusive with `volume_max_percentile`.
    pub volume_max_hm3: Option<f64>,
    /// Minimum as a percentile of the storage range. Mutually exclusive with `volume_min_hm3`.
    pub volume_min_percentile: Option<f64>,
    /// Maximum as a percentile of the storage range. Mutually exclusive with `volume_max_hm3`.
    pub volume_max_percentile: Option<f64>,
}

/// Reference operating volume for a stage range or season.
///
/// The input declares the reference volume either as an absolute storage value
/// (hm³) or as a percentile of the plant's operating range; the two are mutually
/// exclusive. Resolution of the percentile form to an absolute value happens in a
/// later stage, not here.
#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceVolume {
    /// Absolute reference volume `[hm³]`. Finite and `> 0.0`.
    AbsoluteHm3(f64),
    /// Reference volume as a percentile of the operating range, in `[0.0, 1.0]`.
    Percentile(f64),
}

// ── Per-stage resolution ──────────────────────────────────────────────────────

impl ProductionModelConfig {
    /// Resolve this hydro's production-model entry at `stage`: the only per-stage
    /// `SelectionMode` match, so every consumer asking which model a hydro uses at
    /// a stage gets one answer.
    ///
    /// `StageRanges`: the first range covering `stage.id`, else every field
    /// `None`. `Seasonal`: the season matching `stage.season_id`, else
    /// `default_model` with `fpha_config`, `reference_volume`, and `productivity`
    /// all `None` — `default_model` is the config's declared fallback, not a
    /// coverage gap, so it still resolves a model where a `StageRanges` gap would
    /// not.
    #[must_use]
    pub fn resolve_stage(&self, stage: &Stage) -> StageProductionResolution<'_> {
        match &self.selection_mode {
            SelectionMode::StageRanges { ranges } => {
                for range in ranges {
                    let after_start = stage.id >= range.start_stage_id;
                    let before_end = range.end_stage_id.is_none_or(|end| stage.id <= end);
                    if after_start && before_end {
                        return StageProductionResolution {
                            model: Some(range.model),
                            fpha_config: range.fpha_config.as_ref(),
                            reference_volume: range.reference_volume.as_ref(),
                            productivity: range.productivity_mw_per_m3s,
                        };
                    }
                }
                StageProductionResolution {
                    model: None,
                    fpha_config: None,
                    reference_volume: None,
                    productivity: None,
                }
            }
            SelectionMode::Seasonal {
                default_model,
                seasons,
            } => {
                if let Some(season_id) = stage.season_id {
                    for season in seasons {
                        if i32::try_from(season_id).is_ok_and(|sid| sid == season.season_id) {
                            return StageProductionResolution {
                                model: Some(season.model),
                                fpha_config: season.fpha_config.as_ref(),
                                reference_volume: season.reference_volume.as_ref(),
                                productivity: season.productivity_mw_per_m3s,
                            };
                        }
                    }
                }
                StageProductionResolution {
                    model: Some(*default_model),
                    fpha_config: None,
                    reference_volume: None,
                    productivity: None,
                }
            }
        }
    }

    /// Enumerate every declared `SelectionMode` entry, plus — for `Seasonal` — one
    /// synthetic `default_model` entry with `fpha_config: None`: the only
    /// config-level `SelectionMode` match, the counterpart of
    /// [`Self::resolve_stage`].
    ///
    /// The synthetic entry makes `default_model` count in every predicate below,
    /// as a season miss resolves to it: a fold over the declared seasons alone
    /// misses a `default_model == "fpha"` config whose listed seasons are all
    /// non-FPHA.
    #[must_use]
    pub fn selection_entries(&self) -> SelectionEntries<'_> {
        match &self.selection_mode {
            SelectionMode::StageRanges { ranges } => SelectionEntries::StageRanges(ranges.iter()),
            SelectionMode::Seasonal {
                default_model,
                seasons,
            } => SelectionEntries::Seasonal {
                seasons: seasons.iter(),
                default_model: Some(*default_model),
            },
        }
    }

    /// `true` when any entry, `default_model` included, selects the FPHA model.
    #[must_use]
    pub fn uses_fpha(&self) -> bool {
        self.selection_entries()
            .any(|entry| entry.model == HydroGenerationModel::Fpha)
    }

    /// `true` when any entry fits its FPHA hyperplanes from geometry
    /// (`source: "computed"`).
    #[must_use]
    pub fn uses_computed_fpha(&self) -> bool {
        self.selection_entries()
            .any(|entry| entry.fpha_config.is_some_and(|f| f.source == "computed"))
    }

    /// `true` when any entry reads its FPHA hyperplanes from
    /// `fpha_hyperplanes.parquet`: `source: "precomputed"`, or an FPHA entry with
    /// no `fpha_config` — the `Seasonal` `default_model` case, precomputed by
    /// construction. Not the complement of [`Self::uses_computed_fpha`]: a config
    /// with one computed and one precomputed entry is both.
    #[must_use]
    pub fn uses_precomputed_fpha(&self) -> bool {
        self.selection_entries().any(|entry| {
            entry.fpha_config.is_some_and(|f| f.source == "precomputed")
                || (entry.model == HydroGenerationModel::Fpha && entry.fpha_config.is_none())
        })
    }

    /// `true` when the entry resolved at `stage` selects the FPHA model.
    #[must_use]
    pub fn uses_fpha_at(&self, stage: &Stage) -> bool {
        self.resolve_stage(stage).model == Some(HydroGenerationModel::Fpha)
    }
}

// ── Fit-domain resolution ─────────────────────────────────────────────────────

/// A storage range whose lower end is not at or below its upper end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvertedStorageRange {
    /// Lower end (hm³).
    pub v_min_hm3: f64,
    /// Upper end (hm³).
    pub v_max_hm3: f64,
}

impl FphaColumnLayout {
    /// Resolve the volume window `[v_min, v_max]` (hm³) the planes are fit over.
    ///
    /// Each bound is its absolute value, else its percentile of
    /// `[min_storage_hm3, max_storage_hm3]`, else that range's endpoint, then
    /// clamped to the range; the parser refuses an absolute value and a percentile
    /// on one bound, so the absolute value is read first without a conflict. A
    /// `None` layout, or one without a `fitting_window`, resolves to the storage
    /// range. A window that is a single point is not an error.
    ///
    /// # Errors
    ///
    /// [`InvertedStorageRange`] carrying the storage bounds when they are inverted
    /// or NaN, or the clamped window when `v_max < v_min`.
    #[must_use = "the resolved window or its refusal"]
    pub fn resolve_fitting_window(
        layout: Option<&Self>,
        min_storage_hm3: f64,
        max_storage_hm3: f64,
    ) -> Result<(f64, f64), InvertedStorageRange> {
        let storage_range_is_ordered = min_storage_hm3 <= max_storage_hm3;
        if !storage_range_is_ordered {
            return Err(InvertedStorageRange {
                v_min_hm3: min_storage_hm3,
                v_max_hm3: max_storage_hm3,
            });
        }

        let (v_min, v_max) = match layout.and_then(|l| l.fitting_window.as_ref()) {
            None => (min_storage_hm3, max_storage_hm3),
            Some(fw) => {
                let storage_range = max_storage_hm3 - min_storage_hm3;

                let v_min_raw = if let Some(abs) = fw.volume_min_hm3 {
                    abs
                } else if let Some(pct) = fw.volume_min_percentile {
                    min_storage_hm3 + pct * storage_range
                } else {
                    min_storage_hm3
                };

                let v_max_raw = if let Some(abs) = fw.volume_max_hm3 {
                    abs
                } else if let Some(pct) = fw.volume_max_percentile {
                    min_storage_hm3 + pct * storage_range
                } else {
                    max_storage_hm3
                };

                (
                    v_min_raw.clamp(min_storage_hm3, max_storage_hm3),
                    v_max_raw.clamp(min_storage_hm3, max_storage_hm3),
                )
            }
        };

        if v_max < v_min {
            return Err(InvertedStorageRange {
                v_min_hm3: v_min,
                v_max_hm3: v_max,
            });
        }
        Ok((v_min, v_max))
    }

    /// Resolve the upper end (m³/s) of the spillage range the spillage coefficient
    /// is fit over: `spillage_fit_max_m3s` when the layout sets it, else twice the
    /// long-term mean inflow when that is positive, else twice `max_turbined_m3s`.
    ///
    /// The mean test is strict: a hydro without history passes `0.0` and takes the
    /// turbine-limit fallback, where `>= 0.0` would give a zero spillage range.
    #[must_use]
    pub fn resolve_spillage_fit_max_m3s(
        layout: Option<&Self>,
        long_term_mean_inflow_m3s: f64,
        max_turbined_m3s: f64,
    ) -> f64 {
        if let Some(s_max) = layout.and_then(|l| l.spillage_fit_max_m3s) {
            s_max
        } else if long_term_mean_inflow_m3s > 0.0 {
            2.0 * long_term_mean_inflow_m3s
        } else {
            2.0 * max_turbined_m3s
        }
    }
}

/// One `(hydro, stage)` production-model entry, resolved by
/// [`ProductionModelConfig::resolve_stage`] and borrowed from the config.
#[derive(Debug, Clone, PartialEq)]
pub struct StageProductionResolution<'a> {
    /// Model, or `None` for a `StageRanges` gap — the sole case with no default
    /// (a `Seasonal` season miss always resolves to `default_model`).
    pub model: Option<HydroGenerationModel>,
    /// FPHA configuration of the resolved entry.
    pub fpha_config: Option<&'a FphaColumnLayout>,
    /// Reference operating volume of the resolved entry.
    pub reference_volume: Option<&'a ReferenceVolume>,
    /// Constant productivity of the resolved entry [MW/(m³/s)].
    pub productivity: Option<f64>,
}

/// One `SelectionMode` entry's borrowed `(model, fpha_config)` view, yielded by
/// [`ProductionModelConfig::selection_entries`].
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionEntry<'a> {
    /// Model the entry selects.
    pub model: HydroGenerationModel,
    /// FPHA configuration of the entry; `None` on the synthetic `default_model`
    /// entry.
    pub fpha_config: Option<&'a FphaColumnLayout>,
}

/// Enum-dispatched iterator behind [`ProductionModelConfig::selection_entries`]
/// over the two `SelectionMode` arms, avoiding a `Box<dyn Iterator>`.
#[derive(Debug, Clone)]
pub enum SelectionEntries<'a> {
    /// The ranges of a `StageRanges` config.
    StageRanges(slice::Iter<'a, StageRange>),
    /// The seasons of a `Seasonal` config, then its `default_model` once.
    Seasonal {
        /// Seasons not yet yielded.
        seasons: slice::Iter<'a, SeasonConfig>,
        /// `default_model`, taken after the last season.
        default_model: Option<HydroGenerationModel>,
    },
}

impl<'a> Iterator for SelectionEntries<'a> {
    type Item = SelectionEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::StageRanges(ranges) => ranges.next().map(|range| SelectionEntry {
                model: range.model,
                fpha_config: range.fpha_config.as_ref(),
            }),
            Self::Seasonal {
                seasons,
                default_model,
            } => seasons
                .next()
                .map(|season| SelectionEntry {
                    model: season.model,
                    fpha_config: season.fpha_config.as_ref(),
                })
                .or_else(|| {
                    default_model.take().map(|model| SelectionEntry {
                        model,
                        fpha_config: None,
                    })
                }),
        }
    }
}

/// Per-hydro production model configuration loaded from
/// `system/hydro_production_models.json`.
///
/// Specifies how the hydro production function (HPF) model variant is selected
/// for each stage or season. Two selection modes are supported:
///
/// - `stage_ranges`: maps each stage to a model via explicit `[start, end]`
///   intervals.
/// - `seasonal`: maps each stage to a model via its season index, with a
///   fallback default.
///
/// Each hydro may appear at most once. Results are sorted by `hydro_id`
/// ascending.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawProductionModelFile {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of per-hydro production model configurations. Each `hydro_id`
    /// must be unique.
    production_models: Vec<RawProductionModel>,

    /// Optional file-level FPHA plane-reduction block, applied uniformly to
    /// every plant. Absent ⇒ no reduction. Carries a `method` tag selecting
    /// the `angle` or `distance` reduction method and its tolerance.
    #[serde(default)]
    fpha_plane_reduction: Option<RawPlaneReductionConfig>,
}

/// Production model configuration for one hydro plant.
///
/// The `selection_mode` field discriminates between two layouts:
/// `stage_ranges` carries a stage-range array, while `seasonal` carries a
/// `default_model` plus a `seasons` override list.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawProductionModel {
    /// Hydro plant identifier. Must be unique within the file.
    hydro_id: i32,

    /// Tagged-union payload for the model selection.
    #[serde(flatten)]
    selection: RawSelectionMode,
}

/// Model selection layout for a hydro plant, discriminated by the
/// `selection_mode` JSON field.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(tag = "selection_mode", rename_all = "snake_case")]
enum RawSelectionMode {
    /// Stage-range selection: each stage maps to a model via explicit
    /// `[start, end]` ranges.
    StageRanges {
        /// Ordered list of stage range descriptors.
        stage_ranges: Vec<RawStageRange>,
    },
    /// Seasonal selection: each stage maps to a model via its season index.
    Seasonal {
        /// Fallback model for seasons not listed in `seasons`.
        default_model: HydroGenerationModel,
        /// Season-specific model overrides.
        seasons: Vec<RawSeasonConfig>,
    },
}

/// Stage range descriptor for the `stage_ranges` selection mode.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStageRange {
    /// First stage (inclusive) to which this entry applies. Must be <=
    /// `end_stage_id` when `end_stage_id` is set.
    start_stage_id: i32,
    /// Last stage (inclusive) to which this entry applies. `null` = until end
    /// of horizon.
    end_stage_id: Option<i32>,
    /// Production model: `"constant_productivity"` or `"fpha"`.
    model: HydroGenerationModel,
    /// FPHA configuration. Required when `model` is `"fpha"`. Absent or null
    /// otherwise.
    fpha_config: Option<RawFphaColumnLayout>,
    /// Reference operating volume for this stage range, a sibling of
    /// `fpha_config` (not nested). Set exactly one of `volume_hm3` (absolute,
    /// hm³) or `percentile` (`[0.0, 1.0]`). Absent or null = no reference volume
    /// declared.
    reference_volume: Option<RawReferenceVolume>,
    /// Per-stage productivity coefficient (MW/(m³/s)). Optional for a
    /// `"constant_productivity"` model; when absent or null the value is
    /// expected from `system/hydro_energy_productivity.parquet`. When present
    /// must be finite and `>= 0.0` (`0.0` marks a planned outage). Must be
    /// absent or null for `"fpha"`.
    productivity_mw_per_m3s: Option<f64>,
}

/// Season-specific model descriptor for the `seasonal` selection mode.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSeasonConfig {
    /// Season index (0-based, matching the `stages.json` season map).
    season_id: i32,
    /// Production model: `"constant_productivity"` or `"fpha"`.
    model: HydroGenerationModel,
    /// FPHA configuration. Required when `model` is `"fpha"`. Absent or null
    /// otherwise.
    fpha_config: Option<RawFphaColumnLayout>,
    /// Reference operating volume for this season, a sibling of `fpha_config`
    /// (not nested). Set exactly one of `volume_hm3` (absolute, hm³) or
    /// `percentile` (`[0.0, 1.0]`). Absent or null = no reference volume
    /// declared.
    reference_volume: Option<RawReferenceVolume>,
    /// Per-season productivity coefficient (MW/(m³/s)). Optional for a
    /// `"constant_productivity"` model; when absent or null the value is
    /// expected from `system/hydro_energy_productivity.parquet`. When present
    /// must be finite and `>= 0.0` (`0.0` marks a planned outage). Must be
    /// absent or null for `"fpha"`.
    productivity_mw_per_m3s: Option<f64>,
}

/// Configuration for the FPHA production function model.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFphaColumnLayout {
    /// Hyperplane source: `"computed"` (fit from topology) or
    /// `"precomputed"` (from `fpha_hyperplanes.parquet`).
    source: String,
    /// Number of volume discretization points used when computing hyperplanes.
    /// Absent = algorithm default (5).
    volume_discretization_points: Option<i32>,
    /// Number of turbine flow discretization points used when computing
    /// hyperplanes. Absent = algorithm default (5).
    turbine_discretization_points: Option<i32>,
    /// Optional volume fitting window for hyperplane computation. Absent or
    /// null = the plant's storage range, `min_storage_hm3` to
    /// `max_storage_hm3`. `system/hydro_geometry.parquet` must cover that
    /// range whether or not a window is set.
    fitting_window: Option<RawFittingWindow>,
    /// Upper end of the spillage range (m³/s) the spillage coefficient is fit
    /// over. Absent = twice the hydro's long-term mean inflow, or twice its
    /// maximum turbined flow when it has no inflow history.
    spillage_fit_max_m3s: Option<f64>,
    /// Upper end of the turbined-flow range (m³/s) the planes are fit over.
    /// Absent = the hydro's maximum turbined flow.
    turbined_fit_max_m3s: Option<f64>,
}

/// File-level FPHA plane-reduction block, selected by the `method` key.
///
/// `{ "method": "angle", "tolerance_deg": <number> }` merges planes whose
/// normals are within `tolerance_deg` degrees, while
/// `{ "method": "distance", "tolerance_pct": <number>, "n_samples": <integer> }`
/// merges planes whose sampled mean-squared distance stays within
/// `tolerance_pct`. A tolerance key that belongs to the other method is rejected.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum RawPlaneReductionConfig {
    /// Normal-vector angle method. Merges planes whose normals lie within
    /// `tolerance_deg` of each other.
    Angle {
        /// Maximum angle (degrees) between plane normals to treat them as
        /// parallel. Must be finite and in `[0.0, 90.0]` inclusive.
        tolerance_deg: f64,
    },
    /// Mean-squared-distance method. Merges planes whose sampled MSE distance
    /// stays within `tolerance_pct`.
    Distance {
        /// Maximum relative MSE distance (fraction) to treat two planes as
        /// coincident. Must be finite and `>= 0.0`.
        tolerance_pct: f64,
        /// Number of sample points used to estimate the distance. Must be `>= 1`.
        n_samples: u32,
    },
}

/// Volume fitting window restricting the range used for FPHA hyperplane
/// computation.
///
/// Each bound is clamped to the plant's storage range, `min_storage_hm3` to
/// `max_storage_hm3`, and a bound left unset takes that range's endpoint.
///
/// Absolute bounds (`volume_min_hm3` / `volume_max_hm3`) and percentile bounds
/// (`volume_min_percentile` / `volume_max_percentile`) are mutually exclusive:
/// set one pair or the other, not both for the same bound.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[allow(clippy::struct_field_names)]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFittingWindow {
    /// Explicit minimum volume for fitting (hm³). Mutually exclusive with
    /// `volume_min_percentile`.
    volume_min_hm3: Option<f64>,
    /// Explicit maximum volume for fitting (hm³). Mutually exclusive with
    /// `volume_max_percentile`.
    volume_max_hm3: Option<f64>,
    /// Minimum as a percentile of the storage range. Mutually exclusive
    /// with `volume_min_hm3`.
    volume_min_percentile: Option<f64>,
    /// Maximum as a percentile of the storage range. Mutually exclusive
    /// with `volume_max_hm3`.
    volume_max_percentile: Option<f64>,
}

/// Reference operating volume declared on a stage range or season.
///
/// Set exactly one of `volume_hm3` (absolute, hm³) or `percentile` (a fraction
/// of the operating range). The two are mutually exclusive; setting both, or
/// neither, is rejected during validation.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReferenceVolume {
    /// Absolute reference volume (hm³). Mutually exclusive with `percentile`.
    /// When present must be finite and `> 0.0`.
    volume_hm3: Option<f64>,
    /// Reference volume as a percentile of the operating range. Mutually
    /// exclusive with `volume_hm3`. When present must be finite and in
    /// `[0.0, 1.0]`.
    percentile: Option<f64>,
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse `system/hydro_production_models.json` into a [`ProductionModelFile`].
///
/// # Errors
///
/// | Condition                                                   | Error variant              |
/// |------------------------------------------------------------ |--------------------------- |
/// | File not found or permission denied                         | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or unrecognised `selection_mode`        | [`LoadError::ParseError`] / [`LoadError::SchemaError`] |
/// | Duplicate `hydro_id`                                        | [`LoadError::SchemaError`] |
/// | `start_stage_id > end_stage_id` (when `end_stage_id` set)  | [`LoadError::SchemaError`] |
/// | Both absolute and percentile fitting bounds set             | [`LoadError::SchemaError`] |
/// | `fpha_plane_reduction` tolerance out of range / `n_samples < 1` | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use novomodelo_io::extensions::parse_production_models;
/// use std::path::Path;
///
/// let file = parse_production_models(Path::new("system/hydro_production_models.json"))
///     .expect("valid production models file");
/// println!("loaded {} hydro model configs", file.configs.len());
/// ```
pub fn parse_production_models(path: &Path) -> Result<ProductionModelFile, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawProductionModelFile = serde_json::from_str(&raw_text).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("unknown variant") {
            let field = if msg.contains("`constant_productivity`") {
                "model"
            } else {
                "selection_mode"
            };
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: field.to_string(),
                message: msg,
            }
        } else {
            LoadError::parse(path, msg)
        }
    })?;

    validate_production_models(
        &raw.production_models,
        raw.fpha_plane_reduction.as_ref(),
        path,
    )?;

    let mut configs: Vec<ProductionModelConfig> = raw
        .production_models
        .into_iter()
        .map(convert_production_model)
        .collect();

    configs.sort_by_key(|c| c.hydro_id.0);

    let plane_reduction = raw
        .fpha_plane_reduction
        .as_ref()
        .map(convert_plane_reduction);

    Ok(ProductionModelFile {
        configs,
        plane_reduction,
    })
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all cross-entry and per-entry constraints on raw production model data.
fn validate_production_models(
    models: &[RawProductionModel],
    plane_reduction: Option<&RawPlaneReductionConfig>,
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen_ids: HashSet<i32> = HashSet::new();

    for (entry_idx, model) in models.iter().enumerate() {
        if !seen_ids.insert(model.hydro_id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("production_models[{entry_idx}].hydro_id"),
                message: format!(
                    "duplicate hydro_id {} — each hydro may appear at most once",
                    model.hydro_id
                ),
            });
        }

        match &model.selection {
            RawSelectionMode::StageRanges { stage_ranges } => {
                for (range_idx, range) in stage_ranges.iter().enumerate() {
                    validate_stage_range(range, model.hydro_id, entry_idx, range_idx, path)?;
                }
            }
            RawSelectionMode::Seasonal { seasons, .. } => {
                for (season_idx, season) in seasons.iter().enumerate() {
                    validate_model_fields(
                        season.model,
                        season.productivity_mw_per_m3s,
                        season.fpha_config.as_ref(),
                        season.reference_volume.as_ref(),
                        model.hydro_id,
                        &format!("production_models[{entry_idx}].seasons[{season_idx}]"),
                        path,
                    )?;
                }
            }
        }
    }

    if let Some(reduction) = plane_reduction {
        validate_plane_reduction(reduction, path)?;
    }

    Ok(())
}

/// Validate one stage range descriptor.
fn validate_stage_range(
    range: &RawStageRange,
    hydro_id: i32,
    entry_idx: usize,
    range_idx: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if let Some(end) = range.end_stage_id
        && range.start_stage_id > end
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!(
                "production_models[{entry_idx}].stage_ranges[{range_idx}].start_stage_id"
            ),
            message: format!(
                "stage_ranges entry has start_stage_id ({}) > end_stage_id ({}); \
                     start_stage_id must be <= end_stage_id",
                range.start_stage_id, end
            ),
        });
    }

    validate_model_fields(
        range.model,
        range.productivity_mw_per_m3s,
        range.fpha_config.as_ref(),
        range.reference_volume.as_ref(),
        hydro_id,
        &format!("production_models[{entry_idx}].stage_ranges[{range_idx}]"),
        path,
    )
}

/// Validate the model/productivity/`fpha_config`/`reference_volume` fields
/// shared by a stage range and a season entry.
fn validate_model_fields(
    model: HydroGenerationModel,
    productivity_mw_per_m3s: Option<f64>,
    fpha_config: Option<&RawFphaColumnLayout>,
    reference_volume: Option<&RawReferenceVolume>,
    hydro_id: i32,
    field_prefix: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let field_base = format!("{field_prefix}.productivity_mw_per_m3s");

    let is_fpha = model == HydroGenerationModel::Fpha;
    if is_fpha && productivity_mw_per_m3s.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_base,
            message: "productivity_mw_per_m3s must not be set when model is 'fpha'".to_string(),
        });
    }

    // `0.0` is a planned-outage marker; reject only negative or non-finite.
    if !is_fpha
        && let Some(val) = productivity_mw_per_m3s
        && (val < 0.0 || !val.is_finite())
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_base,
            message: format!("productivity_mw_per_m3s must be finite and non-negative, got {val}"),
        });
    }

    if let Some(cfg) = fpha_config {
        validate_fpha_discretization_counts(cfg, hydro_id, field_prefix, path)?;
        validate_fitting_window(
            cfg,
            &format!("{field_prefix}.fpha_config.fitting_window"),
            path,
        )?;
        validate_fit_maxima(cfg, hydro_id, field_prefix, path)?;
    }

    if let Some(rv) = reference_volume {
        validate_reference_volume(rv, &format!("{field_prefix}.reference_volume"), path)?;
    }

    Ok(())
}

/// Reject any FPHA discretization count that is present and negative.
///
/// A negative count is not rejected downstream — the `unwrap_or(_) as usize`
/// resolution wraps it into a huge count past the low-count guards — so it must
/// fail here at the input boundary.
fn validate_fpha_discretization_counts(
    cfg: &RawFphaColumnLayout,
    hydro_id: i32,
    field_prefix: &str,
    path: &Path,
) -> Result<(), LoadError> {
    for (value, name) in [
        (
            cfg.volume_discretization_points,
            "volume_discretization_points",
        ),
        (
            cfg.turbine_discretization_points,
            "turbine_discretization_points",
        ),
    ] {
        if let Some(count) = value
            && count < 0
        {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{field_prefix}.fpha_config.{name}"),
                message: format!(
                    "{name} must be non-negative, got {count} for hydro_id {hydro_id}"
                ),
            });
        }
    }

    Ok(())
}

fn validate_fit_maxima(
    cfg: &RawFphaColumnLayout,
    hydro_id: i32,
    field_prefix: &str,
    path: &Path,
) -> Result<(), LoadError> {
    for (value, name) in [
        (cfg.spillage_fit_max_m3s, "spillage_fit_max_m3s"),
        (cfg.turbined_fit_max_m3s, "turbined_fit_max_m3s"),
    ] {
        if let Some(value) = value
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{field_prefix}.fpha_config.{name}"),
                message: format!(
                    "{name} must be finite and > 0.0, got {value} for hydro_id {hydro_id}"
                ),
            });
        }
    }

    Ok(())
}

/// Reject a fitting window that sets an absolute bound and its percentile
/// counterpart together (the two are mutually exclusive per bound).
fn validate_fitting_window(
    cfg: &RawFphaColumnLayout,
    field_prefix: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let Some(fw) = &cfg.fitting_window else {
        return Ok(());
    };

    if fw.volume_min_hm3.is_some() && fw.volume_min_percentile.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: "mutually exclusive bounds: volume_min_hm3 and volume_min_percentile \
                      cannot both be set; use absolute bounds OR percentiles, not both"
                .to_string(),
        });
    }

    if fw.volume_max_hm3.is_some() && fw.volume_max_percentile.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: "mutually exclusive bounds: volume_max_hm3 and volume_max_percentile \
                      cannot both be set; use absolute bounds OR percentiles, not both"
                .to_string(),
        });
    }

    Ok(())
}

/// Validate a reference-volume entry's absolute-XOR-percentile invariant.
///
/// A value that passes here has exactly one of `volume_hm3` / `percentile` set —
/// `convert_reference_volume` relies on this to disambiguate.
fn validate_reference_volume(
    rv: &RawReferenceVolume,
    field_prefix: &str,
    path: &Path,
) -> Result<(), LoadError> {
    if rv.volume_hm3.is_some() && rv.percentile.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: "mutually exclusive fields: volume_hm3 and percentile cannot both be \
                      set; use an absolute volume OR a percentile, not both"
                .to_string(),
        });
    }

    if rv.volume_hm3.is_none() && rv.percentile.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: "reference_volume must set exactly one of volume_hm3 or percentile"
                .to_string(),
        });
    }

    if let Some(vol) = rv.volume_hm3
        && (!vol.is_finite() || vol <= 0.0)
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: format!("volume_hm3 must be finite and > 0.0, got {vol}"),
        });
    }

    if let Some(pct) = rv.percentile
        && (!pct.is_finite() || !(0.0..=1.0).contains(&pct))
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: format!("percentile must be finite and in [0.0, 1.0], got {pct}"),
        });
    }

    Ok(())
}

/// Validate the per-method tolerance ranges of the file-level plane-reduction
/// block.
///
/// Method exclusivity is already enforced structurally by serde's `method` tag
/// and `deny_unknown_fields`; this is the config-layer range check.
fn validate_plane_reduction(
    reduction: &RawPlaneReductionConfig,
    path: &Path,
) -> Result<(), LoadError> {
    match reduction {
        RawPlaneReductionConfig::Angle { tolerance_deg } => {
            if !tolerance_deg.is_finite() || *tolerance_deg < 0.0 || *tolerance_deg > 90.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: "fpha_plane_reduction".to_string(),
                    message: format!(
                        "angle tolerance_deg must be finite and in [0, 90], got {tolerance_deg}"
                    ),
                });
            }
        }
        RawPlaneReductionConfig::Distance {
            tolerance_pct,
            n_samples,
        } => {
            if !tolerance_pct.is_finite() || *tolerance_pct < 0.0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: "fpha_plane_reduction".to_string(),
                    message: format!(
                        "distance tolerance_pct must be finite and >= 0, got {tolerance_pct}"
                    ),
                });
            }
            if *n_samples < 1 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: "fpha_plane_reduction".to_string(),
                    message: format!("distance n_samples must be >= 1, got {n_samples}"),
                });
            }
        }
    }

    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert a validated raw production model entry into the public type.
fn convert_production_model(raw: RawProductionModel) -> ProductionModelConfig {
    let selection_mode = match raw.selection {
        RawSelectionMode::StageRanges { stage_ranges } => SelectionMode::StageRanges {
            ranges: stage_ranges.into_iter().map(convert_stage_range).collect(),
        },
        RawSelectionMode::Seasonal {
            default_model,
            seasons,
        } => SelectionMode::Seasonal {
            default_model,
            seasons: seasons.into_iter().map(convert_season_config).collect(),
        },
    };

    ProductionModelConfig {
        hydro_id: EntityId::from(raw.hydro_id),
        selection_mode,
    }
}

fn convert_stage_range(raw: RawStageRange) -> StageRange {
    StageRange {
        start_stage_id: raw.start_stage_id,
        end_stage_id: raw.end_stage_id,
        model: raw.model,
        fpha_config: raw.fpha_config.map(convert_fpha_column_layout),
        reference_volume: raw.reference_volume.as_ref().map(convert_reference_volume),
        productivity_mw_per_m3s: raw.productivity_mw_per_m3s,
    }
}

fn convert_season_config(raw: RawSeasonConfig) -> SeasonConfig {
    SeasonConfig {
        season_id: raw.season_id,
        model: raw.model,
        fpha_config: raw.fpha_config.map(convert_fpha_column_layout),
        reference_volume: raw.reference_volume.as_ref().map(convert_reference_volume),
        productivity_mw_per_m3s: raw.productivity_mw_per_m3s,
    }
}

/// Pick the public variant from whichever field is `Some`.
///
/// `validate_reference_volume` guarantees exactly one is `Some`; the `unwrap_or` default is unreachable.
fn convert_reference_volume(raw: &RawReferenceVolume) -> ReferenceVolume {
    match raw.volume_hm3 {
        Some(vol) => ReferenceVolume::AbsoluteHm3(vol),
        None => ReferenceVolume::Percentile(raw.percentile.unwrap_or_default()),
    }
}

fn convert_fpha_column_layout(raw: RawFphaColumnLayout) -> FphaColumnLayout {
    FphaColumnLayout {
        source: raw.source,
        volume_discretization_points: raw.volume_discretization_points,
        turbine_discretization_points: raw.turbine_discretization_points,
        fitting_window: raw.fitting_window.map(|fw| FittingWindow {
            volume_min_hm3: fw.volume_min_hm3,
            volume_max_hm3: fw.volume_max_hm3,
            volume_min_percentile: fw.volume_min_percentile,
            volume_max_percentile: fw.volume_max_percentile,
        }),
        spillage_fit_max_m3s: raw.spillage_fit_max_m3s,
        turbined_fit_max_m3s: raw.turbined_fit_max_m3s,
    }
}

fn convert_plane_reduction(raw: &RawPlaneReductionConfig) -> PlaneReductionConfig {
    match raw {
        RawPlaneReductionConfig::Angle { tolerance_deg } => PlaneReductionConfig::Angle {
            tolerance_deg: *tolerance_deg,
        },
        RawPlaneReductionConfig::Distance {
            tolerance_pct,
            n_samples,
        } => PlaneReductionConfig::Distance {
            tolerance_pct: *tolerance_pct,
            n_samples: *n_samples,
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::match_wildcard_for_single_variants,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::{make_stage, write_json};

    // ── valid stage_ranges mode ───────────────────────────────────────────────

    /// Given a valid file with one hydro using `stage_ranges` mode, returns Ok with
    /// one entry containing the correct SelectionMode variant.
    #[test]
    fn test_valid_stage_ranges_mode() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "fpha",
                "fpha_config": {
                  "source": "computed",
                  "volume_discretization_points": 7,
                  "turbine_discretization_points": 15,
                  "fitting_window": { "volume_min_hm3": null, "volume_max_hm3": null }
                }
              },
              {
                "start_stage_id": 25, "end_stage_id": null,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.9
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;

        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.hydro_id, EntityId::from(0));
        match &m.selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert_eq!(ranges.len(), 2);
                assert_eq!(ranges[0].start_stage_id, 0);
                assert_eq!(ranges[0].end_stage_id, Some(24));
                assert_eq!(ranges[0].model, HydroGenerationModel::Fpha);
                let fpha = ranges[0].fpha_config.as_ref().unwrap();
                assert_eq!(fpha.source, "computed");
                assert_eq!(fpha.volume_discretization_points, Some(7));
                assert_eq!(fpha.turbine_discretization_points, Some(15));
                // Fitting window present but both bounds null
                let fw = fpha.fitting_window.as_ref().unwrap();
                assert!(fw.volume_min_hm3.is_none());
                assert!(fw.volume_max_hm3.is_none());

                assert_eq!(ranges[1].start_stage_id, 25);
                assert!(ranges[1].end_stage_id.is_none());
                assert_eq!(ranges[1].model, HydroGenerationModel::ConstantProductivity);
                assert!(ranges[1].fpha_config.is_none());
                assert_eq!(ranges[1].productivity_mw_per_m3s, Some(0.9));
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    // ── valid seasonal mode ───────────────────────────────────────────────────

    /// Given a valid file with one hydro using `seasonal` mode, returns Ok with one
    /// entry containing the correct SelectionMode variant with default_model and seasons.
    #[test]
    fn test_valid_seasonal_mode() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 5,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [
              {
                "season_id": 0,
                "model": "fpha",
                "fpha_config": { "source": "computed", "volume_discretization_points": 5 }
              },
              {
                "season_id": 1, "model": "fpha",
                "fpha_config": { "source": "computed", "turbine_discretization_points": 10 }
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;

        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.hydro_id, EntityId::from(5));
        match &m.selection_mode {
            SelectionMode::Seasonal {
                default_model,
                seasons,
            } => {
                assert_eq!(*default_model, HydroGenerationModel::ConstantProductivity);
                assert_eq!(seasons.len(), 2);
                assert_eq!(seasons[0].season_id, 0);
                assert_eq!(seasons[0].model, HydroGenerationModel::Fpha);
                let fpha0 = seasons[0].fpha_config.as_ref().unwrap();
                assert_eq!(fpha0.source, "computed");
                assert_eq!(fpha0.volume_discretization_points, Some(5));
                assert!(fpha0.turbine_discretization_points.is_none());

                assert_eq!(seasons[1].season_id, 1);
                let fpha1 = seasons[1].fpha_config.as_ref().unwrap();
                assert_eq!(fpha1.turbine_discretization_points, Some(10));
                assert!(fpha1.volume_discretization_points.is_none());
            }
            other => panic!("expected Seasonal, got: {other:?}"),
        }
    }

    // ── mixed — one stage_ranges, one seasonal ────────────────────────────────

    /// Given a valid file with one hydro in stage_ranges mode and one in seasonal mode,
    /// returns Ok with 2 entries sorted by hydro_id.
    #[test]
    fn test_mixed_modes_sorted_by_hydro_id() {
        let json = r#"{
          "production_models": [
            {
              "hydro_id": 10,
              "selection_mode": "seasonal",
              "default_model": "constant_productivity",
              "seasons": []
            },
            {
              "hydro_id": 3,
              "selection_mode": "stage_ranges",
              "stage_ranges": [
                {
                  "start_stage_id": 0, "end_stage_id": null,
                  "model": "constant_productivity",
                  "productivity_mw_per_m3s": 0.8
                }
              ]
            }
          ]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;

        assert_eq!(models.len(), 2);
        // Sorted by hydro_id ascending
        assert_eq!(models[0].hydro_id, EntityId::from(3));
        assert_eq!(models[1].hydro_id, EntityId::from(10));
        assert!(matches!(
            models[0].selection_mode,
            SelectionMode::StageRanges { .. }
        ));
        assert!(matches!(
            models[1].selection_mode,
            SelectionMode::Seasonal { .. }
        ));
    }

    // ── duplicate hydro_id -> SchemaError ─────────────────────────────────────

    /// Duplicate hydro_id in the file -> SchemaError mentioning the duplicate.
    #[test]
    fn test_duplicate_hydro_id() {
        let json = r#"{
          "production_models": [
            {
              "hydro_id": 5,
              "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "fpha", "fpha_config": { "source": "computed" } }]
            },
            {
              "hydro_id": 5,
              "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }]
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("hydro_id"),
                    "field should mention hydro_id, got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should mention duplicate, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── invalid stage range (start > end) -> SchemaError ─────────────────────

    /// stage_ranges with start_stage_id > end_stage_id -> SchemaError with
    /// field containing "stage_ranges" and message containing "start_stage_id".
    #[test]
    fn test_invalid_stage_range_start_greater_than_end() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 25, "end_stage_id": 10,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.9
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("stage_ranges"),
                    "field should contain 'stage_ranges', got: {field}"
                );
                assert!(
                    message.contains("start_stage_id"),
                    "message should contain 'start_stage_id', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── equal start == end is valid ───────────────────────────────────────────

    /// start_stage_id == end_stage_id is valid (single-stage range).
    #[test]
    fn test_stage_range_start_equals_end_is_valid() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 5, "end_stage_id": 5,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.9
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let result = parse_production_models(f.path());
        assert!(
            result.is_ok(),
            "equal start==end should be valid, got: {result:?}"
        );
    }

    // ── mutually exclusive fitting window -> SchemaError ─────────────────────

    /// Both volume_min_hm3 and volume_min_percentile set -> SchemaError with
    /// message containing "mutually exclusive".
    #[test]
    fn test_mutually_exclusive_fitting_window_min() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "fitting_window": {
                  "volume_min_hm3": 1000.0,
                  "volume_max_hm3": null,
                  "volume_min_percentile": 0.1,
                  "volume_max_percentile": null
                }
              }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("mutually exclusive"),
                    "message should contain 'mutually exclusive', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Both volume_max_hm3 and volume_max_percentile set -> SchemaError with
    /// message containing "mutually exclusive".
    #[test]
    fn test_mutually_exclusive_fitting_window_max() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "fitting_window": {
                  "volume_min_hm3": null,
                  "volume_max_hm3": 8000.0,
                  "volume_min_percentile": null,
                  "volume_max_percentile": 0.9
                }
              }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("mutually exclusive"),
                    "message should contain 'mutually exclusive', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Both absolute and percentile set in seasonal mode -> SchemaError.
    #[test]
    fn test_mutually_exclusive_fitting_window_seasonal() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 1,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [{
              "season_id": 0,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "fitting_window": {
                  "volume_min_hm3": 500.0,
                  "volume_min_percentile": 0.2
                }
              }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    // ── None path wrapper returns empty vec ───────────────────────────────────
    // (tested in extensions/mod.rs — see load_production_models)

    // ── file not found -> IoError ────────────────────────────────────────────

    /// Non-existent path -> IoError.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/path/hydro_production_models.json");
        let err = parse_production_models(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── unknown selection_mode -> SchemaError ─────────────────────────────────

    /// Unknown selection_mode -> SchemaError (tagged union deserialization failure).
    #[test]
    fn test_unknown_selection_mode() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "unknown_mode"
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for unknown selection_mode, got: {err:?}"
        );
    }

    fn assert_model_name_rejected(model_json: &str) {
        let json = format!(
            r#"{{
          "production_models": [{{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{{ "start_stage_id": 0, "end_stage_id": null, {model_json} }}]
          }}]
        }}"#
        );
        let f = write_json(&json);
        let err = parse_production_models(f.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => assert_eq!(field, "model"),
            other => panic!("expected SchemaError on model for {model_json}, got: {other:?}"),
        }
    }

    #[test]
    fn misspelled_model_name_is_rejected() {
        assert_model_name_rejected(r#""model": "fhpa""#);
    }

    #[test]
    fn linearized_head_model_name_is_rejected() {
        assert_model_name_rejected(r#""model": "linearized_head""#);
    }

    // ── empty production_models array -> Ok(vec![]) ──────────────────────────

    /// An empty `production_models` array deserialises to `Ok(Vec::new())`.
    #[test]
    fn test_empty_array_returns_empty_vec() {
        let json = r#"{ "production_models": [] }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        assert!(models.is_empty());
    }

    // ── declaration-order invariance ─────────────────────────────────────────

    /// Reordering the entries in the JSON does not change the output ordering.
    #[test]
    fn test_declaration_order_invariance() {
        let json_asc = r#"{
          "production_models": [
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 5, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 99, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] }
          ]
        }"#;
        let json_desc = r#"{
          "production_models": [
            { "hydro_id": 99, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 5, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] },
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }] }
          ]
        }"#;
        let f_asc = write_json(json_asc);
        let f_desc = write_json(json_desc);
        let models_asc = parse_production_models(f_asc.path()).unwrap().configs;
        let models_desc = parse_production_models(f_desc.path()).unwrap().configs;

        let ids_asc: Vec<i32> = models_asc.iter().map(|m| m.hydro_id.0).collect();
        let ids_desc: Vec<i32> = models_desc.iter().map(|m| m.hydro_id.0).collect();
        assert_eq!(
            ids_asc, ids_desc,
            "output order must be hydro_id-sorted regardless of input"
        );
        assert_eq!(ids_asc, vec![1, 5, 99]);
    }

    // ── fpha_config without fitting_window is valid ───────────────────────────

    /// FPHA config with no fitting_window field at all is valid.
    #[test]
    fn test_fpha_config_without_fitting_window() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": { "source": "precomputed" }
            }]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        assert_eq!(models.len(), 1);
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                let fpha = ranges[0].fpha_config.as_ref().unwrap();
                assert_eq!(fpha.source, "precomputed");
                assert!(fpha.fitting_window.is_none());
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    // ── productivity_mw_per_m3s tests ─────────────────────────────────────────

    /// `constant_productivity` stage range with a positive value parses OK and
    /// exposes `productivity_mw_per_m3s = Some(0.85)`.
    #[test]
    fn constant_productivity_requires_coefficient() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.85
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert_eq!(ranges[0].productivity_mw_per_m3s, Some(0.85));
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// Non-FPHA stage range with `productivity_mw_per_m3s` omitted parses to `Ok` with
    /// `productivity_mw_per_m3s: None`. The parquet override is expected to supply the value.
    #[test]
    fn test_non_fpha_stage_range_without_productivity_is_accepted() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity"
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert!(
                    ranges[0].productivity_mw_per_m3s.is_none(),
                    "expected None when field is omitted, got: {:?}",
                    ranges[0].productivity_mw_per_m3s
                );
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// Non-FPHA stage range with `productivity_mw_per_m3s: null` parses to `Ok` with
    /// `productivity_mw_per_m3s: None`. The parquet override is expected to supply the value.
    #[test]
    fn test_non_fpha_stage_range_with_null_productivity_is_accepted() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": null
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert!(
                    ranges[0].productivity_mw_per_m3s.is_none(),
                    "expected None when field is null, got: {:?}",
                    ranges[0].productivity_mw_per_m3s
                );
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// `fpha` stage range with `productivity_mw_per_m3s` set -> `SchemaError`
    /// with the exact message `"productivity_mw_per_m3s must not be set when model is 'fpha'"`.
    #[test]
    fn fpha_rejects_coefficient() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "fpha",
                "fpha_config": { "source": "computed" },
                "productivity_mw_per_m3s": 1.0
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("productivity_mw_per_m3s"),
                    "field should contain 'productivity_mw_per_m3s', got: {field}"
                );
                assert_eq!(
                    message, "productivity_mw_per_m3s must not be set when model is 'fpha'",
                    "message must match exactly"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Validation rejects non-positive `productivity_mw_per_m3s`.
    #[test]
    fn test_productivity_negative_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": -1.0
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    /// `productivity_mw_per_m3s = 0.0` is accepted as a planned-outage marker.
    #[test]
    fn test_productivity_zero_accepted() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.0
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let parsed = parse_production_models(f.path())
            .expect("zero productivity must be accepted as a planned-outage marker")
            .configs;
        let SelectionMode::StageRanges { ranges } = &parsed[0].selection_mode else {
            panic!("expected StageRanges");
        };
        assert_eq!(ranges[0].productivity_mw_per_m3s, Some(0.0));
    }

    /// Seasonal mode with `productivity_mw_per_m3s` parses correctly.
    #[test]
    fn test_seasonal_productivity_mw_per_m3s() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [
              {
                "season_id": 0,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 0.75
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::Seasonal { seasons, .. } => {
                assert_eq!(seasons[0].productivity_mw_per_m3s, Some(0.75));
            }
            other => panic!("expected Seasonal, got: {other:?}"),
        }
    }

    /// Non-FPHA seasonal entry with `productivity_mw_per_m3s` omitted parses to `Ok` with
    /// `productivity_mw_per_m3s: None`. The parquet override is expected to supply the value.
    #[test]
    fn test_non_fpha_seasonal_without_productivity_is_accepted() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [
              {
                "season_id": 0,
                "model": "constant_productivity"
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::Seasonal { seasons, .. } => {
                assert!(
                    seasons[0].productivity_mw_per_m3s.is_none(),
                    "expected None when field is omitted, got: {:?}",
                    seasons[0].productivity_mw_per_m3s
                );
            }
            other => panic!("expected Seasonal, got: {other:?}"),
        }
    }

    /// An FPHA stage range with `productivity_mw_per_m3s` set is rejected.
    #[test]
    fn test_fpha_stage_range_with_productivity_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "fpha",
                "fpha_config": { "source": "computed" },
                "productivity_mw_per_m3s": 0.9
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("must not be set when model is 'fpha'"),
                    "message should mention fpha rejection, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A negative `productivity_mw_per_m3s` is rejected.
    #[test]
    fn test_negative_productivity_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_mw_per_m3s": -0.1
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("productivity_mw_per_m3s must be finite and non-negative"),
                    "message should mention non-negative requirement, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── fpha_plane_reduction tests ────────────────────────────────────────────

    /// A file with no `fpha_plane_reduction` key parses to `plane_reduction == None`.
    #[test]
    fn test_plane_reduction_absent_is_none() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "fpha", "fpha_config": { "source": "computed" } }]
          }]
        }"#;
        let f = write_json(json);
        let file = parse_production_models(f.path()).unwrap();
        assert!(
            file.plane_reduction.is_none(),
            "absent block must resolve to None, got: {:?}",
            file.plane_reduction
        );
        assert_eq!(file.configs.len(), 1);
    }

    /// A valid `angle` block parses to `Some(Angle { tolerance_deg })`.
    #[test]
    fn test_plane_reduction_angle_valid() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "angle", "tolerance_deg": 5.0 }
        }"#;
        let f = write_json(json);
        let file = parse_production_models(f.path()).unwrap();
        assert_eq!(
            file.plane_reduction,
            Some(PlaneReductionConfig::Angle { tolerance_deg: 5.0 })
        );
    }

    /// A valid `distance` block parses to `Some(Distance { tolerance_pct, n_samples })`.
    #[test]
    fn test_plane_reduction_distance_valid() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "distance", "tolerance_pct": 0.5, "n_samples": 64 }
        }"#;
        let f = write_json(json);
        let file = parse_production_models(f.path()).unwrap();
        assert_eq!(
            file.plane_reduction,
            Some(PlaneReductionConfig::Distance {
                tolerance_pct: 0.5,
                n_samples: 64
            })
        );
    }

    /// `angle` with `tolerance_deg = 95.0` is rejected with a SchemaError naming the range.
    #[test]
    fn test_plane_reduction_angle_out_of_range() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "angle", "tolerance_deg": 95.0 }
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "fpha_plane_reduction");
                assert!(
                    message.contains("[0, 90]"),
                    "message should name the [0, 90] range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `distance` with negative `tolerance_pct` is rejected with a SchemaError.
    #[test]
    fn test_plane_reduction_distance_negative_tolerance() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "distance", "tolerance_pct": -1.0, "n_samples": 64 }
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "fpha_plane_reduction");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `distance` with `n_samples = 0` is rejected with a SchemaError.
    #[test]
    fn test_plane_reduction_distance_zero_samples() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "distance", "tolerance_pct": 0.5, "n_samples": 0 }
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "fpha_plane_reduction");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A distance field on an `angle` method is rejected by serde `deny_unknown_fields`.
    #[test]
    fn test_plane_reduction_cross_method_field_rejected() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "angle", "tolerance_pct": 5.0 }
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::SchemaError { .. } | LoadError::ParseError { .. }
            ),
            "cross-method field must be rejected, got: {err:?}"
        );
    }

    /// An `angle` method carrying BOTH its own required `tolerance_deg` AND the
    /// foreign `tolerance_pct` is rejected — isolating the `deny_unknown_fields`
    /// guarantee from the missing-required-field path (both fields are present,
    /// so the only rejection reason is the unknown `tolerance_pct`).
    #[test]
    fn test_plane_reduction_foreign_field_alongside_required_is_rejected() {
        let json = r#"{
          "production_models": [],
          "fpha_plane_reduction": { "method": "angle", "tolerance_deg": 2.0, "tolerance_pct": 5.0 }
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::SchemaError { .. } | LoadError::ParseError { .. }
            ),
            "a foreign field alongside the required one must be rejected by deny_unknown_fields, got: {err:?}"
        );
    }

    // ── reference_volume ──────────────────────────────────────────────────────

    /// `{ volume_hm3 }` on a stage range parses to the absolute-volume variant.
    #[test]
    fn reference_volume_absolute_parses() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.9,
              "reference_volume": { "volume_hm3": 1234.5 }
            }]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert_eq!(
                    ranges[0].reference_volume,
                    Some(ReferenceVolume::AbsoluteHm3(1234.5))
                );
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// `{ percentile }` on a stage range parses to the percentile variant.
    #[test]
    fn reference_volume_percentile_parses() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.9,
              "reference_volume": { "percentile": 0.5 }
            }]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert_eq!(
                    ranges[0].reference_volume,
                    Some(ReferenceVolume::Percentile(0.5))
                );
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// Both `volume_hm3` and `percentile` set (on a season entry) -> SchemaError
    /// whose message contains "mutually exclusive" and whose field names seasons.
    #[test]
    fn reference_volume_both_set_is_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [{
              "season_id": 0,
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.9,
              "reference_volume": { "volume_hm3": 1.0, "percentile": 0.5 }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    message.contains("mutually exclusive"),
                    "message should contain 'mutually exclusive', got: {message}"
                );
                assert!(
                    field.contains("seasons"),
                    "field should name seasons, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An empty `reference_volume: {}` (neither field set) -> SchemaError.
    #[test]
    fn reference_volume_neither_set_is_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.9,
              "reference_volume": {}
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("exactly one"),
                    "message should require exactly one field, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `percentile` outside `[0.0, 1.0]` -> SchemaError naming the range.
    #[test]
    fn reference_volume_percentile_out_of_range_is_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.9,
              "reference_volume": { "percentile": 1.5 }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("[0.0, 1.0]"),
                    "message should cite the [0.0, 1.0] range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `volume_hm3` of `0.0` (and negative) -> SchemaError.
    #[test]
    fn reference_volume_nonpositive_volume_is_rejected() {
        for bad in ["0.0", "-5.0"] {
            let json = format!(
                r#"{{
              "production_models": [{{
                "hydro_id": 0,
                "selection_mode": "stage_ranges",
                "stage_ranges": [{{
                  "start_stage_id": 0, "end_stage_id": null,
                  "model": "constant_productivity",
                  "productivity_mw_per_m3s": 0.9,
                  "reference_volume": {{ "volume_hm3": {bad} }}
                }}]
              }}]
            }}"#
            );
            let f = write_json(&json);
            let err = parse_production_models(f.path()).unwrap_err();
            match &err {
                LoadError::SchemaError { message, .. } => {
                    assert!(
                        message.contains("> 0.0"),
                        "message should require > 0.0, got: {message}"
                    );
                }
                other => panic!("expected SchemaError for volume_hm3={bad}, got: {other:?}"),
            }
        }
    }

    /// `{ volume_hm3 }` on a season entry parses to the absolute-volume variant.
    #[test]
    fn reference_volume_on_season_entry_parses() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 7,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [{
              "season_id": 0,
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.9,
              "reference_volume": { "volume_hm3": 800.0 }
            }]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap().configs;
        match &models[0].selection_mode {
            SelectionMode::Seasonal { seasons, .. } => {
                assert_eq!(
                    seasons[0].reference_volume,
                    Some(ReferenceVolume::AbsoluteHm3(800.0))
                );
            }
            other => panic!("expected Seasonal, got: {other:?}"),
        }
    }

    /// With the `schema` feature, the generated schema for `RawProductionModelFile`
    /// exposes a `reference_volume` property under both entry schemas.
    #[cfg(feature = "schema")]
    #[test]
    fn reference_volume_appears_in_generated_schema() {
        let schema = schemars::schema_for!(RawProductionModelFile);
        let value = serde_json::to_value(&schema).unwrap();
        let text = serde_json::to_string(&value).unwrap();
        assert!(
            text.contains("reference_volume"),
            "generated schema must expose the reference_volume property"
        );
    }

    // ── FPHA discretization-count non-negativity ──────────────────────────────

    /// Each of the two FPHA discretization counts, set negative in turn, is
    /// rejected with a SchemaError whose field names the offending count and
    /// whose message names the hydro.
    #[test]
    fn fpha_negative_discretization_count_rejected() {
        for field_name in [
            "volume_discretization_points",
            "turbine_discretization_points",
        ] {
            let json = format!(
                r#"{{
              "production_models": [{{
                "hydro_id": 4,
                "selection_mode": "stage_ranges",
                "stage_ranges": [{{
                  "start_stage_id": 0, "end_stage_id": null,
                  "model": "fpha",
                  "fpha_config": {{ "source": "computed", "{field_name}": -1 }}
                }}]
              }}]
            }}"#
            );
            let f = write_json(&json);
            let err = parse_production_models(f.path()).unwrap_err();
            match &err {
                LoadError::SchemaError { field, message, .. } => {
                    assert!(
                        field.contains(field_name),
                        "field should name {field_name}, got: {field}"
                    );
                    assert!(
                        message.contains("hydro_id 4"),
                        "message should name the hydro, got: {message}"
                    );
                }
                other => panic!("expected SchemaError for {field_name}, got: {other:?}"),
            }
        }
    }

    /// The shared discretization-count check also fires for a season entry,
    /// naming the season's hydro.
    #[test]
    fn fpha_negative_discretization_count_rejected_in_seasonal() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 9,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [{
              "season_id": 0,
              "model": "fpha",
              "fpha_config": { "source": "computed", "turbine_discretization_points": -3 }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("turbine_discretization_points"),
                    "field should name turbine_discretization_points, got: {field}"
                );
                assert!(
                    message.contains("hydro_id 9"),
                    "message should name the hydro, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `spillage_discretization_points` and `max_planes_per_hydro` are not keys of
    /// `fpha_config`; `deny_unknown_fields` refuses them.
    #[test]
    fn fpha_config_refuses_spillage_points_and_plane_ceiling() {
        for key in ["spillage_discretization_points", "max_planes_per_hydro"] {
            let json = format!(
                r#"{{
              "production_models": [{{
                "hydro_id": 4,
                "selection_mode": "stage_ranges",
                "stage_ranges": [{{
                  "start_stage_id": 0, "end_stage_id": null,
                  "model": "fpha",
                  "fpha_config": {{ "source": "computed", "{key}": 5 }}
                }}]
              }}]
            }}"#
            );
            let f = write_json(&json);
            let err = parse_production_models(f.path()).unwrap_err();
            assert!(
                err.to_string().contains(key),
                "{key} must be refused as an unknown field, got: {err}"
            );
        }
    }

    /// Absent (`None`) and non-negative FPHA discretization counts pass validation
    /// unchanged.
    #[test]
    fn fpha_absent_or_nonnegative_discretization_counts_accepted() {
        let absent = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": { "source": "computed" }
            }]
          }]
        }"#;
        let f = write_json(absent);
        assert!(
            parse_production_models(f.path()).is_ok(),
            "absent counts must pass unchanged"
        );

        let nonneg = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "volume_discretization_points": 0,
                "turbine_discretization_points": 7
              }
            }]
          }]
        }"#;
        let f = write_json(nonneg);
        assert!(
            parse_production_models(f.path()).is_ok(),
            "non-negative counts must pass unchanged"
        );
    }

    // ── FPHA fit ranges ───────────────────────────────────────────────────────

    const FIT_MAX_FIELDS: [&str; 2] = ["spillage_fit_max_m3s", "turbined_fit_max_m3s"];

    fn fit_max_json(field: &str, value: &str) -> String {
        format!(
            r#"{{
              "production_models": [{{
                "hydro_id": 7,
                "selection_mode": "stage_ranges",
                "stage_ranges": [{{
                  "start_stage_id": 0, "end_stage_id": null,
                  "model": "fpha",
                  "fpha_config": {{ "source": "computed", "{field}": {value} }}
                }}]
              }}]
            }}"#
        )
    }

    /// A positive fit-range maximum round-trips into the public layout.
    #[test]
    fn fit_max_round_trips() {
        for field in FIT_MAX_FIELDS {
            let f = write_json(&fit_max_json(field, "42282.0"));
            let file = parse_production_models(f.path()).unwrap();
            let SelectionMode::StageRanges { ranges } = &file.configs[0].selection_mode else {
                panic!("expected stage ranges");
            };
            let layout = ranges[0].fpha_config.as_ref().unwrap();
            let (spillage, turbined) = if field == "spillage_fit_max_m3s" {
                (Some(42_282.0), None)
            } else {
                (None, Some(42_282.0))
            };
            assert_eq!(layout.spillage_fit_max_m3s, spillage, "{field}");
            assert_eq!(layout.turbined_fit_max_m3s, turbined, "{field}");
        }
    }

    /// A zero or negative fit-range maximum is rejected naming the field and the
    /// hydro.
    #[test]
    fn fit_max_non_positive_rejected() {
        for field in FIT_MAX_FIELDS {
            for value in ["0.0", "-5.0"] {
                let f = write_json(&fit_max_json(field, value));
                match parse_production_models(f.path()).unwrap_err() {
                    LoadError::SchemaError {
                        field: reported,
                        message,
                        ..
                    } => {
                        assert!(
                            reported.ends_with(&format!("fpha_config.{field}")),
                            "{reported}"
                        );
                        assert!(message.contains("hydro_id 7"), "{message}");
                    }
                    other => panic!("expected SchemaError for {field} = {value}, got: {other:?}"),
                }
            }
        }
    }

    #[cfg(feature = "schema")]
    #[test]
    fn fit_max_fields_appear_in_generated_schema() {
        let schema = schemars::schema_for!(RawProductionModelFile);
        let text = serde_json::to_string(&serde_json::to_value(&schema).unwrap()).unwrap();
        for field in FIT_MAX_FIELDS {
            assert!(text.contains(field), "{field}");
        }
    }

    // ── Per-stage resolution ──────────────────────────────────────────────────

    fn layout(source: &str) -> FphaColumnLayout {
        FphaColumnLayout {
            source: source.to_string(),
            volume_discretization_points: None,
            turbine_discretization_points: None,
            fitting_window: None,
            spillage_fit_max_m3s: None,
            turbined_fit_max_m3s: None,
        }
    }

    fn stage_range(
        start_stage_id: i32,
        end_stage_id: Option<i32>,
        model: HydroGenerationModel,
        fpha_config: Option<FphaColumnLayout>,
    ) -> StageRange {
        StageRange {
            start_stage_id,
            end_stage_id,
            model,
            fpha_config,
            reference_volume: None,
            productivity_mw_per_m3s: None,
        }
    }

    fn season(
        season_id: i32,
        model: HydroGenerationModel,
        fpha_config: Option<FphaColumnLayout>,
    ) -> SeasonConfig {
        SeasonConfig {
            season_id,
            model,
            fpha_config,
            reference_volume: None,
            productivity_mw_per_m3s: None,
        }
    }

    fn stage_ranges_config(ranges: Vec<StageRange>) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges { ranges },
        }
    }

    fn seasonal_config(
        default_model: HydroGenerationModel,
        seasons: Vec<SeasonConfig>,
    ) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::Seasonal {
                default_model,
                seasons,
            },
        }
    }

    fn stage_in_season(season_id: Option<usize>) -> Stage {
        let mut stage = make_stage(0);
        stage.season_id = season_id;
        stage
    }

    #[test]
    fn resolve_stage_stage_ranges_hit() {
        let config = stage_ranges_config(vec![
            StageRange {
                reference_volume: Some(ReferenceVolume::Percentile(0.5)),
                productivity_mw_per_m3s: Some(0.42),
                ..stage_range(
                    0,
                    Some(10),
                    HydroGenerationModel::ConstantProductivity,
                    Some(layout("precomputed")),
                )
            },
            stage_range(
                11,
                None,
                HydroGenerationModel::Fpha,
                Some(layout("computed")),
            ),
        ]);
        let computed = layout("computed");
        let precomputed = layout("precomputed");

        for id in [0, 5, 10] {
            assert_eq!(
                config.resolve_stage(&make_stage(id)),
                StageProductionResolution {
                    model: Some(HydroGenerationModel::ConstantProductivity),
                    fpha_config: Some(&precomputed),
                    reference_volume: Some(&ReferenceVolume::Percentile(0.5)),
                    productivity: Some(0.42),
                },
                "stage {id}"
            );
        }
        for id in [11, 100] {
            assert_eq!(
                config.resolve_stage(&make_stage(id)),
                StageProductionResolution {
                    model: Some(HydroGenerationModel::Fpha),
                    fpha_config: Some(&computed),
                    reference_volume: None,
                    productivity: None,
                },
                "stage {id}"
            );
        }
    }

    #[test]
    fn resolve_stage_stage_ranges_gap_is_all_none() {
        let config = stage_ranges_config(vec![stage_range(
            5,
            Some(10),
            HydroGenerationModel::Fpha,
            Some(layout("computed")),
        )]);
        let gap = StageProductionResolution {
            model: None,
            fpha_config: None,
            reference_volume: None,
            productivity: None,
        };

        for id in [0, 4, 11, 20] {
            assert_eq!(config.resolve_stage(&make_stage(id)), gap, "stage {id}");
            assert!(!config.uses_fpha_at(&make_stage(id)), "stage {id}");
        }
        assert!(config.uses_fpha_at(&make_stage(7)));
    }

    #[test]
    fn resolve_stage_seasonal_hit() {
        let config = seasonal_config(
            HydroGenerationModel::Fpha,
            vec![
                SeasonConfig {
                    reference_volume: Some(ReferenceVolume::AbsoluteHm3(1200.0)),
                    productivity_mw_per_m3s: Some(0.77),
                    ..season(2, HydroGenerationModel::ConstantProductivity, None)
                },
                season(3, HydroGenerationModel::Fpha, Some(layout("computed"))),
            ],
        );
        let computed = layout("computed");

        assert_eq!(
            config.resolve_stage(&stage_in_season(Some(2))),
            StageProductionResolution {
                model: Some(HydroGenerationModel::ConstantProductivity),
                fpha_config: None,
                reference_volume: Some(&ReferenceVolume::AbsoluteHm3(1200.0)),
                productivity: Some(0.77),
            }
        );
        assert_eq!(
            config.resolve_stage(&stage_in_season(Some(3))),
            StageProductionResolution {
                model: Some(HydroGenerationModel::Fpha),
                fpha_config: Some(&computed),
                reference_volume: None,
                productivity: None,
            }
        );
    }

    #[test]
    fn resolve_stage_seasonal_miss_falls_back_to_default_model() {
        let config = seasonal_config(
            HydroGenerationModel::ConstantProductivity,
            vec![
                SeasonConfig {
                    reference_volume: Some(ReferenceVolume::Percentile(0.3)),
                    ..season(1, HydroGenerationModel::Fpha, Some(layout("precomputed")))
                },
                SeasonConfig {
                    productivity_mw_per_m3s: Some(0.6),
                    ..season(2, HydroGenerationModel::ConstantProductivity, None)
                },
            ],
        );
        let default_entry = StageProductionResolution {
            model: Some(HydroGenerationModel::ConstantProductivity),
            fpha_config: None,
            reference_volume: None,
            productivity: None,
        };

        for season_id in [Some(99), None] {
            assert_eq!(
                config.resolve_stage(&stage_in_season(season_id)),
                default_entry,
                "season {season_id:?}"
            );
        }
    }

    #[test]
    fn default_fpha_with_constant_seasons_predicates() {
        let config = seasonal_config(
            HydroGenerationModel::Fpha,
            vec![SeasonConfig {
                productivity_mw_per_m3s: Some(0.5),
                ..season(1, HydroGenerationModel::ConstantProductivity, None)
            }],
        );

        assert!(config.uses_fpha());
        assert!(config.uses_precomputed_fpha());
        assert!(!config.uses_computed_fpha());
        assert!(!config.uses_fpha_at(&stage_in_season(Some(1))));
        assert!(config.uses_fpha_at(&stage_in_season(Some(99))));
        assert!(config.uses_fpha_at(&stage_in_season(None)));
    }

    #[test]
    fn mixed_computed_and_precomputed_predicates() {
        let mixed = stage_ranges_config(vec![
            stage_range(
                0,
                Some(4),
                HydroGenerationModel::Fpha,
                Some(layout("computed")),
            ),
            stage_range(
                5,
                None,
                HydroGenerationModel::Fpha,
                Some(layout("precomputed")),
            ),
        ]);
        assert!(mixed.uses_fpha());
        assert!(mixed.uses_computed_fpha());
        assert!(mixed.uses_precomputed_fpha());

        let computed_only = stage_ranges_config(vec![stage_range(
            0,
            None,
            HydroGenerationModel::Fpha,
            Some(layout("computed")),
        )]);
        assert!(computed_only.uses_fpha());
        assert!(computed_only.uses_computed_fpha());
        assert!(!computed_only.uses_precomputed_fpha());

        let constant_only = stage_ranges_config(vec![stage_range(
            0,
            None,
            HydroGenerationModel::ConstantProductivity,
            None,
        )]);
        assert!(!constant_only.uses_fpha());
        assert!(!constant_only.uses_computed_fpha());
        assert!(!constant_only.uses_precomputed_fpha());
    }

    /// The synthetic `default_model` entry follows the declared seasons, so a
    /// `default_model == "fpha"` config with no `fpha_config` reads as
    /// precomputed FPHA even when every listed season is non-FPHA.
    #[test]
    fn selection_entries_classifies_default_fpha_as_precomputed() {
        let config = seasonal_config(
            HydroGenerationModel::Fpha,
            vec![season(1, HydroGenerationModel::ConstantProductivity, None)],
        );

        assert_eq!(
            config.selection_entries().collect::<Vec<_>>(),
            [
                SelectionEntry {
                    model: HydroGenerationModel::ConstantProductivity,
                    fpha_config: None,
                },
                SelectionEntry {
                    model: HydroGenerationModel::Fpha,
                    fpha_config: None,
                },
            ]
        );
        assert!(config.uses_precomputed_fpha());
    }

    /// Wherever a season miss resolves the FPHA `default_model`, the
    /// `fpha_hyperplanes.parquet` gate (`uses_precomputed_fpha`) must open: a
    /// classification blind to `default_model` would skip the load the
    /// resolution then reads.
    #[test]
    fn seasonal_default_fpha_classifies_as_precomputed_source() {
        let config = seasonal_config(
            HydroGenerationModel::Fpha,
            vec![
                season(1, HydroGenerationModel::ConstantProductivity, None),
                season(2, HydroGenerationModel::ConstantProductivity, None),
            ],
        );
        let miss = config.resolve_stage(&stage_in_season(Some(7)));

        assert_eq!(miss.model, Some(HydroGenerationModel::Fpha));
        assert!(miss.fpha_config.is_none());
        assert!(config.uses_precomputed_fpha());
        assert!(!config.uses_computed_fpha());
    }

    // ── Fit-domain resolution ─────────────────────────────────────────────────

    const UNSET_WINDOW: FittingWindow = FittingWindow {
        volume_min_hm3: None,
        volume_max_hm3: None,
        volume_min_percentile: None,
        volume_max_percentile: None,
    };

    fn windowed(fitting_window: FittingWindow) -> FphaColumnLayout {
        FphaColumnLayout {
            fitting_window: Some(fitting_window),
            ..layout("computed")
        }
    }

    #[test]
    fn resolve_fitting_window_without_a_window_is_the_storage_range() {
        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(None, 1_000.0, 30_000.0),
            Ok((1_000.0, 30_000.0))
        );
        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&layout("computed")), 1_000.0, 30_000.0),
            Ok((1_000.0, 30_000.0))
        );
    }

    #[test]
    fn resolve_fitting_window_takes_absolute_bounds_inside_the_storage_range() {
        let layout = windowed(FittingWindow {
            volume_min_hm3: Some(1_000.0),
            volume_max_hm3: Some(30_000.0),
            ..UNSET_WINDOW
        });

        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&layout), 0.0, 34_116.0),
            Ok((1_000.0, 30_000.0))
        );
    }

    #[test]
    fn resolve_fitting_window_fills_an_unset_bound_from_the_storage_range() {
        let only_min = windowed(FittingWindow {
            volume_min_hm3: Some(5_000.0),
            ..UNSET_WINDOW
        });
        let only_max = windowed(FittingWindow {
            volume_max_hm3: Some(20_000.0),
            ..UNSET_WINDOW
        });

        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&only_min), 1_000.0, 30_000.0),
            Ok((5_000.0, 30_000.0))
        );
        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&only_max), 1_000.0, 30_000.0),
            Ok((1_000.0, 20_000.0))
        );
    }

    /// derived: 100 + 0.1 * 1_900 = 290 and 100 + 0.9 * 1_900 = 1_810.
    #[test]
    fn resolve_fitting_window_reads_percentiles_of_the_storage_range() {
        let layout = windowed(FittingWindow {
            volume_min_percentile: Some(0.1),
            volume_max_percentile: Some(0.9),
            ..UNSET_WINDOW
        });

        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&layout), 100.0, 2_000.0),
            Ok((290.0, 1_810.0))
        );
    }

    /// derived: the maximum is 100 + 0.9 * 1_900 = 1_810.
    #[test]
    fn resolve_fitting_window_mixes_an_absolute_min_with_a_percentile_max() {
        let layout = windowed(FittingWindow {
            volume_min_hm3: Some(200.0),
            volume_max_percentile: Some(0.9),
            ..UNSET_WINDOW
        });

        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&layout), 100.0, 2_000.0),
            Ok((200.0, 1_810.0))
        );
    }

    #[test]
    fn resolve_fitting_window_clamps_bounds_to_the_storage_range() {
        let below = windowed(FittingWindow {
            volume_min_hm3: Some(-500.0),
            volume_max_hm3: Some(20_000.0),
            ..UNSET_WINDOW
        });
        let above = windowed(FittingWindow {
            volume_min_hm3: Some(5_000.0),
            volume_max_hm3: Some(50_000.0),
            ..UNSET_WINDOW
        });

        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&below), 1_000.0, 30_000.0),
            Ok((1_000.0, 20_000.0))
        );
        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&above), 1_000.0, 30_000.0),
            Ok((5_000.0, 30_000.0))
        );
    }

    #[test]
    fn resolve_fitting_window_prefers_the_absolute_bound_over_a_percentile() {
        let layout = windowed(FittingWindow {
            volume_min_hm3: Some(5_000.0),
            volume_min_percentile: Some(0.9),
            ..UNSET_WINDOW
        });

        let (v_min, _) =
            FphaColumnLayout::resolve_fitting_window(Some(&layout), 1_000.0, 30_000.0).unwrap();

        assert_eq!(v_min, 5_000.0);
    }

    #[test]
    fn resolve_fitting_window_refuses_an_inverted_window() {
        let layout = windowed(FittingWindow {
            volume_min_hm3: Some(1_500.0),
            volume_max_hm3: Some(1_000.0),
            ..UNSET_WINDOW
        });

        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(Some(&layout), 0.0, 34_116.0),
            Err(InvertedStorageRange {
                v_min_hm3: 1_500.0,
                v_max_hm3: 1_000.0,
            })
        );
    }

    #[test]
    fn resolve_fitting_window_refuses_an_inverted_or_nan_storage_range() {
        assert_eq!(
            FphaColumnLayout::resolve_fitting_window(None, 2_000.0, 1_000.0),
            Err(InvertedStorageRange {
                v_min_hm3: 2_000.0,
                v_max_hm3: 1_000.0,
            })
        );

        let err =
            FphaColumnLayout::resolve_fitting_window(Some(&layout("computed")), f64::NAN, 1_000.0)
                .unwrap_err();
        assert!(err.v_min_hm3.is_nan());
        assert_eq!(err.v_max_hm3, 1_000.0);
    }

    #[test]
    fn resolve_spillage_fit_max_prefers_the_input_then_twice_the_mean_then_twice_the_turbine_limit()
    {
        let with_input = FphaColumnLayout {
            spillage_fit_max_m3s: Some(42_000.0),
            ..layout("computed")
        };
        let without_input = layout("computed");
        let resolve = FphaColumnLayout::resolve_spillage_fit_max_m3s;

        assert_eq!(resolve(Some(&with_input), 1_500.0, 3_000.0), 42_000.0);
        assert_eq!(resolve(Some(&with_input), 0.0, 3_000.0), 42_000.0);
        assert_eq!(resolve(Some(&without_input), 1_500.0, 3_000.0), 3_000.0);
        assert_eq!(resolve(Some(&without_input), 0.0, 3_000.0), 6_000.0);
        assert_eq!(resolve(Some(&without_input), -1.0, 2_000.0), 4_000.0);
        assert_eq!(resolve(None, 1_500.0, 3_000.0), 3_000.0);
    }
}
