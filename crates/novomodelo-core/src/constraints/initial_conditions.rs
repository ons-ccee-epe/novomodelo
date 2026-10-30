//! Initial conditions for the optimization study.
//!
//! [`InitialConditions`] holds the reservoir storage levels and past inflow
//! values at the start of the study. Two storage arrays are kept separate
//! because filling hydros can have an initial volume below dead storage
//! (`min_storage_hm3`), which is not a valid operating level for regular hydros.
//!
//! See `internal-structures.md §16` and `input-constraints.md §1` for the
//! full specification including validation rules.
//!
//! # Examples
//!
//! ```
//! use chrono::NaiveDate;
//! use novomodelo_core::{AnticipatedCommitmentHistory, EntityId, HydroStorage, InitialConditions};
//!
//! let ic = InitialConditions {
//!     storage: vec![
//!         HydroStorage { hydro_id: EntityId(0), value_hm3: 15_000.0 },
//!         HydroStorage { hydro_id: EntityId(1), value_hm3:  8_500.0 },
//!     ],
//!     filling_storage: vec![
//!         HydroStorage { hydro_id: EntityId(10), value_hm3: 200.0 },
//!     ],
//!     past_anticipated_commitments: vec![
//!         AnticipatedCommitmentHistory {
//!             thermal_id: EntityId(20),
//!             start_date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
//!             end_date: NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
//!             value_mw: 100.0,
//!         },
//!     ],
//!     recent_observations: vec![],
//!     past_defluences: vec![],
//!     thermal_unit_states: vec![],
//! };
//!
//! assert_eq!(ic.storage.len(), 2);
//! assert_eq!(ic.filling_storage.len(), 1);
//! assert_eq!(ic.past_anticipated_commitments.len(), 1);
//! assert_eq!(ic.recent_observations.len(), 0);
//! assert_eq!(ic.past_defluences.len(), 0);
//! ```

use chrono::NaiveDate;

use crate::EntityId;

/// Initial storage volume for a single hydro plant.
///
/// For operating hydros, `value_hm3` must be within
/// `[min_storage_hm3, max_storage_hm3]` (validated by `novomodelo-io`).
/// For filling hydros (present in [`InitialConditions::filling_storage`]),
/// `value_hm3` must be within `[0.0, min_storage_hm3)` — strictly below the
/// dead volume (validated by `novomodelo-io`). Equality with `min_storage_hm3`
/// belongs to neither the filling range nor the operating range.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStorage {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    pub hydro_id: EntityId,
    /// Reservoir volume at the start of the study, in hm³.
    pub value_hm3: f64,
}

/// Past defluence (release) for the arc fed by a single upstream hydro over a
/// specific date range.
///
/// Used to seed the in-transit water travel-time buckets from an arc's
/// pre-study upstream releases. Each entry represents the average release rate
/// (in m³/s) over `[start_date, end_date)` — `end_date` exclusive, must be
/// after `start_date` — for one upstream hydro. Multiple entries per hydro are
/// allowed; date ranges for the same hydro must not overlap, though adjacent
/// ranges (`start_date == previous end_date`) are accepted.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroPastDefluence {
    /// Upstream hydro plant identifier whose release feeds the arc. Must
    /// reference a hydro entity in the system.
    pub hydro_id: EntityId,
    /// Start of the release window (inclusive).
    pub start_date: NaiveDate,
    /// End of the release window (exclusive). Must be after `start_date`.
    pub end_date: NaiveDate,
    /// Average release rate over the window, in m³/s. Must be finite and
    /// non-negative.
    pub value_m3s: f64,
}

/// One externally-decided anticipated-commitment window for a single thermal
/// plant.
///
/// A commitment is a `[start_date, end_date)` window (`end_date` exclusive,
/// after `start_date`) carrying `value_mw`, the MW rate the LP delivers held
/// constant over the window — mirroring [`HydroPastDefluence`]'s windowed
/// shape. A plant with `N` committed pre-study delivery stages is expressed as
/// contiguous windows tiling those stages; the LP integrates the rate against
/// each covered stage's duration through the fishing equality
/// `sum_b gen[i][b] * block_hours_b == value_mw * stage_total_hours`. The
/// decisions are sunk cost: their per-MWh cost does not enter the study
/// objective.
///
/// # Sorting invariant
///
/// Callers MUST sort the containing `Vec<AnticipatedCommitmentHistory>` by
/// `(thermal_id, start_date)` ascending before
/// `SystemBuilder::initial_conditions`, to satisfy declaration-order invariance.
///
/// # Division of responsibility
///
/// `novomodelo-core` has no view of the entity registry or the stage calendar, so
/// the `novomodelo-io` semantic validator (not `novomodelo-core`) enforces:
/// - The plant's windows tile the delivery stages it decides before the study
///   at coverage `1.0` — no gap, no overlap. Those stages fall on both sides of
///   the study horizon: the leading in-study stages and, when the plant's lead
///   reaches past the horizon, the post-horizon stages it decides before the
///   study. A window may not cover a stage the study itself decides, nor one
///   past the plant's decision reach (a hard error, no fallback; the "Pre-study
///   anticipated commitments: calendar-derived coverage" contract).
/// - Every window's `value_mw` in `[min_generation_mw, max_generation_mw]` (an
///   out-of-bounds rate makes the covered stage's fishing equality infeasible).
/// - `thermal_id` references a thermal whose `anticipated_config` is `Some`.
/// - Every anticipated thermal has at least one window and no two windows share
///   a `(thermal_id, start_date)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnticipatedCommitmentHistory {
    /// Thermal plant identifier. Must reference an anticipated thermal entity.
    pub thermal_id: EntityId,
    /// Start of the commitment window (inclusive).
    pub start_date: NaiveDate,
    /// End of the commitment window (exclusive). Must be after `start_date`.
    pub end_date: NaiveDate,
    /// Externally-decided MW rate delivered over the window, held constant.
    pub value_mw: f64,
}

/// Observed inflow for a single hydro plant over a specific date range.
///
/// Used to seed the lag accumulator when a study begins mid-season. Each entry
/// represents the average inflow (in m³/s) observed between `start_date`
/// (inclusive) and `end_date` (exclusive) for one hydro. Multiple entries per
/// hydro are allowed for rolling revisions with several observed weeks.
///
/// Date ranges for the same hydro must not overlap; adjacent ranges
/// (`start_date == previous end_date`) are accepted.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecentObservation {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    pub hydro_id: EntityId,
    /// Start of the observation period (inclusive).
    pub start_date: NaiveDate,
    /// End of the observation period (exclusive). Must be after `start_date`.
    pub end_date: NaiveDate,
    /// Average inflow observed during the period, in m³/s. Must be finite;
    /// negative values are accepted (the quantity is incremental inflow).
    pub value_m3s: f64,
}

/// Operating status of a thermal unit at the start of the study.
///
/// `StartingUp` and `ShuttingDown` are on-statuses following the unit's startup
/// or shutdown trajectory, so they count as on-time. Postcard encodes variants
/// by position: append new variants, never reorder (pinned by
/// `thermal_unit_status_postcard_discriminant_pin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ThermalUnitStatus {
    /// Offline and not following a trajectory.
    Off,
    /// Online at steady state.
    On,
    /// Online, following the startup trajectory.
    StartingUp,
    /// Online, following the shutdown trajectory.
    ShuttingDown,
}

impl ThermalUnitStatus {
    /// Whether the unit is on; `true` for every status except `Off`.
    #[must_use]
    pub fn is_on(self) -> bool {
        match self {
            Self::Off => false,
            Self::On | Self::StartingUp | Self::ShuttingDown => true,
        }
    }

    /// The trajectory the unit is following; `None` for `Off` and `On`.
    #[must_use]
    pub fn transition(self) -> Option<ThermalUnitTransition> {
        match self {
            Self::Off | Self::On => None,
            Self::StartingUp => Some(ThermalUnitTransition::StartingUp),
            Self::ShuttingDown => Some(ThermalUnitTransition::ShuttingDown),
        }
    }
}

/// The trajectory an on-status unit is following; returned by
/// [`ThermalUnitStatus::transition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalUnitTransition {
    /// Following the startup trajectory.
    StartingUp,
    /// Following the shutdown trajectory.
    ShuttingDown,
}

/// Initial state of one thermal unit at the start of the study.
///
/// - **Timing.** `status` and `previous_generation_mw` describe the half-hour just
///   before the study start; `previous_generation_mw` is the average output over
///   that half-hour.
/// - **Clock.** `time_in_state_hours` runs on the unit's on/off clock
///   ([`ThermalUnit`](crate::ThermalUnit)): from switch-on for an on-status, and from
///   the end of the shutdown trajectory for `Off`. The obligation left at the study
///   start is `max(0, min_up_time_hours - time_in_state_hours)` for an on-status and
///   `max(0, min_down_time_hours - time_in_state_hours)` for `Off`.
/// - **Position.** For `StartingUp` / `ShuttingDown`, the trajectory position is the
///   earliest index of the matching trajectory whose level below `min_generation_mw`
///   equals `previous_generation_mw`, and the trajectory resumes at the next index.
/// - **Limitation.** A trajectory interval held at exactly the unit minimum reports
///   `On`, since the transient statuses require output below the minimum.
/// - **Optional per unit.** A unit may have no record; its initial state is then
///   undefined.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalUnitInitialState {
    /// Owning plant.
    pub thermal_id: EntityId,
    /// Unit, unique within the plant.
    pub unit_id: EntityId,
    /// Operating status in the half-hour before the study start.
    pub status: ThermalUnitStatus,
    /// Hours in the current state \[h\]; a large finite value for "indefinitely".
    pub time_in_state_hours: f64,
    /// Average output in the half-hour before the study start \[MW\].
    pub previous_generation_mw: f64,
}

/// Initial system state at the start of the optimization study.
///
/// All arrays are sorted by their entity ID after loading to satisfy
/// declaration-order invariance.
///
/// A hydro must appear in exactly one of the two storage arrays, never both:
/// hydros with a `filling` configuration belong in [`filling_storage`], all
/// others (including late-entry hydros) in
/// [`storage`](InitialConditions::storage).
///
/// The `#[serde(default)]` fields below (`past_anticipated_commitments` onward)
/// are part of the postcard wire format MPI broadcast round-trips: a new field
/// must be appended after `thermal_unit_states` (the current trailing field),
/// never inserted earlier in the struct. Each is always emitted on output, even
/// when empty: `skip_serializing_if` would break the postcard round-trip.
///
/// [`filling_storage`]: InitialConditions::filling_storage
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InitialConditions {
    /// Initial storage for operating hydros, in hm³ per hydro.
    pub storage: Vec<HydroStorage>,
    /// Initial storage for filling hydros (below dead volume), in hm³ per hydro.
    ///
    /// Tied to the hydro's
    /// [`FillingConfig::start_stage_id`](crate::FillingConfig::start_stage_id):
    /// `start_stage_id == 0` means a partially-filled stage-0 level in
    /// `[0, min_storage_hm3)`; `> 0` means `0` (empty pit), frozen through the
    /// `PreFilling` phase until Filling begins.
    pub filling_storage: Vec<HydroStorage>,
    /// Past externally-decided anticipated commitments per anticipated thermal
    /// plant; empty without any. See [`AnticipatedCommitmentHistory`] for the
    /// per-entry contract and sunk-cost semantics.
    #[cfg_attr(feature = "serde", serde(default))]
    pub past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,
    /// Observed inflow data for partial periods before the study start, to seed
    /// the lag accumulator when a study begins mid-season. See
    /// [`RecentObservation`] for the per-entry contract.
    #[cfg_attr(feature = "serde", serde(default))]
    pub recent_observations: Vec<RecentObservation>,
    /// Past defluence (release) windows per arc, keyed by the upstream hydro
    /// whose release feeds the arc; empty when no arcs need seeding. See
    /// [`HydroPastDefluence`] for the per-entry contract.
    #[cfg_attr(feature = "serde", serde(default))]
    pub past_defluences: Vec<HydroPastDefluence>,
    /// Initial state per thermal unit, keyed by `(thermal_id, unit_id)`; units
    /// without a record have none. See [`ThermalUnitInitialState`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub thermal_unit_states: Vec<ThermalUnitInitialState>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "serde")]
    use crate::test_support::{ThermalUnitStateSpec, f64_bits_eq, make_thermal_unit_state};

    #[test]
    fn test_initial_conditions_construction() {
        let ic = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(0),
                    value_hm3: 15_000.0,
                },
                HydroStorage {
                    hydro_id: EntityId(1),
                    value_hm3: 8_500.0,
                },
            ],
            filling_storage: vec![HydroStorage {
                hydro_id: EntityId(10),
                value_hm3: 200.0,
            }],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
            thermal_unit_states: vec![],
        };

        assert_eq!(ic.storage.len(), 2);
        assert_eq!(ic.filling_storage.len(), 1);
        assert_eq!(ic.storage[0].hydro_id, EntityId(0));
        assert_eq!(ic.storage[0].value_hm3, 15_000.0);
        assert_eq!(ic.storage[1].hydro_id, EntityId(1));
        assert_eq!(ic.filling_storage[0].hydro_id, EntityId(10));
        assert_eq!(ic.filling_storage[0].value_hm3, 200.0);
        assert!(ic.recent_observations.is_empty());
    }

    #[test]
    fn test_initial_conditions_default_is_empty() {
        let ic = InitialConditions::default();
        assert!(ic.storage.is_empty());
        assert!(ic.filling_storage.is_empty());
        assert!(ic.recent_observations.is_empty());
    }

    #[test]
    fn test_hydro_storage_clone() {
        let hs = HydroStorage {
            hydro_id: EntityId(5),
            value_hm3: 1_234.5,
        };
        let cloned = hs.clone();
        assert_eq!(hs, cloned);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_roundtrip() {
        let ic = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(0),
                    value_hm3: 15_000.0,
                },
                HydroStorage {
                    hydro_id: EntityId(1),
                    value_hm3: 8_500.0,
                },
            ],
            filling_storage: vec![HydroStorage {
                hydro_id: EntityId(10),
                value_hm3: 200.0,
            }],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
            thermal_unit_states: vec![],
        };

        let json = serde_json::to_string(&ic).unwrap();
        let deserialized: InitialConditions = serde_json::from_str(&json).unwrap();
        assert_eq!(ic, deserialized);
    }

    #[test]
    fn test_recent_observation_construction_and_clone() {
        let obs = RecentObservation {
            hydro_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_m3s: 500.0,
        };
        let cloned = obs.clone();
        assert_eq!(obs, cloned);
    }

    #[test]
    fn test_initial_conditions_construction_with_recent_observations() {
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![RecentObservation {
                hydro_id: EntityId(0),
                start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                value_m3s: 500.0,
            }],
            past_defluences: vec![],
            thermal_unit_states: vec![],
        };
        assert_eq!(ic.recent_observations.len(), 1);
        assert_eq!(ic.recent_observations[0].hydro_id, EntityId(0));
        assert_eq!(ic.recent_observations[0].value_m3s, 500.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_roundtrip_with_recent_observations() {
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![
                RecentObservation {
                    hydro_id: EntityId(0),
                    start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_m3s: 500.0,
                },
                RecentObservation {
                    hydro_id: EntityId(0),
                    start_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 4, 11)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_m3s: 480.0,
                },
            ],
            past_defluences: vec![],
            thermal_unit_states: vec![],
        };
        let json = serde_json::to_string(&ic).unwrap();
        let deserialized: InitialConditions = serde_json::from_str(&json).unwrap();
        assert_eq!(ic, deserialized);
        assert_eq!(deserialized.recent_observations.len(), 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_default_recent_observations_absent() {
        let json = r#"{"storage":[],"filling_storage":[]}"#;
        let ic: InitialConditions = serde_json::from_str(json).unwrap();
        assert!(ic.recent_observations.is_empty());
    }

    // --- AnticipatedCommitmentHistory tests ---

    #[test]
    fn test_past_anticipated_commitments_default_empty() {
        let ic = InitialConditions::default();
        assert!(ic.past_anticipated_commitments.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_anticipated_commitment_history_serde_roundtrip() {
        let ach = AnticipatedCommitmentHistory {
            thermal_id: EntityId(7),
            start_date: NaiveDate::from_ymd_opt(2026, 1, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 2, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_mw: 100.0,
        };
        let json = serde_json::to_string(&ach).unwrap();
        let deserialized: AnticipatedCommitmentHistory = serde_json::from_str(&json).unwrap();
        assert_eq!(ach, deserialized);
        assert!(
            !json.contains("season_ids"),
            "JSON must not contain 'season_ids', got: {json}"
        );
    }

    #[test]
    fn test_initial_conditions_with_anticipated_commitments() {
        let ic = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(3),
                    start_date: NaiveDate::from_ymd_opt(2026, 1, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 2, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_mw: 50.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(5),
                    start_date: NaiveDate::from_ymd_opt(2026, 1, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 2, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_mw: 200.0,
                },
            ],
            recent_observations: vec![],
            past_defluences: vec![],
            thermal_unit_states: vec![],
        };
        assert_eq!(ic.past_anticipated_commitments.len(), 2);
        assert_eq!(ic.past_anticipated_commitments[0].thermal_id, EntityId(3));
        assert_eq!(ic.past_anticipated_commitments[0].value_mw, 50.0);
        assert_eq!(ic.past_anticipated_commitments[1].thermal_id, EntityId(5));
        assert_eq!(ic.past_anticipated_commitments[1].value_mw, 200.0);
    }

    /// Compile-time enforcement that `AnticipatedCommitmentHistory` carries the
    /// windowed field set (`thermal_id`, `start_date`, `end_date`, `value_mw`)
    /// and no `season_ids` or positional per-stage MW vector field.
    ///
    /// Rust's exhaustive struct-literal syntax will fail to compile if any
    /// undeclared field is present or any declared field is missing.
    #[test]
    fn test_anticipated_commitment_history_has_no_season_ids_field() {
        let ach = AnticipatedCommitmentHistory {
            thermal_id: EntityId(0),
            start_date: NaiveDate::from_ymd_opt(2026, 1, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 2, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_mw: 0.0,
        };
        assert_eq!(ach.thermal_id, EntityId(0));
        assert_eq!(ach.value_mw, 0.0);
    }

    // --- HydroPastDefluence / past_defluences tests ---

    #[test]
    fn test_hydro_past_defluence_construction_and_clone() {
        let hpd = HydroPastDefluence {
            hydro_id: EntityId(4),
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_m3s: 700.0,
        };
        assert_eq!(hpd.hydro_id, EntityId(4));
        assert_eq!(hpd.value_m3s, 700.0);
        assert_eq!(hpd.clone(), hpd);
    }

    #[test]
    fn test_past_defluences_default_empty() {
        let ic = InitialConditions::default();
        assert!(ic.past_defluences.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_default_past_defluences_absent() {
        let json = r#"{"storage":[],"filling_storage":[]}"#;
        let ic: InitialConditions = serde_json::from_str(json).unwrap();
        assert!(ic.past_defluences.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_past_defluences_postcard_round_trip() {
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![
                HydroPastDefluence {
                    hydro_id: EntityId(0),
                    start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_m3s: 700.0,
                },
                HydroPastDefluence {
                    hydro_id: EntityId(0),
                    start_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 4, 11)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_m3s: 650.0,
                },
            ],
            thermal_unit_states: vec![],
        };

        let bytes = postcard::to_allocvec(&ic).unwrap();
        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(ic, restored);
        assert_eq!(bytes, postcard::to_allocvec(&restored).unwrap());
        assert_eq!(restored.past_defluences.len(), 2);
    }

    #[cfg(feature = "serde")]
    fn past_defluence_window() -> HydroPastDefluence {
        HydroPastDefluence {
            hydro_id: EntityId(0),
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_m3s: 700.0,
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_thermal_unit_states_postcard_field_order_is_last() {
        let base = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![RecentObservation {
                hydro_id: EntityId(0),
                start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                value_m3s: 500.0,
            }],
            past_defluences: vec![past_defluence_window()],
            thermal_unit_states: vec![],
        };
        let mut with_state = base.clone();
        with_state.thermal_unit_states = vec![make_thermal_unit_state(ThermalUnitStateSpec {
            thermal_id: 1,
            unit_id: 1,
            status: ThermalUnitStatus::On,
            time_in_state_hours: 6.5,
            previous_generation_mw: 45.0,
        })];

        let prefix = postcard::to_allocvec(&base).unwrap();
        let full = postcard::to_allocvec(&with_state).unwrap();

        assert!(full.len() > prefix.len());
        assert_eq!(
            &full[..prefix.len() - 1],
            &prefix[..prefix.len() - 1],
            "thermal_unit_states must serialize last (append-last wire contract)"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn initial_conditions_json_without_thermal_unit_states_key_loads_with_no_records() {
        let json = r#"{
            "storage": [{"hydro_id": 0, "value_hm3": 1000.0}],
            "filling_storage": [],
            "past_anticipated_commitments": [],
            "recent_observations": [],
            "past_defluences": []
        }"#;
        let ic: InitialConditions = serde_json::from_str(json).unwrap();
        assert!(ic.thermal_unit_states.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn initial_conditions_postcard_roundtrip_preserves_thermal_unit_states() {
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            past_defluences: vec![past_defluence_window()],
            thermal_unit_states: thermal_unit_state_records().to_vec(),
            ..InitialConditions::default()
        };

        let bytes = postcard::to_allocvec(&ic).unwrap();
        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(ic, restored);
        assert_eq!(bytes, postcard::to_allocvec(&restored).unwrap());
        assert_eq!(restored.thermal_unit_states.len(), 4);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_status_postcard_discriminant_pin() {
        let pins = [
            (ThermalUnitStatus::Off, 0x00_u8),
            (ThermalUnitStatus::On, 0x01),
            (ThermalUnitStatus::StartingUp, 0x02),
            (ThermalUnitStatus::ShuttingDown, 0x03),
        ];
        for (status, discriminant) in pins {
            let bytes = postcard::to_allocvec(&status).unwrap();
            assert_eq!(
                bytes,
                vec![discriminant],
                "{status:?} must serialize to the single postcard byte {discriminant:#04x}"
            );
            let restored: ThermalUnitStatus = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(restored, status);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_status_serializes_as_bare_pascal_case_string() {
        let expected = [
            (ThermalUnitStatus::Off, r#""Off""#),
            (ThermalUnitStatus::On, r#""On""#),
            (ThermalUnitStatus::StartingUp, r#""StartingUp""#),
            (ThermalUnitStatus::ShuttingDown, r#""ShuttingDown""#),
        ];
        for (status, json) in expected {
            assert_eq!(serde_json::to_string(&status).unwrap(), json);
            let restored: ThermalUnitStatus = serde_json::from_str(json).unwrap();
            assert_eq!(restored, status);
        }
    }

    #[test]
    fn thermal_unit_status_accessors_reproduce_on_off_and_transition_view() {
        let table = [
            (ThermalUnitStatus::Off, false, None),
            (ThermalUnitStatus::On, true, None),
            (
                ThermalUnitStatus::StartingUp,
                true,
                Some(ThermalUnitTransition::StartingUp),
            ),
            (
                ThermalUnitStatus::ShuttingDown,
                true,
                Some(ThermalUnitTransition::ShuttingDown),
            ),
        ];
        for (status, is_on, transition) in table {
            assert_eq!(status.is_on(), is_on, "{status:?}.is_on()");
            assert_eq!(status.transition(), transition, "{status:?}.transition()");
        }
    }

    #[cfg(feature = "serde")]
    const THERMAL_UNIT_STATE_KEYS: [&str; 5] = [
        "thermal_id",
        "unit_id",
        "status",
        "time_in_state_hours",
        "previous_generation_mw",
    ];

    #[cfg(feature = "serde")]
    fn thermal_unit_state_records() -> [ThermalUnitInitialState; 4] {
        [
            (1, 1, ThermalUnitStatus::Off, 12.0, 0.0),
            (1, 2, ThermalUnitStatus::On, 6.5, 45.0),
            (2, 1, ThermalUnitStatus::StartingUp, 0.5, 10.0),
            (2, 2, ThermalUnitStatus::ShuttingDown, 3.0, 5.0),
        ]
        .map(
            |(thermal_id, unit_id, status, time_in_state_hours, previous_generation_mw)| {
                make_thermal_unit_state(ThermalUnitStateSpec {
                    thermal_id,
                    unit_id,
                    status,
                    time_in_state_hours,
                    previous_generation_mw,
                })
            },
        )
    }

    #[cfg(feature = "serde")]
    fn assert_state_f64_fields_bit_exact(a: &ThermalUnitInitialState, b: &ThermalUnitInitialState) {
        assert!(f64_bits_eq(a.time_in_state_hours, b.time_in_state_hours));
        assert!(f64_bits_eq(
            a.previous_generation_mw,
            b.previous_generation_mw
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_initial_state_json_roundtrip_preserves_every_field_bit_exact() {
        for record in thermal_unit_state_records() {
            let json = serde_json::to_string(&record).unwrap();
            let restored: ThermalUnitInitialState = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, record);
            assert_state_f64_fields_bit_exact(&restored, &record);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_initial_state_postcard_roundtrip_preserves_every_field_bit_exact() {
        for record in thermal_unit_state_records() {
            let bytes = postcard::to_allocvec(&record).unwrap();
            let restored: ThermalUnitInitialState = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(restored, record);
            assert_state_f64_fields_bit_exact(&restored, &record);
            assert_eq!(postcard::to_allocvec(&restored).unwrap(), bytes);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_initial_state_document_missing_any_key_fails_to_deserialize() {
        let [_, record, ..] = thermal_unit_state_records();
        let value = serde_json::to_value(record).unwrap();
        assert_eq!(
            value.as_object().unwrap().len(),
            THERMAL_UNIT_STATE_KEYS.len(),
            "a field was added or removed without updating THERMAL_UNIT_STATE_KEYS"
        );
        for key in THERMAL_UNIT_STATE_KEYS {
            let mut document = value.clone();
            document.as_object_mut().unwrap().remove(key);
            assert!(
                serde_json::from_value::<ThermalUnitInitialState>(document).is_err(),
                "document without `{key}` must fail to deserialize"
            );
        }
    }
}
