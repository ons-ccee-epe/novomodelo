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
//! use cobre_core::{AnticipatedCommitmentHistory, EntityId, HydroStorage, InitialConditions};
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
/// `[min_storage_hm3, max_storage_hm3]` (validated by `cobre-io`).
/// For filling hydros (present in [`InitialConditions::filling_storage`]),
/// `value_hm3` must be within `[0.0, min_storage_hm3)` — strictly below the
/// dead volume (validated by `cobre-io`). Equality with `min_storage_hm3`
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
/// `cobre-core` has no view of the entity registry or the stage calendar, so
/// the `cobre-io` semantic validator (not `cobre-core`) enforces:
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
/// must be appended after `past_defluences` (the current trailing field),
/// never inserted earlier in the struct.
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
    ///
    /// Always emitted on output even when empty — omitting it would break the
    /// postcard round-trip used by MPI broadcast.
    #[cfg_attr(feature = "serde", serde(default))]
    pub past_defluences: Vec<HydroPastDefluence>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };

        let bytes = postcard::to_allocvec(&ic).unwrap();
        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(ic, restored);
        assert_eq!(bytes, postcard::to_allocvec(&restored).unwrap());
        assert_eq!(restored.past_defluences.len(), 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_past_defluences_postcard_field_order_is_last() {
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
            past_defluences: vec![],
        };
        let mut with_defl = base.clone();
        with_defl.past_defluences = vec![HydroPastDefluence {
            hydro_id: EntityId(0),
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_m3s: 700.0,
        }];

        let prefix = postcard::to_allocvec(&base).unwrap();
        let full = postcard::to_allocvec(&with_defl).unwrap();

        assert!(full.len() > prefix.len());
        assert_eq!(
            &full[..prefix.len() - 1],
            &prefix[..prefix.len() - 1],
            "past_defluences must serialize last (append-last wire contract)"
        );
    }
}
