//! Post-study boundary calendar and per-cell thermal cost/bounds.
//!
//! An in-study GNL commitment delivered after the dispatch horizon has no
//! dispatched stage to anchor its delivery-stage hours, fuel cost, or generation
//! bounds against. `post_study_stages.json` is the sole post-horizon surface:
//! [`PostStudyStages`] is the standalone boundary input carrying an ordered
//! post-horizon calendar segment plus a per-`(thermal, post-study stage)`
//! cost/bounds table those commitments resolve into by date.
//!
//! The dispatch horizon is untouched: post-study stages are never dispatched and
//! never a [`Stage`](crate::temporal::Stage) in `system.stages()` — they are a
//! boundary-only input. The setup-side resolved calendar/discount/lookup artifacts
//! are built downstream; this type is the parsed, validated input carried on
//! [`System`](crate::System).
//!
//! # Examples
//!
//! ```
//! use chrono::NaiveDate;
//! use cobre_core::{EntityId, PostStudyStage, PostStudyStages, PostStudyThermalBound};
//!
//! let ps = PostStudyStages {
//!     stages: vec![PostStudyStage {
//!         start_date: NaiveDate::from_ymd_opt(2026, 11, 1).unwrap(),
//!         duration_hours: 720.0,
//!     }],
//!     thermal_bounds: vec![PostStudyThermalBound {
//!         thermal_id: EntityId(86),
//!         post_study_stage_index: 0,
//!         cost_per_mwh: 210.0,
//!         min_mw: 0.0,
//!         max_mw: 350.0,
//!     }],
//! };
//!
//! assert_eq!(ps.stages.len(), 1);
//! assert_eq!(ps.thermal_bounds[0].thermal_id, EntityId(86));
//! ```

use chrono::NaiveDate;

use crate::EntityId;

/// One post-horizon calendar stage: a `[start_date, start_date + duration_hours)`
/// segment on which post-study deliveries are costed and bounded.
///
/// `end_date` is not declared; it is `start_date` advanced by `duration_hours`.
/// The containing [`PostStudyStages::stages`] is date-contiguous (each stage's
/// end equals the next stage's `start_date`) and its first `start_date` equals
/// the study horizon end (enforced by the `cobre-io` semantic validator, which
/// has the study calendar this crate does not).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostStudyStage {
    /// Stage start date (inclusive).
    pub start_date: NaiveDate,
    /// Stage duration in hours; must be finite and positive.
    pub duration_hours: f64,
}

/// Cost and generation bounds for one `(thermal, post-study stage)` cell.
///
/// `post_study_stage_index` indexes into [`PostStudyStages::stages`]. The
/// `[min_mw, max_mw]` interval is the plant's delivery capability at that
/// post-study stage — the sole post-horizon bound surface, directly costing
/// and bounding the delivery with no separate committed interval to reconcile
/// against. The `cobre-io` semantic validator rejects a post-study stage an
/// anticipated thermal's lead reaches with no such cell declared.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostStudyThermalBound {
    /// Thermal plant identifier. Must reference an anticipated thermal entity.
    pub thermal_id: EntityId,
    /// Index into [`PostStudyStages::stages`].
    pub post_study_stage_index: usize,
    /// Fuel cost (`$/MWh`) at this cell; must be finite.
    pub cost_per_mwh: f64,
    /// Lower bound of the delivered MW rate at this cell.
    pub min_mw: f64,
    /// Upper bound of the delivered MW rate at this cell. `min_mw <= max_mw`.
    pub max_mw: f64,
}

/// Post-study boundary input: an ordered post-horizon calendar segment plus the
/// per-`(thermal, post-study stage)` cost/bounds table.
///
/// `None` on [`System`](crate::System) when `post_study_stages.json` is absent —
/// inert and additive. Both vectors are stored in canonical order: `stages`
/// ascending by `start_date`; `thermal_bounds` by
/// `(thermal_id, post_study_stage_index)` (declaration-order invariance).
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostStudyStages {
    /// Ordered post-horizon calendar stages, ascending by `start_date`.
    pub stages: Vec<PostStudyStage>,
    /// Per-`(thermal, post-study stage)` cost/bounds, in
    /// `(thermal_id, post_study_stage_index)` order.
    pub thermal_bounds: Vec<PostStudyThermalBound>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PostStudyStages {
        PostStudyStages {
            stages: vec![
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 11, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 720.0,
                },
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 12, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 744.0,
                },
            ],
            thermal_bounds: vec![PostStudyThermalBound {
                thermal_id: EntityId(86),
                post_study_stage_index: 0,
                cost_per_mwh: 210.0,
                min_mw: 0.0,
                max_mw: 350.0,
            }],
        }
    }

    #[test]
    fn test_construction_and_clone() {
        let ps = sample();
        assert_eq!(ps.stages.len(), 2);
        assert_eq!(ps.thermal_bounds.len(), 1);
        assert_eq!(ps.thermal_bounds[0].post_study_stage_index, 0);
        assert_eq!(ps.clone(), ps);
    }

    #[test]
    fn test_default_is_empty() {
        let ps = PostStudyStages::default();
        assert!(ps.stages.is_empty());
        assert!(ps.thermal_bounds.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let ps = sample();
        let json = serde_json::to_string(&ps).unwrap();
        let back: PostStudyStages = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, back);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_postcard_roundtrip() {
        let ps = sample();
        let bytes = postcard::to_allocvec(&ps).unwrap();
        let back: PostStudyStages = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(ps, back);
        assert_eq!(bytes, postcard::to_allocvec(&back).unwrap());
    }
}
