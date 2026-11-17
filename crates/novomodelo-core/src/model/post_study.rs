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
//! use novomodelo_core::{
//!     EntityId, PostStudyStage, PostStudyStages, PostStudyThermalBound, WaterBucketStages,
//! };
//!
//! let ps = PostStudyStages {
//!     thermal_stages: vec![PostStudyStage {
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
//!     water_bucket_stages: WaterBucketStages::FromThermalStages,
//! };
//!
//! assert_eq!(ps.thermal_stages.len(), 1);
//! assert_eq!(ps.thermal_bounds[0].thermal_id, EntityId(86));
//! ```

use chrono::NaiveDate;

use crate::EntityId;
use crate::Hydro;

/// One post-horizon calendar stage: a `[start_date, start_date + duration_hours)`
/// segment on which post-study deliveries are costed and bounded.
///
/// `end_date` is not declared; it is `start_date` advanced by `duration_hours`.
/// The containing [`PostStudyStages::thermal_stages`] is date-contiguous (each stage's
/// end equals the next stage's `start_date`) and its first `start_date` equals
/// the study horizon end (enforced by the `novomodelo-io` semantic validator, which
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
/// `post_study_stage_index` indexes into [`PostStudyStages::thermal_stages`]. The
/// `[min_mw, max_mw]` interval is the plant's delivery capability at that
/// post-study stage — the sole post-horizon bound surface, directly costing
/// and bounding the delivery with no separate committed interval to reconcile
/// against. The `novomodelo-io` semantic validator rejects a post-study stage an
/// anticipated thermal's lead reaches with no such cell declared.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostStudyThermalBound {
    /// Thermal plant identifier. Must reference an anticipated thermal entity.
    pub thermal_id: EntityId,
    /// Index into [`PostStudyStages::thermal_stages`].
    pub post_study_stage_index: usize,
    /// Fuel cost (`$/MWh`) at this cell; must be finite.
    pub cost_per_mwh: f64,
    /// Lower bound of the delivered MW rate at this cell.
    pub min_mw: f64,
    /// Upper bound of the delivered MW rate at this cell. `min_mw <= max_mw`.
    pub max_mw: f64,
}

/// Post-study boundary input: the post-horizon calendars plus the
/// per-`(thermal, post-study stage)` cost/bounds table.
///
/// Two calendars are carried: `thermal_stages` dates thermal deliveries and the
/// bounds table; `water_bucket_stages` dates water-bucket slots.
///
/// `None` on [`System`](crate::System) when `post_study_stages.json` is absent —
/// inert and additive. Both vectors are stored in canonical order:
/// `thermal_stages` ascending by `start_date`; `thermal_bounds` by
/// `(thermal_id, post_study_stage_index)` (declaration-order invariance).
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostStudyStages {
    /// Ordered post-horizon thermal calendar stages, ascending by `start_date`.
    pub thermal_stages: Vec<PostStudyStage>,
    /// Per-`(thermal, post-study stage)` cost/bounds, in
    /// `(thermal_id, post_study_stage_index)` order.
    pub thermal_bounds: Vec<PostStudyThermalBound>,
    /// Post-horizon water-bucket calendar.
    pub water_bucket_stages: WaterBucketStages,
}

/// Source of the post-horizon water-bucket calendar.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WaterBucketStages {
    /// The water calendar, used exactly: no padding, no fallback.
    Declared(Vec<PostStudyStage>),
    /// The water calendar is [`PostStudyStages::thermal_stages`] — the format a
    /// deck used before the water calendar could be declared separately.
    #[default]
    FromThermalStages,
}

/// A travel-time arc the declared water calendar is too short to carry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UncoveredArc {
    /// Hydro declaring the arc.
    pub hydro_id: EntityId,
    /// The arc's `travel_time_hours`.
    pub travel_time_hours: f64,
    /// Total hours of the water calendar, summed in slice order.
    pub declared_hours: f64,
}

/// The deepest travel-time arc `water_stages` does not cover, or `None` when
/// every arc is covered.
///
/// Only hydros passing [`Hydro::declares_travel_time_arc`] count; the deepest is
/// the one with the largest `travel_time_hours`. Coverage is the plain
/// left-to-right sum of the stages' `duration_hours` reaching that travel time.
///
/// - Equality covers.
/// - Ties go to the lowest `EntityId`.
/// - The result is independent of the order of `hydros`.
///
/// A `NaN` sum covers nothing, so the arc is reported.
#[must_use]
pub fn deepest_uncovered_travel_time_arc(
    hydros: &[Hydro],
    water_stages: &[PostStudyStage],
) -> Option<UncoveredArc> {
    let declared_hours = water_stages
        .iter()
        .fold(0.0, |sum, stage| sum + stage.duration_hours);
    let (hydro_id, travel_time_hours) = hydros
        .iter()
        .filter(|hydro| hydro.declares_travel_time_arc())
        .filter_map(|hydro| hydro.travel_time_hours.map(|hours| (hydro.id, hours)))
        .max_by(|(id_a, hours_a), (id_b, hours_b)| {
            hours_a.total_cmp(hours_b).then_with(|| id_b.cmp(id_a))
        })?;
    if declared_hours >= travel_time_hours {
        None
    } else {
        Some(UncoveredArc {
            hydro_id,
            travel_time_hours,
            declared_hours,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{HydroSpec, date, make_hydro};

    fn sample() -> PostStudyStages {
        PostStudyStages {
            thermal_stages: vec![
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
            water_bucket_stages: WaterBucketStages::FromThermalStages,
        }
    }

    #[test]
    fn test_construction_and_clone() {
        let ps = sample();
        assert_eq!(ps.thermal_stages.len(), 2);
        assert_eq!(ps.thermal_bounds.len(), 1);
        assert_eq!(ps.thermal_bounds[0].post_study_stage_index, 0);
        assert_eq!(ps.clone(), ps);
    }

    #[test]
    fn test_default_is_empty() {
        let ps = PostStudyStages::default();
        assert!(ps.thermal_stages.is_empty());
        assert!(ps.thermal_bounds.is_empty());
        assert_eq!(ps.water_bucket_stages, WaterBucketStages::FromThermalStages);
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
        let declared = PostStudyStages {
            water_bucket_stages: WaterBucketStages::Declared(vec![PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2026, 11, 1)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                duration_hours: 168.0,
            }]),
            ..sample()
        };
        for ps in [sample(), declared] {
            let bytes = postcard::to_allocvec(&ps).unwrap();
            let back: PostStudyStages = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(ps, back);
            assert_eq!(bytes, postcard::to_allocvec(&back).unwrap());
        }
    }

    fn arc_hydro(id: i32, travel_time_hours: f64) -> Hydro {
        let mut hydro = make_hydro(HydroSpec {
            id,
            downstream_id: Some(99),
            ..HydroSpec::default()
        });
        hydro.travel_time_hours = Some(travel_time_hours);
        hydro
    }

    fn water_calendar(hours: &[f64]) -> Vec<PostStudyStage> {
        hours
            .iter()
            .map(|&duration_hours| PostStudyStage {
                start_date: date(2026, 11, 1),
                duration_hours,
            })
            .collect()
    }

    #[test]
    fn deepest_uncovered_travel_time_arc_is_none_without_an_arc() {
        let no_travel_time = make_hydro(HydroSpec::default());
        assert_eq!(
            deepest_uncovered_travel_time_arc(&[no_travel_time], &water_calendar(&[168.0])),
            None
        );
        assert_eq!(
            deepest_uncovered_travel_time_arc(&[], &water_calendar(&[])),
            None
        );
    }

    #[test]
    fn deepest_uncovered_travel_time_arc_equality_covers() {
        let hydros = [arc_hydro(1, 336.0)];
        assert_eq!(
            deepest_uncovered_travel_time_arc(&hydros, &water_calendar(&[168.0, 168.0])),
            None
        );
    }

    #[test]
    fn deepest_uncovered_travel_time_arc_reports_one_hour_short() {
        let hydros = [arc_hydro(1, 336.0)];
        assert_eq!(
            deepest_uncovered_travel_time_arc(&hydros, &water_calendar(&[168.0, 167.0])),
            Some(UncoveredArc {
                hydro_id: EntityId(1),
                travel_time_hours: 336.0,
                declared_hours: 335.0,
            })
        );
    }

    #[test]
    fn deepest_uncovered_travel_time_arc_reports_the_longest_travel_time() {
        let hydros = [
            arc_hydro(2, 400.0),
            arc_hydro(5, 500.0),
            arc_hydro(1, 300.0),
        ];
        let arc = deepest_uncovered_travel_time_arc(&hydros, &water_calendar(&[168.0, 168.0]))
            .expect("both longer arcs are uncovered");
        assert_eq!(arc.hydro_id, EntityId(5));
        assert_eq!(arc.travel_time_hours, 500.0);
    }

    #[test]
    fn deepest_uncovered_travel_time_arc_ignores_hydro_order() {
        let calendar = water_calendar(&[168.0, 168.0]);
        for hydros in [
            [arc_hydro(7, 400.0), arc_hydro(3, 400.0)],
            [arc_hydro(3, 400.0), arc_hydro(7, 400.0)],
        ] {
            let arc = deepest_uncovered_travel_time_arc(&hydros, &calendar)
                .expect("400 h exceeds the 336 h calendar");
            assert_eq!(arc.hydro_id, EntityId(3));
        }
    }

    #[test]
    fn deepest_uncovered_travel_time_arc_ignores_a_hydro_without_a_downstream() {
        let mut no_downstream = arc_hydro(2, 500.0);
        no_downstream.downstream_id = None;
        let hydros = [arc_hydro(1, 336.0), no_downstream];
        assert_eq!(
            deepest_uncovered_travel_time_arc(&hydros, &water_calendar(&[168.0, 168.0])),
            None
        );
    }
}
