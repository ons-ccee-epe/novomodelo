//! Derivation of per-hydro PAR lag-slot and accumulator seeds from the
//! layered `inflow_history` record and `recent_observations` conditioning
//! series ([`crate::season_cast`]).

use std::collections::HashMap;

use cobre_core::{EntityId, Hydro, InflowHistoryRow, RecentObservation, SeasonMap, Stage};

#[cfg(test)]
use crate::season_cast::nth_previous_occurrence;
use crate::season_cast::{
    RealizedWindow, StageCalendar, cast, merge_layered_windows, season_period_window,
};

/// Per-hydro PAR lag-slot and accumulator seeds, indexed by canonical hydro
/// position — the iteration order [`derive_inflow_seeds`] walks `hydros` in.
#[derive(Debug)]
pub struct DerivedInflowSeeds {
    /// Lag-slot seeds, entity-major: `lag_values[pos * l_state + lag]`, lag
    /// index `0` is lag 1 (most recent).
    pub lag_values: Vec<f64>,
    /// Per-hydro accumulator seed for the first stage's in-progress occurrence.
    pub accum: Vec<f64>,
    /// Per-hydro coverage fraction seed.
    pub weight: Vec<f64>,
}

impl DerivedInflowSeeds {
    /// An all-zero seed for `n_hydros` hydros and `l_state` lag slots.
    #[must_use]
    pub fn zero(n_hydros: usize, l_state: usize) -> Self {
        Self {
            lag_values: vec![0.0_f64; n_hydros * l_state],
            accum: vec![0.0_f64; n_hydros],
            weight: vec![0.0_f64; n_hydros],
        }
    }

    /// Borrowed view over this owned seed, paired with `l_state`.
    #[must_use]
    pub fn as_seed(&self, l_state: usize) -> DerivedSeed<'_> {
        DerivedSeed {
            lag_values: &self.lag_values,
            l_state,
            accum: &self.accum,
            weight: &self.weight,
        }
    }
}

/// Borrowed, [`Copy`] view of a stage-0 derived seed. Every slice is ordered
/// by canonical hydro position (the order [`derive_inflow_seeds`] walks
/// `hydros` in), so a caller indexes with that position directly. An empty
/// `accum`/`weight` means "no seed": the accumulator resets to zero, matching
/// a period-boundary start.
#[derive(Debug, Clone, Copy)]
pub struct DerivedSeed<'a> {
    /// Lag-slot seed, entity-major: `lag_values[pos * l_state + lag]`, lag
    /// `0` is the most recent.
    pub lag_values: &'a [f64],
    /// Per-hydro stride of `lag_values`.
    pub l_state: usize,
    /// Per-hydro mid-period accumulator seed.
    pub accum: &'a [f64],
    /// Per-hydro coverage-fraction seed.
    pub weight: &'a [f64],
}

/// Derive [`DerivedInflowSeeds`] from the layered `record`/`conditioning`
/// inflow series, one hydro at a time in `hydros`' canonical order (the loop
/// index is the returned position).
///
/// `conditioning` shadows `record` day-wise via [`merge_layered_windows`]
/// before every [`crate::season_cast::cast`]. Returns
/// [`DerivedInflowSeeds::zero`] when `hydros` is empty, `first_stage.season_id`
/// is `None`, or that id is absent from `season_map.seasons`.
#[must_use]
pub fn derive_inflow_seeds(
    record: &[InflowHistoryRow],
    conditioning: &[RecentObservation],
    hydros: &[Hydro],
    first_stage: &Stage,
    season_map: &SeasonMap,
    l_state: usize,
) -> DerivedInflowSeeds {
    let n_hydros = hydros.len();
    if n_hydros == 0 {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    }

    let Some(season_id) = first_stage.season_id else {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    };

    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    };

    let calendar = StageCalendar::new(std::slice::from_ref(first_stage));
    let in_progress = season_period_window(season_map, season_def, first_stage);

    // Bucketed once, in `record`'s/`conditioning`'s own declared order: a
    // per-hydro bucket's row order must match its filtered-scan equivalent,
    // since `merge_layered_windows` resolves an overlap to the first-listed
    // covering window.
    let mut record_by_hydro: HashMap<EntityId, Vec<RealizedWindow>> = HashMap::new();
    for row in record {
        record_by_hydro
            .entry(row.hydro_id)
            .or_default()
            .push(RealizedWindow {
                start_date: row.start_date,
                end_date: row.end_date,
                value_m3s: row.value_m3s,
            });
    }
    let mut conditioning_by_hydro: HashMap<EntityId, Vec<RealizedWindow>> = HashMap::new();
    for obs in conditioning {
        conditioning_by_hydro
            .entry(obs.hydro_id)
            .or_default()
            .push(RealizedWindow {
                start_date: obs.start_date,
                end_date: obs.end_date,
                value_m3s: obs.value_m3s,
            });
    }

    let mut seeds = DerivedInflowSeeds::zero(n_hydros, l_state);
    let occurrences = calendar.season_occurrences(season_map, season_def, l_state);

    for (pos, hydro) in hydros.iter().enumerate() {
        let record_windows = record_by_hydro
            .get(&hydro.id)
            .map_or(&[][..], Vec::as_slice);
        let conditioning_windows = conditioning_by_hydro
            .get(&hydro.id)
            .map_or(&[][..], Vec::as_slice);
        let merged = merge_layered_windows(record_windows, conditioning_windows);

        let in_progress_projection = cast(&merged, &in_progress);
        seeds.accum[pos] = in_progress_projection.value * in_progress_projection.coverage;
        seeds.weight[pos] = in_progress_projection.coverage;

        if let Some(occurrences) = &occurrences {
            for (k, occurrence) in occurrences.iter().enumerate().skip(1) {
                seeds.lag_values[pos * l_state + (k - 1)] = cast(&merged, occurrence).value;
            }
        }
    }

    seeds
}

/// Reference oracle for [`derive_inflow_seeds`]: the retired per-hydro
/// full-slice filter and per-`k` [`StageCalendar::season_occurrence`]
/// restart, kept to prove the bucketed, incrementally-walked derivation is
/// bit-identical to a direct filter-then-restart derivation.
#[cfg(test)]
fn derive_inflow_seeds_reference(
    record: &[InflowHistoryRow],
    conditioning: &[RecentObservation],
    hydros: &[Hydro],
    first_stage: &Stage,
    season_map: &SeasonMap,
    l_state: usize,
) -> DerivedInflowSeeds {
    let n_hydros = hydros.len();
    if n_hydros == 0 {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    }

    let Some(season_id) = first_stage.season_id else {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    };

    let Some(season_def) = season_map.seasons.iter().find(|s| s.id == season_id) else {
        return DerivedInflowSeeds::zero(n_hydros, l_state);
    };

    let calendar = StageCalendar::new(std::slice::from_ref(first_stage));
    let in_progress = season_period_window(season_map, season_def, first_stage);

    let mut seeds = DerivedInflowSeeds::zero(n_hydros, l_state);

    for (pos, hydro) in hydros.iter().enumerate() {
        let record_windows: Vec<RealizedWindow> = record
            .iter()
            .filter(|row| row.hydro_id == hydro.id)
            .map(|row| RealizedWindow {
                start_date: row.start_date,
                end_date: row.end_date,
                value_m3s: row.value_m3s,
            })
            .collect();
        let conditioning_windows: Vec<RealizedWindow> = conditioning
            .iter()
            .filter(|obs| obs.hydro_id == hydro.id)
            .map(|obs| RealizedWindow {
                start_date: obs.start_date,
                end_date: obs.end_date,
                value_m3s: obs.value_m3s,
            })
            .collect();
        let merged = merge_layered_windows(&record_windows, &conditioning_windows);

        let in_progress_projection = cast(&merged, &in_progress);
        seeds.accum[pos] = in_progress_projection.value * in_progress_projection.coverage;
        seeds.weight[pos] = in_progress_projection.coverage;

        for k in 1..=l_state {
            let Some(projection) = calendar.season_occurrence(season_map, season_def, &merged, k)
            else {
                continue;
            };
            seeds.lag_values[pos * l_state + (k - 1)] = projection.value;
        }
    }

    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        test_support::{HydroSpec, MirrorUnitGroup, StageSpec, date, single_block},
    };

    use crate::test_support::{MonthlyLabels, monthly_season_map, weekly_season_map};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn make_stage(
        index: usize,
        start: NaiveDate,
        end: NaiveDate,
        season_id: Option<usize>,
    ) -> Stage {
        let days = u32::try_from((end - start).num_days()).unwrap();
        cobre_core::test_support::make_stage(StageSpec {
            id: i32::try_from(index).unwrap(),
            index: Some(index),
            start_date: start,
            end_date: end,
            season_id,
            blocks: single_block("SINGLE", f64::from(days) * 24.0),
            ..Default::default()
        })
    }

    fn make_hydro(id: i32) -> Hydro {
        cobre_core::test_support::make_hydro(HydroSpec {
            id,
            name: format!("H{id}"),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            operational_start_date: date(2020, 1, 1),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        })
    }

    #[test]
    #[allow(clippy::float_cmp)] // full-coverage windows make cast's weighted-value/overlap ratio bit-exact
    fn test_derive_inflow_seeds_full_coverage() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(1)];
        let record = vec![
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 100.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 2, 1),
                end_date: d(2026, 3, 1),
                value_m3s: 200.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 1, 1),
                end_date: d(2026, 2, 1),
                value_m3s: 300.0,
            },
        ];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 3);

        assert_eq!(seeds.lag_values[0], 100.0);
        assert_eq!(seeds.lag_values[1], 200.0);
        assert_eq!(seeds.lag_values[2], 300.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // whole-day-hours coverage ratio and its product with rate are bit-exact
    fn test_derive_inflow_seeds_partial_inprogress_accumulator() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let first_stage = make_stage(0, d(2026, 4, 11), d(2026, 5, 2), Some(3));
        let hydros = vec![make_hydro(1)];
        let rate = 500.0;
        let record = vec![InflowHistoryRow {
            hydro_id: EntityId(1),
            start_date: d(2026, 4, 1),
            end_date: d(2026, 4, 11),
            value_m3s: rate,
        }];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 0);

        assert!(seeds.lag_values.is_empty());
        let covered_hours = 10.0 * 24.0;
        let period_hours = 30.0 * 24.0;
        let coverage = covered_hours / period_hours;
        assert_eq!(seeds.weight[0], coverage);
        assert_eq!(seeds.accum[0], rate * coverage);
    }

    #[test]
    #[allow(clippy::float_cmp)] // derivation and the hand-computed merge_layered_windows call share the same operand order, so the ratios are bit-identical
    fn test_derive_inflow_seeds_conditioning_shadows_record() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(7)];
        let record = vec![InflowHistoryRow {
            hydro_id: EntityId(7),
            start_date: d(2026, 3, 1),
            end_date: d(2026, 4, 1),
            value_m3s: 100.0,
        }];
        let conditioning = vec![RecentObservation {
            hydro_id: EntityId(7),
            start_date: d(2026, 3, 10),
            end_date: d(2026, 3, 20),
            value_m3s: 999.0,
        }];

        let record_windows = vec![RealizedWindow {
            start_date: d(2026, 3, 1),
            end_date: d(2026, 4, 1),
            value_m3s: 100.0,
        }];
        let conditioning_windows = vec![RealizedWindow {
            start_date: d(2026, 3, 10),
            end_date: d(2026, 3, 20),
            value_m3s: 999.0,
        }];
        let hand_merged = merge_layered_windows(&record_windows, &conditioning_windows);

        let season_def = season_map.seasons.iter().find(|s| s.id == 3).unwrap();
        let in_progress = season_period_window(&season_map, season_def, &first_stage);
        let march = nth_previous_occurrence(&season_map, season_def, &in_progress, 1).unwrap();
        let expected_lag1 = cast(&hand_merged, &march).value;

        let seeds = derive_inflow_seeds(
            &record,
            &conditioning,
            &hydros,
            &first_stage,
            &season_map,
            1,
        );

        assert_eq!(seeds.lag_values[0], expected_lag1);
        assert_ne!(expected_lag1, 100.0);
    }

    #[test]
    fn test_derive_inflow_seeds_guard_conditions_return_zero() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(1)];
        let l_state = 2;

        let empty_hydros: Vec<Hydro> = Vec::new();
        let seeds =
            derive_inflow_seeds(&[], &[], &empty_hydros, &first_stage, &season_map, l_state);
        let zero = DerivedInflowSeeds::zero(0, l_state);
        assert_eq!(seeds.lag_values, zero.lag_values);
        assert_eq!(seeds.accum, zero.accum);
        assert_eq!(seeds.weight, zero.weight);

        let no_season_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), None);
        let seeds = derive_inflow_seeds(&[], &[], &hydros, &no_season_stage, &season_map, l_state);
        let zero = DerivedInflowSeeds::zero(hydros.len(), l_state);
        assert_eq!(seeds.lag_values, zero.lag_values);
        assert_eq!(seeds.accum, zero.accum);
        assert_eq!(seeds.weight, zero.weight);

        let unresolvable_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(99));
        let seeds =
            derive_inflow_seeds(&[], &[], &hydros, &unresolvable_stage, &season_map, l_state);
        let zero = DerivedInflowSeeds::zero(hydros.len(), l_state);
        assert_eq!(seeds.lag_values, zero.lag_values);
        assert_eq!(seeds.accum, zero.accum);
        assert_eq!(seeds.weight, zero.weight);
    }

    #[test]
    #[allow(clippy::float_cmp)] // full-coverage window whole-day-hours arithmetic keeps the ratio bit-exact
    fn test_derive_inflow_seeds_weekly_cycle_populates_lags() {
        let season_map = weekly_season_map();
        let first_stage = make_stage(2, d(2024, 1, 15), d(2024, 1, 22), Some(2));
        let hydros = vec![make_hydro(9)];
        let record = vec![InflowHistoryRow {
            hydro_id: EntityId(9),
            start_date: d(2024, 1, 8),
            end_date: d(2024, 1, 15),
            value_m3s: 250.0,
        }];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 1);

        assert_eq!(seeds.lag_values[0], 250.0);
    }

    /// Regression: `hydros` in canonical `(operational_start_date, id)` order
    /// can be id-DESCENDING — here hydro id=1's earlier commissioning date
    /// sorts it before hydro id=0. Each hydro's own record must still land in
    /// its OWN canonical-position slot: the derivation resolves position by
    /// iterating `hydros` directly (`hydros.iter().enumerate()`), never by
    /// `binary_search_by_key` (which requires id-ascending order and would
    /// silently misattribute a record under this staggered ordering).
    #[test]
    #[allow(clippy::float_cmp)] // full-coverage window whole-day-hours arithmetic keeps the value bit-exact
    fn test_seed_correct_under_staggered_commissioning_dates() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));

        let mut hydro_1_earlier = make_hydro(1);
        hydro_1_earlier.operational_start_date = d(2024, 1, 1);
        let mut hydro_0_later = make_hydro(0);
        hydro_0_later.operational_start_date = d(2025, 6, 1);
        let hydros = vec![hydro_1_earlier, hydro_0_later];

        let record = vec![
            InflowHistoryRow {
                hydro_id: EntityId(0),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 500.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 300.0,
            },
        ];

        let seeds = derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, 1);

        assert_eq!(
            seeds.lag_values[0], 300.0,
            "hydro id=1 (canonical position 0) lag should be its own record value"
        );
        assert_eq!(
            seeds.lag_values[1], 500.0,
            "hydro id=0 (canonical position 1) lag should be its own record value"
        );
    }

    #[test]
    fn test_bucketed_seeds_match_per_hydro_filter_reference() {
        let season_map = monthly_season_map(MonthlyLabels::OneBased);
        let first_stage = make_stage(0, d(2026, 4, 1), d(2026, 5, 1), Some(3));
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let record = vec![
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 1, 1),
                end_date: d(2026, 2, 1),
                value_m3s: 100.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(2),
                start_date: d(2026, 1, 1),
                end_date: d(2026, 2, 1),
                value_m3s: 400.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 2, 1),
                end_date: d(2026, 3, 1),
                value_m3s: 200.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(2),
                start_date: d(2026, 2, 1),
                end_date: d(2026, 3, 1),
                value_m3s: 500.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(1),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 300.0,
            },
            InflowHistoryRow {
                hydro_id: EntityId(2),
                start_date: d(2026, 3, 1),
                end_date: d(2026, 4, 1),
                value_m3s: 600.0,
            },
        ];
        let conditioning = vec![
            RecentObservation {
                hydro_id: EntityId(2),
                start_date: d(2026, 3, 10),
                end_date: d(2026, 3, 20),
                value_m3s: 999.0,
            },
            RecentObservation {
                hydro_id: EntityId(1),
                start_date: d(2026, 3, 10),
                end_date: d(2026, 3, 20),
                value_m3s: 888.0,
            },
        ];

        let actual = derive_inflow_seeds(
            &record,
            &conditioning,
            &hydros,
            &first_stage,
            &season_map,
            3,
        );
        let expected = derive_inflow_seeds_reference(
            &record,
            &conditioning,
            &hydros,
            &first_stage,
            &season_map,
            3,
        );

        assert_eq!(actual.lag_values, expected.lag_values);
        assert_eq!(actual.accum, expected.accum);
        assert_eq!(actual.weight, expected.weight);
    }

    #[test]
    fn test_as_seed_aliases_owned_contents_and_carries_l_state() {
        let owned = DerivedInflowSeeds {
            lag_values: vec![1.0, 2.0, 3.0, 4.0],
            accum: vec![5.0, 6.0],
            weight: vec![7.0, 8.0],
        };

        let seed = owned.as_seed(2);

        assert_eq!(seed.l_state, 2);
        assert_eq!(seed.lag_values, owned.lag_values.as_slice());
        assert_eq!(seed.accum, owned.accum.as_slice());
        assert_eq!(seed.weight, owned.weight.as_slice());
        assert_eq!(seed.lag_values.as_ptr(), owned.lag_values.as_ptr());
        assert_eq!(seed.accum.as_ptr(), owned.accum.as_ptr());
        assert_eq!(seed.weight.as_ptr(), owned.weight.as_ptr());
    }
}
