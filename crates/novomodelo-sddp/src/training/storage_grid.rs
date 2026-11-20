//! The even grid of storage levels the first training iteration visits in place
//! of the storage its forward pass reached.

use std::ops::Range;

use novomodelo_core::Hydro;
use novomodelo_core::commissioning::hydro_operating_active;

use crate::{lp::builder::StateBox, training::trajectory::TrajectoryRecord};

/// Which hydros the grid moves at each stage but the last, whose outgoing state
/// no backward solve reads. A hydro that is filling, or not operating at the
/// stage or its successor, keeps the forward's storage: its storage follows the
/// commissioning trajectory, not the stage bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageGrid {
    n_hydros: usize,
    moved: Vec<bool>,
}

/// One rank's share of an iteration's forward passes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GridTrials {
    /// Global index of this rank's first local trial.
    pub(crate) first: usize,
    /// Trials this rank holds records for.
    pub(crate) local: usize,
    /// Forward passes across all ranks.
    pub(crate) total: usize,
}

impl StorageGrid {
    /// Resolves the moved hydros from `hydros` in canonical order over the study's stages.
    #[must_use]
    pub(crate) fn new(hydros: &[Hydro], study_stage_ids: &[i32]) -> Self {
        let operating = |hydro: &Hydro, stage_id: i32| {
            hydro_operating_active(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage_id,
            )
        };
        let moved = study_stage_ids
            .windows(2)
            .flat_map(|stages| {
                hydros.iter().map(move |hydro| {
                    hydro.filling.is_none()
                        && operating(hydro, stages[0])
                        && operating(hydro, stages[1])
                })
            })
            .collect();
        Self {
            n_hydros: hydros.len(),
            moved,
        }
    }

    /// Places global trial `k` of `N` (one-based) at `k/N` of each moved hydro's
    /// useful volume between the bounds of the stage that produces the state.
    ///
    /// Reads the producing stage's box, never the consumer's: only that box keeps
    /// the trial point inside the incoming range the next stage's recourse covers.
    pub(crate) fn overwrite_storage(
        &self,
        records: &mut [TrajectoryRecord],
        state_boxes: &[StateBox],
        storage: Range<usize>,
        n_stages: usize,
        trials: GridTrials,
    ) {
        for m in 0..trials.local {
            let trial = trials.first + m;
            for (t, state_box) in state_boxes[..n_stages.saturating_sub(1)].iter().enumerate() {
                let state = &mut records[m * n_stages + t].state;
                for (h, j) in storage.clone().enumerate() {
                    if self.moved[t * self.n_hydros + h] {
                        state[j] = grid_storage(
                            state_box.lower[j],
                            state_box.upper[j],
                            trial,
                            trials.total,
                        );
                    }
                }
            }
        }
    }
}

/// The storage of zero-based `trial` of `total`; `min` keeps rounding from lifting
/// the top trial past `upper`.
fn grid_storage(lower: f64, upper: f64, trial: usize, total: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let fraction = (trial + 1) as f64 / total as f64;
    (lower + fraction * (upper - lower)).min(upper)
}

#[cfg(test)]
mod tests {
    use novomodelo_core::FillingConfig;

    use super::{GridTrials, StorageGrid, grid_storage};
    use crate::{
        lp::builder::StateBox, test_support::minimal_hydros, training::trajectory::TrajectoryRecord,
    };

    const N_STAGES: usize = 3;
    const N_HYDROS: usize = 2;
    const N_STATE: usize = 4;
    const FORWARD_VALUE: f64 = -7.0;
    const STAGE_SCALE: [f64; N_STAGES] = [1.0, 2.0, 3.0];

    fn stage_boxes() -> Vec<StateBox> {
        STAGE_SCALE
            .iter()
            .map(|scale| StateBox {
                lower: vec![
                    10.0 * scale,
                    20.0 * scale,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                upper: vec![110.0 * scale, 60.0 * scale, f64::INFINITY, f64::INFINITY],
            })
            .collect()
    }

    fn forward_records(rows: usize) -> Vec<TrajectoryRecord> {
        vec![
            TrajectoryRecord {
                state: vec![FORWARD_VALUE; N_STATE],
                ..TrajectoryRecord::default()
            };
            rows * N_STAGES
        ]
    }

    fn every_hydro_moved() -> StorageGrid {
        StorageGrid::new(&minimal_hydros(N_HYDROS), &[0, 1, 2])
    }

    fn overwritten(rows: usize, first: usize, local: usize, total: usize) -> Vec<TrajectoryRecord> {
        let mut records = forward_records(rows);
        every_hydro_moved().overwrite_storage(
            &mut records,
            &stage_boxes(),
            0..N_HYDROS,
            N_STAGES,
            GridTrials {
                first,
                local,
                total,
            },
        );
        records
    }

    #[test]
    fn each_trial_sits_at_its_share_of_the_producing_stages_useful_volume() {
        let records = overwritten(4, 0, 4, 4);
        let expected = [
            [[35.0, 30.0], [70.0, 60.0]],
            [[60.0, 40.0], [120.0, 80.0]],
            [[85.0, 50.0], [170.0, 100.0]],
            [[110.0, 60.0], [220.0, 120.0]],
        ];
        for (m, stages) in expected.iter().enumerate() {
            for (t, storage) in stages.iter().enumerate() {
                assert_eq!(
                    &records[m * N_STAGES + t].state[..N_HYDROS],
                    storage,
                    "trial {m}, stage {t}"
                );
            }
        }
    }

    #[test]
    fn inflow_lags_and_the_last_stage_keep_the_forward_values() {
        let records = overwritten(2, 0, 2, 2);
        for m in 0..2 {
            for t in 0..N_STAGES {
                assert_eq!(
                    &records[m * N_STAGES + t].state[N_HYDROS..],
                    &[FORWARD_VALUE; N_STATE - N_HYDROS],
                    "lags of trial {m}, stage {t}"
                );
            }
            assert_eq!(
                &records[m * N_STAGES + N_STAGES - 1].state[..N_HYDROS],
                &[FORWARD_VALUE; N_HYDROS],
                "last stage of trial {m}"
            );
        }
    }

    #[test]
    fn a_rank_split_places_each_trial_where_one_rank_does() {
        let whole = overwritten(4, 0, 4, 4);
        let rank0 = overwritten(3, 0, 3, 4);
        let rank1 = overwritten(3, 3, 1, 4);
        let state = |records: &[TrajectoryRecord], m: usize, t: usize| {
            records[m * N_STAGES + t].state.clone()
        };
        for t in 0..N_STAGES {
            for m in 0..3 {
                assert_eq!(
                    state(&rank0, m, t),
                    state(&whole, m, t),
                    "trial {m}, stage {t}"
                );
            }
            assert_eq!(
                state(&rank1, 0, t),
                state(&whole, 3, t),
                "trial 3, stage {t}"
            );
            for padding in 1..3 {
                assert_eq!(
                    state(&rank1, padding, t),
                    vec![FORWARD_VALUE; N_STATE],
                    "padding row {padding}, stage {t}"
                );
            }
        }
    }

    #[test]
    fn no_trial_leaves_the_stage_bounds_and_the_top_trial_reaches_the_ceiling() {
        for (lower, upper) in [(0.1, 0.3), (0.001, 12_345.678), (17.7, 17.7), (0.0, 0.7)] {
            for total in 1..=64 {
                for trial in 0..total {
                    let storage = grid_storage(lower, upper, trial, total);
                    assert!(
                        (lower..=upper).contains(&storage),
                        "trial {trial} of {total} at {storage} outside [{lower}, {upper}]"
                    );
                }
                let top = grid_storage(lower, upper, total - 1, total);
                assert!(
                    (upper - top).abs() <= 1e-12 * upper.max(1.0),
                    "top of {total} at {top}, ceiling {upper}"
                );
            }
        }
    }

    #[test]
    fn a_filling_or_out_of_service_hydro_keeps_the_forward_storage() {
        let mut hydros = minimal_hydros(4);
        hydros[1].filling = Some(FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 0.0,
        });
        hydros[2].entry_stage_id = Some(2);
        hydros[3].exit_stage_id = Some(2);

        let grid = StorageGrid::new(&hydros, &[0, 1, 2, 3]);

        assert_eq!(
            grid.moved,
            vec![
                true, false, false, true, //
                true, false, false, false, //
                true, false, true, false,
            ]
        );
    }
}
