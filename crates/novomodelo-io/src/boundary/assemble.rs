//! Assembly of a parsed `boundary_conditions.json` and the `stages.json`
//! post-study calendars into the core boundary types, with the sort keys the
//! legacy readers apply.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use novomodelo_core::{
    AnticipatedCommitmentHistory, EntityId, HydroPastDefluence, HydroStorage, InitialConditions,
    PostStudyStage, PostStudyStages, PostStudyThermalBound, RecentObservation, WaterBucketStages,
};

use crate::LoadError;
use crate::boundary::file::{BoundaryConditionsFile, ThermalBoundary};
use crate::stages::{PostStudyCalendarStage, PostStudyCalendars};

/// Builds [`InitialConditions`] and, when a post-study calendar or bound is
/// declared, [`PostStudyStages`]. `calendars` are sorted ascending by
/// `start_date`, as [`parse_stages`](crate::parse_stages) returns them.
///
/// # Errors
///
/// [`LoadError::SchemaError`] when a `post_study_bounds` row names a `stage_id`
/// absent from the thermal calendar.
// Rationale: read by the schema-layer dispatch.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn assemble_boundary(
    file: BoundaryConditionsFile,
    calendars: &PostStudyCalendars,
    filling_hydros: &HashSet<i32>,
    path: &Path,
) -> Result<(InitialConditions, Option<PostStudyStages>), LoadError> {
    let post_study = assemble_post_study(&file.thermals, calendars, path)?;
    Ok((
        assemble_initial_conditions(file, filling_hydros),
        post_study,
    ))
}

fn assemble_initial_conditions(
    file: BoundaryConditionsFile,
    filling_hydros: &HashSet<i32>,
) -> InitialConditions {
    let mut storage = Vec::new();
    let mut filling_storage = Vec::new();
    let mut recent_observations = Vec::new();
    let mut past_defluences = Vec::new();
    for hydro in file.hydros {
        let hydro_id = EntityId(hydro.hydro_id);
        if let Some(value_hm3) = hydro.storage_hm3 {
            let seed = HydroStorage {
                hydro_id,
                value_hm3,
            };
            if filling_hydros.contains(&hydro.hydro_id) {
                filling_storage.push(seed);
            } else {
                storage.push(seed);
            }
        }
        recent_observations.extend(hydro.recent_inflows.into_iter().map(|window| {
            RecentObservation {
                hydro_id,
                start_date: window.start_date,
                end_date: window.end_date,
                value_m3s: window.value,
            }
        }));
        past_defluences.extend(hydro.past_defluences.into_iter().map(|window| {
            HydroPastDefluence {
                hydro_id,
                start_date: window.start_date,
                end_date: window.end_date,
                value_m3s: window.value,
            }
        }));
    }
    let mut past_anticipated_commitments: Vec<AnticipatedCommitmentHistory> = file
        .thermals
        .into_iter()
        .flat_map(|thermal| {
            let thermal_id = EntityId(thermal.thermal_id);
            thermal
                .commitments
                .into_iter()
                .map(move |window| AnticipatedCommitmentHistory {
                    thermal_id,
                    start_date: window.start_date,
                    end_date: window.end_date,
                    value_mw: window.value,
                })
        })
        .collect();

    storage.sort_by_key(|seed| seed.hydro_id.0);
    filling_storage.sort_by_key(|seed| seed.hydro_id.0);
    recent_observations.sort_by_key(|obs: &RecentObservation| (obs.hydro_id.0, obs.start_date));
    past_defluences
        .sort_by_key(|defluence: &HydroPastDefluence| (defluence.hydro_id.0, defluence.start_date));
    past_anticipated_commitments
        .sort_by_key(|commitment| (commitment.thermal_id.0, commitment.start_date));

    InitialConditions {
        storage,
        filling_storage,
        past_anticipated_commitments,
        recent_observations,
        past_defluences,
    }
}

fn assemble_post_study(
    thermals: &[ThermalBoundary],
    calendars: &PostStudyCalendars,
    path: &Path,
) -> Result<Option<PostStudyStages>, LoadError> {
    let has_bounds = thermals
        .iter()
        .any(|thermal| !thermal.post_study_bounds.is_empty());
    if calendars.thermal.is_none() && calendars.water.is_none() && !has_bounds {
        return Ok(None);
    }

    let thermal_calendar = calendars.thermal.as_deref().unwrap_or_default();
    let stage_index: HashMap<i32, usize> = thermal_calendar
        .iter()
        .enumerate()
        .map(|(index, stage)| (stage.id, index))
        .collect();

    let mut thermal_bounds = Vec::new();
    for (k, thermal) in thermals.iter().enumerate() {
        for (i, bound) in thermal.post_study_bounds.iter().enumerate() {
            let Some(&post_study_stage_index) = stage_index.get(&bound.stage_id) else {
                let declared: Vec<i32> = thermal_calendar.iter().map(|stage| stage.id).collect();
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("thermals[{k}].post_study_bounds[{i}].stage_id"),
                    message: format!(
                        "stage_id {} of thermal_id {} is not a post_study_thermal_stages id \
                         in stages.json (declared: {declared:?})",
                        bound.stage_id, thermal.thermal_id
                    ),
                });
            };
            thermal_bounds.push(PostStudyThermalBound {
                thermal_id: EntityId(thermal.thermal_id),
                post_study_stage_index,
                cost_per_mwh: bound.cost_per_mwh,
                min_mw: bound.min_mw,
                max_mw: bound.max_mw,
            });
        }
    }
    thermal_bounds.sort_by_key(|bound| (bound.thermal_id.0, bound.post_study_stage_index));

    Ok(Some(PostStudyStages {
        thermal_stages: to_post_study_stages(thermal_calendar),
        thermal_bounds,
        water_bucket_stages: WaterBucketStages::Declared(to_post_study_stages(
            calendars.water.as_deref().unwrap_or_default(),
        )),
    }))
}

fn to_post_study_stages(calendar: &[PostStudyCalendarStage]) -> Vec<PostStudyStage> {
    calendar
        .iter()
        .map(|stage| PostStudyStage {
            start_date: stage.start_date,
            duration_hours: duration_hours(stage),
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
// Rationale: a stage spans far fewer than 2^53 days.
fn duration_hours(stage: &PostStudyCalendarStage) -> f64 {
    (stage.end_date - stage.start_date).num_days() as f64 * 24.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    use chrono::NaiveDate;
    use novomodelo_core::{
        AnticipatedCommitmentHistory, EntityId, HydroPastDefluence, HydroStorage,
        InitialConditions, PostStudyStage, PostStudyStages, RecentObservation, WaterBucketStages,
    };

    use super::assemble_boundary;
    use crate::LoadError;
    use crate::boundary::file::{
        BoundaryConditionsFile, HydroBoundary, PostStudyBound, ThermalBoundary, Window,
    };
    use crate::stages::{PostStudyCalendarStage, PostStudyCalendars};
    use crate::test_support::date;

    fn assemble(
        file: BoundaryConditionsFile,
        calendars: &PostStudyCalendars,
        filling: &[i32],
    ) -> Result<(InitialConditions, Option<PostStudyStages>), LoadError> {
        let filling_hydros: HashSet<i32> = filling.iter().copied().collect();
        assemble_boundary(
            file,
            calendars,
            &filling_hydros,
            Path::new("boundary_conditions.json"),
        )
    }

    fn file(hydros: Vec<HydroBoundary>, thermals: Vec<ThermalBoundary>) -> BoundaryConditionsFile {
        BoundaryConditionsFile { hydros, thermals }
    }

    fn hydro(hydro_id: i32, storage_hm3: Option<f64>) -> HydroBoundary {
        HydroBoundary {
            hydro_id,
            storage_hm3,
            recent_inflows: vec![],
            past_defluences: vec![],
        }
    }

    /// Bounds carry `cost_per_mwh == stage_id` so assertions can tell them apart.
    fn thermal(thermal_id: i32, stage_ids: &[i32]) -> ThermalBoundary {
        ThermalBoundary {
            thermal_id,
            commitments: vec![],
            post_study_bounds: stage_ids
                .iter()
                .map(|&stage_id| PostStudyBound {
                    stage_id,
                    cost_per_mwh: f64::from(stage_id),
                    min_mw: 0.0,
                    max_mw: 50.0,
                })
                .collect(),
        }
    }

    fn window(start_date: NaiveDate, end_date: NaiveDate, value: f64) -> Window {
        Window {
            start_date,
            end_date,
            value,
        }
    }

    fn calendar_stage(
        id: i32,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> PostStudyCalendarStage {
        PostStudyCalendarStage {
            id,
            start_date,
            end_date,
        }
    }

    /// January and February 2027, ids `first_id` and `first_id + 1`.
    fn jan_feb_calendar(first_id: i32) -> Vec<PostStudyCalendarStage> {
        vec![
            calendar_stage(first_id, date(2027, 1, 1), date(2027, 2, 1)),
            calendar_stage(first_id + 1, date(2027, 2, 1), date(2027, 3, 1)),
        ]
    }

    #[test]
    fn routes_storage_by_filling() {
        let boundary = file(
            vec![hydro(3, Some(5.0)), hydro(2, None), hydro(1, Some(10.0))],
            vec![],
        );

        let (ic, _) = assemble(boundary, &PostStudyCalendars::default(), &[3]).unwrap();

        assert_eq!(
            ic.filling_storage,
            vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 5.0
            }]
        );
        assert_eq!(
            ic.storage,
            vec![HydroStorage {
                hydro_id: EntityId(1),
                value_hm3: 10.0
            }]
        );
    }

    #[test]
    fn keeps_undeclared_hydro_records() {
        let mut undeclared = hydro(9, Some(7.0));
        undeclared.recent_inflows = vec![window(date(2025, 12, 15), date(2026, 1, 1), 12.0)];
        undeclared.past_defluences = vec![window(date(2025, 11, 1), date(2026, 1, 1), 20.0)];

        let (ic, _) = assemble(
            file(vec![undeclared], vec![]),
            &PostStudyCalendars::default(),
            &[],
        )
        .unwrap();

        assert_eq!(
            ic.storage,
            vec![HydroStorage {
                hydro_id: EntityId(9),
                value_hm3: 7.0
            }]
        );
        assert!(ic.filling_storage.is_empty());
        assert_eq!(
            ic.recent_observations,
            vec![RecentObservation {
                hydro_id: EntityId(9),
                start_date: date(2025, 12, 15),
                end_date: date(2026, 1, 1),
                value_m3s: 12.0,
            }]
        );
        assert_eq!(
            ic.past_defluences,
            vec![HydroPastDefluence {
                hydro_id: EntityId(9),
                start_date: date(2025, 11, 1),
                end_date: date(2026, 1, 1),
                value_m3s: 20.0,
            }]
        );
    }

    #[test]
    fn derives_duration_hours_from_dates() {
        let calendars = PostStudyCalendars {
            thermal: Some(jan_feb_calendar(121)),
            water: None,
        };

        let (_, post_study) = assemble(file(vec![], vec![]), &calendars, &[]).unwrap();

        let stages: Vec<(NaiveDate, u64)> = post_study
            .unwrap()
            .thermal_stages
            .iter()
            .map(|stage| (stage.start_date, stage.duration_hours.to_bits()))
            .collect();
        assert_eq!(
            stages,
            vec![
                (date(2027, 1, 1), 744.0_f64.to_bits()),
                (date(2027, 2, 1), 672.0_f64.to_bits()),
            ]
        );
    }

    #[test]
    fn absent_water_key_declares_empty_water_calendar() {
        let calendars = PostStudyCalendars {
            thermal: Some(jan_feb_calendar(121)),
            water: None,
        };

        let (_, post_study) = assemble(file(vec![], vec![]), &calendars, &[]).unwrap();

        assert_eq!(
            post_study.unwrap().water_bucket_stages,
            WaterBucketStages::Declared(vec![])
        );
    }

    #[test]
    fn maps_declared_water_calendar() {
        let calendars = PostStudyCalendars {
            thermal: None,
            water: Some(vec![calendar_stage(
                201,
                date(2027, 3, 1),
                date(2027, 4, 1),
            )]),
        };

        let (_, post_study) = assemble(file(vec![], vec![]), &calendars, &[]).unwrap();

        assert_eq!(
            post_study,
            Some(PostStudyStages {
                thermal_stages: vec![],
                thermal_bounds: vec![],
                water_bucket_stages: WaterBucketStages::Declared(vec![PostStudyStage {
                    start_date: date(2027, 3, 1),
                    duration_hours: 744.0,
                }]),
            })
        );
    }

    #[test]
    fn no_calendars_and_no_bounds_yield_none() {
        let (_, post_study) = assemble(
            file(vec![], vec![thermal(86, &[])]),
            &PostStudyCalendars::default(),
            &[],
        )
        .unwrap();

        assert_eq!(post_study, None);
    }

    #[test]
    fn rejects_unknown_post_study_stage_id() {
        let calendars = PostStudyCalendars {
            thermal: Some(jan_feb_calendar(121)),
            water: None,
        };

        let err = assemble(file(vec![], vec![thermal(86, &[99])]), &calendars, &[]).unwrap_err();

        let LoadError::SchemaError { field, message, .. } = err else {
            panic!("expected SchemaError, got {err:?}");
        };
        assert_eq!(field, "thermals[0].post_study_bounds[0].stage_id");
        assert!(message.contains("99"), "{message}");
        assert!(message.contains("[121, 122]"), "{message}");
    }

    #[test]
    fn rejects_bounds_without_thermal_calendar() {
        let err = assemble(
            file(vec![], vec![thermal(86, &[121])]),
            &PostStudyCalendars::default(),
            &[],
        )
        .unwrap_err();

        let LoadError::SchemaError { field, .. } = err else {
            panic!("expected SchemaError, got {err:?}");
        };
        assert_eq!(field, "thermals[0].post_study_bounds[0].stage_id");
    }

    #[test]
    fn sorts_vectors_by_legacy_keys() {
        let mut hydro_5 = hydro(5, Some(1.0));
        hydro_5.recent_inflows = vec![
            window(date(2026, 1, 8), date(2026, 1, 15), 2.0),
            window(date(2026, 1, 1), date(2026, 1, 8), 1.0),
        ];
        hydro_5.past_defluences = vec![
            window(date(2026, 1, 8), date(2026, 1, 15), 4.0),
            window(date(2026, 1, 1), date(2026, 1, 8), 3.0),
        ];
        let mut hydro_2 = hydro(2, Some(2.0));
        hydro_2.recent_inflows = vec![window(date(2026, 1, 20), date(2026, 1, 27), 5.0)];
        hydro_2.past_defluences = vec![window(date(2026, 1, 20), date(2026, 1, 27), 6.0)];

        let mut thermal_20 = thermal(20, &[122, 121]);
        thermal_20.commitments = vec![
            window(date(2026, 2, 1), date(2026, 3, 1), 8.0),
            window(date(2026, 1, 1), date(2026, 2, 1), 7.0),
        ];
        let mut thermal_10 = thermal(10, &[122]);
        thermal_10.commitments = vec![window(date(2026, 5, 1), date(2026, 6, 1), 9.0)];

        let calendars = PostStudyCalendars {
            thermal: Some(jan_feb_calendar(121)),
            water: None,
        };
        let (ic, post_study) = assemble(
            file(vec![hydro_5, hydro_2], vec![thermal_20, thermal_10]),
            &calendars,
            &[],
        )
        .unwrap();

        assert_eq!(
            ic.storage.iter().map(|s| s.hydro_id.0).collect::<Vec<_>>(),
            [2, 5]
        );
        assert_eq!(
            ic.recent_observations
                .iter()
                .map(|o| (o.hydro_id.0, o.start_date))
                .collect::<Vec<_>>(),
            [
                (2, date(2026, 1, 20)),
                (5, date(2026, 1, 1)),
                (5, date(2026, 1, 8)),
            ]
        );
        assert_eq!(
            ic.past_defluences
                .iter()
                .map(|d| (d.hydro_id.0, d.start_date))
                .collect::<Vec<_>>(),
            [
                (2, date(2026, 1, 20)),
                (5, date(2026, 1, 1)),
                (5, date(2026, 1, 8)),
            ]
        );
        assert_eq!(
            ic.past_anticipated_commitments,
            vec![
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(10),
                    start_date: date(2026, 5, 1),
                    end_date: date(2026, 6, 1),
                    value_mw: 9.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(20),
                    start_date: date(2026, 1, 1),
                    end_date: date(2026, 2, 1),
                    value_mw: 7.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(20),
                    start_date: date(2026, 2, 1),
                    end_date: date(2026, 3, 1),
                    value_mw: 8.0,
                },
            ]
        );
        assert_eq!(
            post_study
                .unwrap()
                .thermal_bounds
                .iter()
                .map(|b| (b.thermal_id.0, b.post_study_stage_index, b.cost_per_mwh))
                .collect::<Vec<_>>(),
            [(10, 1, 122.0), (20, 0, 121.0), (20, 1, 122.0)]
        );
    }
}
