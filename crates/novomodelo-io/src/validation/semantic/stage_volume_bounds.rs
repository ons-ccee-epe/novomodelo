//! Layer 5a — stage-volume bound rules over the resolved hm³ overlay: `semantic.5a.59` (a
//! cross-row minimum above its maximum), `semantic.5a.60` (diversion volume capacity) and
//! `semantic.5a.61` (pumping volume capacity).

use novomodelo_core::entities::{Hydro, PumpingStation};
use novomodelo_core::temporal::{Block, Stage};
use novomodelo_core::{ResolvedBounds, effective_diversion_m3s, effective_pumping_m3s};

use super::super::{ValidationContext, rules, schema::ParsedData};
use super::envelope_tolerance;
use crate::StageIdResolver;
use crate::resolution::{BoundsEntitySlices, BoundsOverrides, resolve_bounds};

const HYDRO_BOUNDS_FILE: &str = "constraints/hydro_bounds.parquet";
const PUMPING_BOUNDS_FILE: &str = "constraints/pumping_bounds.parquet";

/// Rule 5a.59: a stage-volume minimum above its maximum once the stage rows of an entity are
/// merged. Resolution is skipped for a deck with no hm³ value.
pub(super) fn check_stage_volume_bounds(data: &ParsedData, ctx: &mut ValidationContext) {
    let declares_volume = data.hydro_bounds.iter().any(|row| {
        row.min_outflow_hm3.is_some()
            || row.max_outflow_hm3.is_some()
            || row.min_diversion_hm3.is_some()
            || row.max_diversion_hm3.is_some()
    }) || data
        .pumping_bounds
        .iter()
        .any(|row| row.min_hm3.is_some() || row.max_hm3.is_some());
    if !declares_volume {
        return;
    }

    let (bounds, study_stages) = resolve_stage_volume_inputs(data);
    check_inverted(&bounds, data, study_stages, ctx);
    check_diversion_capacity(&bounds, &data.hydros, study_stages, ctx);
    check_pumping_capacity(&bounds, &data.pumping_stations, study_stages, ctx);
}

fn resolve_stage_volume_inputs(data: &ParsedData) -> (ResolvedBounds, &[Stage]) {
    let study_stages: &[Stage] = match data.stages.stages.iter().position(|s| s.id >= 0) {
        Some(idx) => &data.stages.stages[idx..],
        None => &[],
    };
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let blocks_per_stage: Vec<usize> = study_stages.iter().map(|s| s.blocks.len()).collect();
    let bounds = resolve_bounds(
        &BoundsEntitySlices {
            hydros: &data.hydros,
            thermals: &[],
            lines: &[],
            pumping_stations: &data.pumping_stations,
            contracts: &[],
        },
        study_stages.len(),
        0,
        stage_resolver.index_map(),
        &BoundsOverrides {
            hydro: &data.hydro_bounds,
            thermal: &[],
            line: &[],
            pumping: &data.pumping_bounds,
            contract: &[],
        },
        &blocks_per_stage,
    );
    (bounds, study_stages)
}

fn check_inverted(
    bounds: &ResolvedBounds,
    data: &ParsedData,
    study_stages: &[Stage],
    ctx: &mut ValidationContext,
) {
    let overlay = bounds.stage_volume_overlay();
    let mut reject_inverted =
        |file: &str,
         entity: &str,
         stage_id: i32,
         pairs: &[(&str, Option<f64>, &str, Option<f64>)]| {
            for &(min_col, min, max_col, max) in pairs {
                if let (Some(min), Some(max)) = (min, max)
                    && min > max
                {
                    ctx.emit(
                    &rules::SEMANTIC_STAGE_VOLUME_BOUNDS_INVERTED,
                    file,
                    Some(entity),
                    format!("{entity}: stage_id={stage_id} resolves {min_col}={min} above {max_col}={max}"),
                );
                }
            }
        };

    for (h, hydro) in data.hydros.iter().enumerate() {
        let entity = format!("Hydro {}", hydro.id.0);
        for (s, stage) in study_stages.iter().enumerate() {
            let cell = overlay.hydro(h, s);
            reject_inverted(
                HYDRO_BOUNDS_FILE,
                &entity,
                stage.id,
                &[
                    (
                        "min_outflow_hm3",
                        cell.min_outflow_hm3,
                        "max_outflow_hm3",
                        cell.max_outflow_hm3,
                    ),
                    (
                        "min_diversion_hm3",
                        cell.min_diversion_hm3,
                        "max_diversion_hm3",
                        cell.max_diversion_hm3,
                    ),
                ],
            );
        }
    }

    for (p, station) in data.pumping_stations.iter().enumerate() {
        let entity = format!("Pumping station {}", station.id.0);
        for (s, stage) in study_stages.iter().enumerate() {
            let cell = overlay.pumping(p, s);
            reject_inverted(
                PUMPING_BOUNDS_FILE,
                &entity,
                stage.id,
                &[("min_hm3", cell.min_hm3, "max_hm3", cell.max_hm3)],
            );
        }
    }
}

/// Stage volume range `(cap_lo, cap_hi)` the per-block flow bounds allow, summed in block order,
/// a fixed floating-point order.
fn stage_capacity(blocks: &[Block], bounds_at: impl Fn(usize) -> (f64, f64)) -> (f64, f64) {
    let (mut cap_lo, mut cap_hi) = (0.0, 0.0);
    for (k, block) in blocks.iter().enumerate() {
        let (lo, hi) = bounds_at(k);
        cap_lo += block.tau() * lo;
        cap_hi += block.tau() * hi;
    }
    (cap_lo, cap_hi)
}

/// Rule 5a.60: a stage diversion volume bound no per-block flow can meet. Hydros without a
/// channel are left to rule 5a.46.
fn check_diversion_capacity(
    bounds: &ResolvedBounds,
    hydros: &[Hydro],
    study_stages: &[Stage],
    ctx: &mut ValidationContext,
) {
    let overlay = bounds.stage_volume_overlay();
    for (h, hydro) in hydros.iter().enumerate() {
        if hydro.diversion.is_none() {
            continue;
        }
        let entity = format!("Hydro {}", hydro.id.0);
        for (s, stage) in study_stages.iter().enumerate() {
            let cell = overlay.hydro(h, s);
            if cell.min_diversion_hm3.is_none() && cell.max_diversion_hm3.is_none() {
                continue;
            }
            let (cap_lo, cap_hi) = stage_capacity(&stage.blocks, |k| {
                effective_diversion_m3s(bounds, hydro, h, s, stage.id, k)
            });
            if let Some(v) = cell.min_diversion_hm3
                && cap_hi.is_finite()
                && v > cap_hi + envelope_tolerance(cap_hi)
            {
                ctx.emit(
                    &rules::SEMANTIC_STAGE_VOLUME_DIVERSION_CAPACITY,
                    HYDRO_BOUNDS_FILE,
                    Some(&entity),
                    format!(
                        "{entity}: stage_id={} sets min_diversion_hm3={v}, above the {cap_hi} hm³ \
                         the stage's diversion flow bounds allow",
                        stage.id
                    ),
                );
            }
            if let Some(v) = cell.max_diversion_hm3
                && v < cap_lo - envelope_tolerance(cap_lo)
            {
                ctx.emit(
                    &rules::SEMANTIC_STAGE_VOLUME_DIVERSION_CAPACITY,
                    HYDRO_BOUNDS_FILE,
                    Some(&entity),
                    format!(
                        "{entity}: stage_id={} sets max_diversion_hm3={v}, below the {cap_lo} hm³ \
                         the stage's diversion flow floors force",
                        stage.id
                    ),
                );
            }
        }
    }
}

/// Rule 5a.61: a stage pumped-volume bound no per-block flow can meet. The flow bounds come
/// from `effective_pumping_m3s`, which is `[0, 0]` outside the station's commissioning window.
fn check_pumping_capacity(
    bounds: &ResolvedBounds,
    stations: &[PumpingStation],
    study_stages: &[Stage],
    ctx: &mut ValidationContext,
) {
    let overlay = bounds.stage_volume_overlay();
    for (p, station) in stations.iter().enumerate() {
        let entity = format!("Pumping station {}", station.id.0);
        for (s, stage) in study_stages.iter().enumerate() {
            let cell = overlay.pumping(p, s);
            if cell.min_hm3.is_none() && cell.max_hm3.is_none() {
                continue;
            }
            let (cap_lo, cap_hi) = stage_capacity(&stage.blocks, |k| {
                effective_pumping_m3s(bounds, station, p, s, stage.id, k)
            });
            if let Some(v) = cell.min_hm3
                && cap_hi.is_finite()
                && v > cap_hi + envelope_tolerance(cap_hi)
            {
                ctx.emit(
                    &rules::SEMANTIC_STAGE_VOLUME_PUMPING_CAPACITY,
                    PUMPING_BOUNDS_FILE,
                    Some(&entity),
                    format!(
                        "{entity}: stage_id={} sets min_hm3={v}, above the {cap_hi} hm³ \
                         the stage's pumping flow bounds allow",
                        stage.id
                    ),
                );
            }
            if let Some(v) = cell.max_hm3
                && v < cap_lo - envelope_tolerance(cap_lo)
            {
                ctx.emit(
                    &rules::SEMANTIC_STAGE_VOLUME_PUMPING_CAPACITY,
                    PUMPING_BOUNDS_FILE,
                    Some(&entity),
                    format!(
                        "{entity}: stage_id={} sets max_hm3={v}, below the {cap_lo} hm³ \
                         the stage's pumping flow floors force",
                        stage.id
                    ),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use novomodelo_core::EntityId;
    use novomodelo_core::entities::{DiversionChannel, FillingConfig, PumpingStation};
    use novomodelo_core::temporal::Block;

    use super::super::validate_semantic_hydro_thermal;
    use super::{HYDRO_BOUNDS_FILE, PUMPING_BOUNDS_FILE};
    use crate::ValidationEntry;
    use crate::constraints::{HydroBoundsRow, PumpingBoundsRow};
    use crate::test_support::*;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    fn volume_study() -> ParsedData {
        make_data(
            vec![make_hydro(1, None), make_hydro(2, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        )
    }

    fn hydro_volume_row(hydro_id: i32, stage_id: i32) -> HydroBoundsRow {
        HydroBoundsRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            ..Default::default()
        }
    }

    fn pumping_volume_row(station_id: i32, stage_id: i32) -> PumpingBoundsRow {
        PumpingBoundsRow {
            station_id: EntityId::from(station_id),
            stage_id,
            min_m3s: None,
            max_m3s: None,
            min_hm3: None,
            max_hm3: None,
            block_id: None,
        }
    }

    fn pumping_station(id: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            operational_start_date: date(2024, 1, 1),
            bus_id: EntityId::from(1),
            source_hydro_id: EntityId::from(1),
            destination_hydro_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    fn inversion_findings(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::InvalidValue
                    && [HYDRO_BOUNDS_FILE, PUMPING_BOUNDS_FILE]
                        .iter()
                        .any(|file| e.file == Path::new(file))
            })
            .cloned()
            .collect()
    }

    const DIVERSION_CAPACITY_MESSAGE: &str = "the stage's diversion flow";
    const PUMPING_CAPACITY_MESSAGE: &str = "the stage's pumping flow";

    fn stage_blocks() -> Vec<Block> {
        [10.0, 20.0]
            .into_iter()
            .enumerate()
            .map(|(index, duration_hours)| Block {
                index,
                name: format!("B{index}"),
                duration_hours,
            })
            .collect()
    }

    fn block_volume(blocks: &[Block], flows_m3s: &[f64]) -> f64 {
        blocks
            .iter()
            .zip(flows_m3s)
            .fold(0.0, |acc, (block, flow)| acc + block.tau() * flow)
    }

    fn channel_study() -> ParsedData {
        let mut data = volume_study();
        data.hydros[0].diversion = Some(DiversionChannel {
            downstream_id: EntityId::from(2),
            max_flow_m3s: None,
        });
        data.stages.stages[0].blocks = stage_blocks();
        data
    }

    fn pumping_study() -> ParsedData {
        let mut data = volume_study();
        data.pumping_stations = vec![pumping_station(1)];
        data.stages.stages[0].blocks = stage_blocks();
        data
    }

    fn findings_containing(data: &ParsedData, fragment: &str) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .into_iter()
            .filter(|e| e.message.contains(fragment))
            .cloned()
            .collect()
    }

    fn capacity_findings(data: &ParsedData) -> Vec<ValidationEntry> {
        findings_containing(data, DIVERSION_CAPACITY_MESSAGE)
    }

    #[test]
    fn cross_row_outflow_volume_inversion_is_rejected() {
        let mut data = volume_study();
        data.hydro_bounds = vec![
            HydroBoundsRow {
                min_outflow_hm3: Some(9.0),
                ..hydro_volume_row(1, 0)
            },
            HydroBoundsRow {
                max_outflow_hm3: Some(4.0),
                ..hydro_volume_row(1, 0)
            },
        ];

        let findings = inversion_findings(&data);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert_eq!(findings[0].file, Path::new(HYDRO_BOUNDS_FILE));
        assert_eq!(findings[0].entity.as_deref(), Some("Hydro 1"));
        assert!(
            findings[0].message.contains("min_outflow_hm3=9"),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn cross_row_consistent_volume_bounds_are_accepted() {
        let mut data = volume_study();
        data.hydro_bounds = vec![
            HydroBoundsRow {
                min_outflow_hm3: Some(4.0),
                ..hydro_volume_row(1, 0)
            },
            HydroBoundsRow {
                max_outflow_hm3: Some(9.0),
                ..hydro_volume_row(1, 0)
            },
        ];

        assert!(inversion_findings(&data).is_empty());
    }

    #[test]
    fn cross_row_pumping_volume_inversion_is_rejected() {
        let mut data = volume_study();
        data.pumping_stations = vec![pumping_station(1)];
        data.pumping_bounds = vec![
            PumpingBoundsRow {
                min_hm3: Some(9.0),
                ..pumping_volume_row(1, 0)
            },
            PumpingBoundsRow {
                max_hm3: Some(4.0),
                ..pumping_volume_row(1, 0)
            },
        ];

        let findings = inversion_findings(&data);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert_eq!(findings[0].file, Path::new(PUMPING_BOUNDS_FILE));
        assert_eq!(findings[0].entity.as_deref(), Some("Pumping station 1"));
        assert!(
            findings[0].message.contains("min_hm3=9"),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn deck_without_volume_columns_skips_resolution() {
        let mut data = volume_study();
        data.pumping_stations = vec![pumping_station(1)];
        data.hydro_bounds = vec![HydroBoundsRow {
            min_outflow_m3s: Some(4.0),
            max_outflow_m3s: Some(9.0),
            ..hydro_volume_row(1, 0)
        }];
        data.pumping_bounds = vec![PumpingBoundsRow {
            min_m3s: Some(4.0),
            max_m3s: Some(9.0),
            ..pumping_volume_row(1, 0)
        }];

        assert!(inversion_findings(&data).is_empty());
    }

    #[test]
    fn min_diversion_volume_above_capacity_is_rejected() {
        let mut data = channel_study();
        data.hydro_bounds = vec![HydroBoundsRow {
            max_diversion_m3s: Some(100.0),
            min_diversion_hm3: Some(11.0),
            ..hydro_volume_row(1, 0)
        }];
        let cap = block_volume(&data.stages.stages[0].blocks, &[100.0, 100.0]);

        let findings = capacity_findings(&data);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert_eq!(findings[0].file, Path::new(HYDRO_BOUNDS_FILE));
        assert_eq!(findings[0].entity.as_deref(), Some("Hydro 1"));
        assert!(
            findings[0]
                .message
                .contains(&format!("min_diversion_hm3=11, above the {cap} hm³")),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn min_diversion_volume_at_capacity_is_accepted() {
        let mut data = channel_study();
        let cap = block_volume(&data.stages.stages[0].blocks, &[100.0, 100.0]);
        data.hydro_bounds = vec![HydroBoundsRow {
            max_diversion_m3s: Some(100.0),
            min_diversion_hm3: Some(cap),
            ..hydro_volume_row(1, 0)
        }];

        assert!(capacity_findings(&data).is_empty());
    }

    #[test]
    fn max_diversion_volume_below_forced_floor_is_rejected() {
        let mut data = channel_study();
        data.hydro_bounds = vec![HydroBoundsRow {
            min_diversion_m3s: Some(50.0),
            max_diversion_hm3: Some(5.0),
            ..hydro_volume_row(1, 0)
        }];
        let floor = block_volume(&data.stages.stages[0].blocks, &[50.0, 50.0]);

        let findings = capacity_findings(&data);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert!(
            findings[0]
                .message
                .contains(&format!("max_diversion_hm3=5, below the {floor} hm³")),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn filling_hydro_diversion_volume_is_rejected() {
        let mut data = channel_study();
        data.hydros[0].filling = Some(FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 0.0,
        });
        data.hydro_bounds = vec![HydroBoundsRow {
            max_diversion_m3s: Some(100.0),
            min_diversion_hm3: Some(0.1),
            ..hydro_volume_row(1, 0)
        }];

        let findings = capacity_findings(&data);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert!(
            findings[0]
                .message
                .contains("min_diversion_hm3=0.1, above the 0 hm³"),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn no_channel_hydro_is_skipped() {
        let mut data = volume_study();
        data.hydro_bounds = vec![HydroBoundsRow {
            min_diversion_hm3: Some(5.0),
            ..hydro_volume_row(1, 0)
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        assert!(capacity_findings(&data).is_empty());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.message.contains("declares no diversion channel")),
            "errors: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn per_block_override_enters_the_capacity() {
        let mut data = channel_study();
        data.hydro_bounds = vec![
            HydroBoundsRow {
                max_diversion_m3s: Some(100.0),
                min_diversion_hm3: Some(5.0),
                ..hydro_volume_row(1, 0)
            },
            HydroBoundsRow {
                max_diversion_m3s: Some(10.0),
                block_id: Some(1),
                ..hydro_volume_row(1, 0)
            },
        ];
        let cap = block_volume(&data.stages.stages[0].blocks, &[100.0, 10.0]);

        let findings = capacity_findings(&data);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert!(
            findings[0]
                .message
                .contains(&format!("min_diversion_hm3=5, above the {cap} hm³")),
            "message: {}",
            findings[0].message
        );

        data.hydro_bounds.pop();
        assert!(capacity_findings(&data).is_empty());
    }

    #[test]
    fn min_pumping_volume_above_capacity_is_rejected() {
        let mut data = pumping_study();
        data.pumping_bounds = vec![PumpingBoundsRow {
            max_m3s: Some(50.0),
            min_hm3: Some(6.0),
            ..pumping_volume_row(1, 0)
        }];
        let cap = block_volume(&data.stages.stages[0].blocks, &[50.0, 50.0]);

        let findings = findings_containing(&data, PUMPING_CAPACITY_MESSAGE);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert_eq!(findings[0].file, Path::new(PUMPING_BOUNDS_FILE));
        assert_eq!(findings[0].entity.as_deref(), Some("Pumping station 1"));
        assert!(
            findings[0]
                .message
                .contains(&format!("min_hm3=6, above the {cap} hm³")),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn min_pumping_volume_at_capacity_is_accepted() {
        let mut data = pumping_study();
        let cap = block_volume(&data.stages.stages[0].blocks, &[50.0, 50.0]);
        data.pumping_bounds = vec![PumpingBoundsRow {
            max_m3s: Some(50.0),
            min_hm3: Some(cap),
            ..pumping_volume_row(1, 0)
        }];

        assert!(findings_containing(&data, PUMPING_CAPACITY_MESSAGE).is_empty());
    }

    #[test]
    fn dormant_station_volume_floor_is_rejected() {
        let mut data = pumping_study();
        data.pumping_stations[0].entry_stage_id = Some(1);
        data.pumping_bounds = vec![PumpingBoundsRow {
            min_hm3: Some(0.5),
            ..pumping_volume_row(1, 0)
        }];

        let findings = findings_containing(&data, PUMPING_CAPACITY_MESSAGE);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert!(
            findings[0].message.contains("min_hm3=0.5, above the 0 hm³"),
            "message: {}",
            findings[0].message
        );
    }

    #[test]
    fn max_pumping_volume_below_forced_floor_is_rejected() {
        let mut data = pumping_study();
        data.pumping_bounds = vec![PumpingBoundsRow {
            min_m3s: Some(50.0),
            max_hm3: Some(5.0),
            ..pumping_volume_row(1, 0)
        }];
        let floor = block_volume(&data.stages.stages[0].blocks, &[50.0, 50.0]);

        let findings = findings_containing(&data, PUMPING_CAPACITY_MESSAGE);

        assert_eq!(findings.len(), 1, "findings: {findings:?}");
        assert!(
            findings[0]
                .message
                .contains(&format!("max_hm3=5, below the {floor} hm³")),
            "message: {}",
            findings[0].message
        );
    }
}
