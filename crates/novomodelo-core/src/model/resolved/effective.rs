//! Effective per-block bounds of an entity's LP columns: what the column admits,
//! shared by the stage-LP builder and input validation.

use crate::commissioning::{commissioning_active, hydro_operating_active};
use crate::entities::{Hydro, PumpingStation};

use super::ResolvedBounds;

/// The `(min, max)` diversion bounds \[m³/s\] of `hydro` at one block: exactly what
/// the diversion column admits.
///
/// CONTRACT: a dormant hydro drops BOTH bounds to `0`, or a positive floor leaves
/// the infeasible `[min > 0, 0]`. Dormant is `hydro.filling.is_some()` or not
/// [`hydro_operating_active`]; gating on the phase alone re-opens diversion at
/// `Operating` for a filling hydro, which never diverts. The bounds are the
/// resolved ones, never `hydro.diversion.max_flow_m3s`, which silently drops a
/// wired per-stage or per-block override.
///
/// `stage_id` is the study stage id; `stage_index` and `block_index` index the
/// resolved table.
#[must_use]
pub fn effective_diversion_m3s(
    bounds: &ResolvedBounds,
    hydro: &Hydro,
    hydro_index: usize,
    stage_index: usize,
    stage_id: i32,
    block_index: usize,
) -> (f64, f64) {
    let dormant = hydro.filling.is_some()
        || !hydro_operating_active(
            hydro.filling.as_ref(),
            hydro.entry_stage_id,
            hydro.exit_stage_id,
            stage_id,
        );
    if dormant {
        return (0.0, 0.0);
    }
    let hb = bounds.hydro_bounds_at_block(hydro_index, stage_index, block_index);
    (
        hb.min_diversion_m3s.unwrap_or(0.0),
        hb.max_diversion_m3s.unwrap_or(f64::INFINITY),
    )
}

/// The `(min, max)` pumping bounds \[m³/s\] of `station` at one block: exactly what
/// the pumping-flow column admits.
///
/// CONTRACT: a station outside its commissioning window drops BOTH bounds to `0`,
/// or a positive floor leaves the infeasible `[min > 0, 0]`.
///
/// `stage_id` is the study stage id; `stage_index` and `block_index` index the
/// resolved table.
#[must_use]
pub fn effective_pumping_m3s(
    bounds: &ResolvedBounds,
    station: &PumpingStation,
    station_index: usize,
    stage_index: usize,
    stage_id: i32,
    block_index: usize,
) -> (f64, f64) {
    if !commissioning_active(station.entry_stage_id, station.exit_stage_id, stage_id) {
        return (0.0, 0.0);
    }
    let pb = bounds.pumping_bounds_at_block(station_index, stage_index, block_index);
    (pb.min_flow_m3s, pb.max_flow_m3s)
}

#[cfg(test)]
mod tests {
    use super::{effective_diversion_m3s, effective_pumping_m3s};
    use crate::entities::{FillingConfig, Hydro, PumpingStation};
    use crate::model::resolved::{
        BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, ContractBlockBounds,
        HydroBlockBounds, HydroStageBounds, LineBlockBounds, PumpingBlockBounds,
        ResolvedBlockBounds, ResolvedBounds, ThermalBlockBounds, ThermalStageBounds,
    };
    use crate::test_support::{
        HydroSpec, PumpingSpec, StageSpec, make_hydro, make_pumping_station, make_stage,
    };

    const N_STAGES: usize = 3;

    fn table(hydro_block: HydroBlockBounds) -> ResolvedBounds {
        table_of(
            hydro_block,
            PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
        )
    }

    fn pumping_table(pumping_block: PumpingBlockBounds) -> ResolvedBounds {
        table_of(HydroBlockBounds::default(), pumping_block)
    }

    fn table_of(
        hydro_block: HydroBlockBounds,
        pumping_block: PumpingBlockBounds,
    ) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 1,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block,
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block,
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    fn open_channel() -> HydroBlockBounds {
        HydroBlockBounds {
            min_diversion_m3s: Some(5.0),
            max_diversion_m3s: Some(20.0),
            ..HydroBlockBounds::default()
        }
    }

    fn unset_channel() -> HydroBlockBounds {
        HydroBlockBounds {
            min_diversion_m3s: None,
            max_diversion_m3s: None,
            ..HydroBlockBounds::default()
        }
    }

    fn diversion_at(
        bounds: &ResolvedBounds,
        hydro: &Hydro,
        stage_id: i32,
        block: usize,
    ) -> (f64, f64) {
        let stage = make_stage(StageSpec {
            id: stage_id,
            ..StageSpec::default()
        });
        effective_diversion_m3s(bounds, hydro, 0, stage.index, stage.id, block)
    }

    #[test]
    fn diversion_of_a_filling_hydro_is_closed_at_every_stage() {
        let mut hydro = make_hydro(HydroSpec::default());
        hydro.filling = Some(FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 0.0,
        });
        hydro.entry_stage_id = Some(1);
        let bounds = table(open_channel());

        assert_eq!(diversion_at(&bounds, &hydro, 0, 0), (0.0, 0.0));
        assert_eq!(diversion_at(&bounds, &hydro, 1, 0), (0.0, 0.0));
        assert_eq!(diversion_at(&bounds, &hydro, 2, 0), (0.0, 0.0));
    }

    #[test]
    fn diversion_of_a_prefilling_hydro_is_closed() {
        let mut hydro = make_hydro(HydroSpec::default());
        hydro.entry_stage_id = Some(2);
        let bounds = table(open_channel());

        assert_eq!(diversion_at(&bounds, &hydro, 1, 0), (0.0, 0.0));
        assert_eq!(diversion_at(&bounds, &hydro, 2, 0), (5.0, 20.0));
    }

    #[test]
    fn diversion_without_overrides_is_zero_to_unbounded() {
        let hydro = make_hydro(HydroSpec::default());
        let bounds = table(unset_channel());

        assert_eq!(diversion_at(&bounds, &hydro, 1, 0), (0.0, f64::INFINITY));
    }

    #[test]
    fn diversion_reads_the_block_override() {
        let hydro = make_hydro(HydroSpec::default());
        let mut bounds = table(unset_channel());
        let mut overlay = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            max_blocks: 2,
        });
        overlay
            .hydro_override_mut(0, 1, 1)
            .expect("in-range override cell")
            .max_diversion_m3s = Some(7.0);
        bounds.set_block_overlay(overlay);

        assert_eq!(diversion_at(&bounds, &hydro, 1, 0), (0.0, f64::INFINITY));
        assert_eq!(diversion_at(&bounds, &hydro, 1, 1), (0.0, 7.0));
    }

    fn open_pump() -> PumpingBlockBounds {
        PumpingBlockBounds {
            min_flow_m3s: 2.0,
            max_flow_m3s: 8.0,
        }
    }

    fn station_entering_at_stage_two() -> PumpingStation {
        let mut station = make_pumping_station(PumpingSpec::default());
        station.entry_stage_id = Some(2);
        station
    }

    fn pumping_at(bounds: &ResolvedBounds, station: &PumpingStation, stage_id: i32) -> (f64, f64) {
        let stage = make_stage(StageSpec {
            id: stage_id,
            ..StageSpec::default()
        });
        effective_pumping_m3s(bounds, station, 0, stage.index, stage.id, 0)
    }

    #[test]
    fn pumping_outside_its_window_is_closed() {
        let station = station_entering_at_stage_two();
        let bounds = pumping_table(open_pump());

        assert_eq!(pumping_at(&bounds, &station, 1), (0.0, 0.0));
    }

    #[test]
    fn pumping_inside_its_window_reads_the_resolved_bounds() {
        let station = station_entering_at_stage_two();
        let bounds = pumping_table(open_pump());

        assert_eq!(pumping_at(&bounds, &station, 2), (2.0, 8.0));
    }
}
