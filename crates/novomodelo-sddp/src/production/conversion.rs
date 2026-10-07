//! Field-for-field conversions from `Simulation*Result` to write-payload types.

use cobre_io::output::simulation_writer::{
    AnticipatedLaneWriteRecord, BusWriteRecord, ContractWriteRecord, CostWriteRecord,
    ExchangeWriteRecord, GenericViolationWriteRecord, HydroBusWriteRecord, HydroWriteRecord,
    InflowLagWriteRecord, NonControllableWriteRecord, PumpingWriteRecord, ScenarioWritePayload,
    StageWritePayload, ThermalWriteRecord, TransitBucketWriteRecord, TransitSeedWriteRecord,
};

#[cfg(test)]
use crate::setup::NodeId;
use crate::simulation::types::{
    SimulationAnticipatedLaneResult, SimulationBusResult, SimulationContractResult,
    SimulationCostResult, SimulationExchangeResult, SimulationGenericViolationResult,
    SimulationHydroBusResult, SimulationHydroResult, SimulationInflowLagResult,
    SimulationNonControllableResult, SimulationPumpingResult, SimulationScenarioResult,
    SimulationStageResult, SimulationThermalResult, SimulationTransitBucketResult,
    SimulationTransitSeedResult,
};

impl IntoWriteRecord for SimulationCostResult {
    type Record = CostWriteRecord;
    fn into_write_record(self, node_id: i32) -> CostWriteRecord {
        let s = self;
        CostWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            total_cost: s.total_cost,
            immediate_cost: s.immediate_cost,
            future_cost: s.future_cost,
            discount_factor: s.discount_factor,
            thermal_cost: s.thermal_cost,
            anticipated_thermal_cost: s.anticipated_thermal_cost,
            contract_cost: s.contract_cost,
            deficit_cost: s.deficit_cost,
            excess_cost: s.excess_cost,
            storage_violation_cost: s.storage_violation_cost,
            filling_target_cost: s.filling_target_cost,
            hydro_violation_cost: s.hydro_violation_cost,
            outflow_violation_below_cost: s.outflow_violation_below_cost,
            outflow_violation_above_cost: s.outflow_violation_above_cost,
            turbined_violation_cost: s.turbined_violation_cost,
            generation_violation_cost: s.generation_violation_cost,
            evaporation_violation_cost: s.evaporation_violation_cost,
            withdrawal_violation_cost: s.withdrawal_violation_cost,
            inflow_penalty_cost: s.inflow_penalty_cost,
            generic_violation_cost: s.generic_violation_cost,
            spillage_cost: s.spillage_cost,
            turbined_cost: s.turbined_cost,
            curtailment_cost: s.curtailment_cost,
            exchange_cost: s.exchange_cost,
            pumping_cost: s.pumping_cost,
        }
    }
}

impl IntoWriteRecord for SimulationHydroResult {
    type Record = HydroWriteRecord;
    fn into_write_record(self, node_id: i32) -> HydroWriteRecord {
        let s = self;
        HydroWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            hydro_id: s.hydro_id,
            turbined_m3s: s.turbined_m3s,
            spillage_m3s: s.spillage_m3s,
            evaporation_m3s: s.evaporation_m3s,
            diverted_inflow_m3s: s.diverted_inflow_m3s,
            diverted_outflow_m3s: s.diverted_outflow_m3s,
            incremental_inflow_m3s: s.incremental_inflow_m3s,
            inflow_m3s: s.inflow_m3s,
            storage_initial_hm3: s.storage_initial_hm3,
            storage_final_hm3: s.storage_final_hm3,
            generation_mw: s.generation_mw,
            equivalent_productivity_mw_per_m3s: s.equivalent_productivity_mw_per_m3s,
            accumulated_productivity_mw_per_m3s: s.accumulated_productivity_mw_per_m3s,
            incremental_inflow_energy_mw: s.incremental_inflow_energy_mw,
            stored_energy_initial_mwh: s.stored_energy_initial_mwh,
            stored_energy_final_mwh: s.stored_energy_final_mwh,
            spillage_cost: s.spillage_cost,
            water_value_per_hm3: s.water_value_per_hm3,
            storage_binding_code: s.storage_binding_code,
            operative_state_code: s.operative_state_code,
            turbined_slack_m3s: s.turbined_slack_m3s,
            outflow_slack_below_m3s: s.outflow_slack_below_m3s,
            outflow_slack_above_m3s: s.outflow_slack_above_m3s,
            generation_slack_mw: s.generation_slack_mw,
            storage_violation_below_hm3: s.storage_violation_below_hm3,
            filling_target_violation_hm3: s.filling_target_violation_hm3,
            evaporation_violation_pos_m3s: s.evaporation_violation_pos_m3s,
            evaporation_violation_neg_m3s: s.evaporation_violation_neg_m3s,
            inflow_nonnegativity_slack_m3s: s.inflow_nonnegativity_slack_m3s,
            water_withdrawal_violation_pos_m3s: s.water_withdrawal_violation_pos_m3s,
            water_withdrawal_violation_neg_m3s: s.water_withdrawal_violation_neg_m3s,
            integrated_equivalent_productivity_mw_per_m3s: s
                .integrated_equivalent_productivity_mw_per_m3s,
            integrated_accumulated_productivity_mw_per_m3s: s
                .integrated_accumulated_productivity_mw_per_m3s,
            stored_energy_initial_mw: s.stored_energy_initial_mw,
            stored_energy_final_mw: s.stored_energy_final_mw,
        }
    }
}

impl IntoWriteRecord for SimulationHydroBusResult {
    type Record = HydroBusWriteRecord;
    fn into_write_record(self, node_id: i32) -> HydroBusWriteRecord {
        let s = self;
        HydroBusWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            hydro_id: s.hydro_id,
            bus_id: s.bus_id,
            turbined_m3s: s.turbined_m3s,
            generation_mw: s.generation_mw,
        }
    }
}

impl IntoWriteRecord for SimulationThermalResult {
    type Record = ThermalWriteRecord;
    fn into_write_record(self, node_id: i32) -> ThermalWriteRecord {
        let s = self;
        ThermalWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            thermal_id: s.thermal_id,
            generation_mw: s.generation_mw,
            generation_cost: s.generation_cost,
            is_anticipated: s.is_anticipated,
            anticipated_committed_mw: s.anticipated_committed_mw,
            anticipated_decision_mw: s.anticipated_decision_mw,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl IntoWriteRecord for SimulationExchangeResult {
    type Record = ExchangeWriteRecord;
    fn into_write_record(self, node_id: i32) -> ExchangeWriteRecord {
        let s = self;
        ExchangeWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            line_id: s.line_id,
            direct_flow_mw: s.direct_flow_mw,
            reverse_flow_mw: s.reverse_flow_mw,
            exchange_cost: s.exchange_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl IntoWriteRecord for SimulationBusResult {
    type Record = BusWriteRecord;
    fn into_write_record(self, node_id: i32) -> BusWriteRecord {
        let s = self;
        BusWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            bus_id: s.bus_id,
            load_mw: s.load_mw,
            deficit_mw: s.deficit_mw,
            excess_mw: s.excess_mw,
            spot_price: s.spot_price,
        }
    }
}

impl IntoWriteRecord for SimulationPumpingResult {
    type Record = PumpingWriteRecord;
    fn into_write_record(self, node_id: i32) -> PumpingWriteRecord {
        let s = self;
        PumpingWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            pumping_station_id: s.pumping_station_id,
            pumped_flow_m3s: s.pumped_flow_m3s,
            power_consumption_mw: s.power_consumption_mw,
            pumping_cost: s.pumping_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl IntoWriteRecord for SimulationContractResult {
    type Record = ContractWriteRecord;
    fn into_write_record(self, node_id: i32) -> ContractWriteRecord {
        let s = self;
        ContractWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            contract_id: s.contract_id,
            power_mw: s.power_mw,
            price_per_mwh: s.price_per_mwh,
            total_cost: s.total_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl IntoWriteRecord for SimulationNonControllableResult {
    type Record = NonControllableWriteRecord;
    fn into_write_record(self, node_id: i32) -> NonControllableWriteRecord {
        let s = self;
        NonControllableWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            non_controllable_id: s.non_controllable_id,
            generation_mw: s.generation_mw,
            available_mw: s.available_mw,
            curtailment_mw: s.curtailment_mw,
            curtailment_cost: s.curtailment_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl IntoWriteRecord for SimulationInflowLagResult {
    type Record = InflowLagWriteRecord;
    fn into_write_record(self, node_id: i32) -> InflowLagWriteRecord {
        let s = self;
        InflowLagWriteRecord {
            stage_id: s.stage_id,
            node_id,
            hydro_id: s.hydro_id,
            lag_index: s.lag_index,
            inflow_m3s: s.inflow_m3s,
        }
    }
}

impl IntoWriteRecord for SimulationTransitBucketResult {
    type Record = TransitBucketWriteRecord;
    fn into_write_record(self, node_id: i32) -> TransitBucketWriteRecord {
        let s = self;
        TransitBucketWriteRecord {
            stage_id: s.stage_id,
            node_id,
            hydro_id: s.hydro_id,
            lag: s.lag,
            in_transit_volume_hm3: s.in_transit_volume_hm3,
            delayed_arrival_hm3: s.delayed_arrival_hm3,
        }
    }
}

impl IntoWriteRecord for SimulationAnticipatedLaneResult {
    type Record = AnticipatedLaneWriteRecord;
    fn into_write_record(self, node_id: i32) -> AnticipatedLaneWriteRecord {
        let s = self;
        AnticipatedLaneWriteRecord {
            stage_id: s.stage_id,
            node_id,
            thermal_id: s.thermal_id,
            delivery_date: s.delivery_date,
            deposited_decision_mw: s.deposited_decision_mw,
            carried_committed_mw: s.carried_committed_mw,
        }
    }
}

impl IntoWriteRecord for SimulationGenericViolationResult {
    type Record = GenericViolationWriteRecord;
    fn into_write_record(self, node_id: i32) -> GenericViolationWriteRecord {
        let s = self;
        GenericViolationWriteRecord {
            stage_id: s.stage_id,
            node_id,
            block_id: s.block_id,
            constraint_id: s.constraint_id,
            slack_value: s.slack_value,
            slack_cost: s.slack_cost,
        }
    }
}

impl From<SimulationTransitSeedResult> for TransitSeedWriteRecord {
    fn from(src: SimulationTransitSeedResult) -> Self {
        Self {
            hydro_id: src.hydro_id,
            start_date: src.start_date,
            end_date: src.end_date,
            value_m3s: src.value_m3s,
        }
    }
}

/// Convert a `Simulation*Result` into its cobre-io write record, stamping the
/// stage's `node_id` (the source carries no node axis).
///
/// A local trait rather than `From<(S, i32)>`: the orphan rule forbids
/// implementing the foreign `From` for a foreign write-record type keyed on a
/// tuple, but a local trait may be implemented on the local source types.
trait IntoWriteRecord {
    type Record;
    fn into_write_record(self, node_id: i32) -> Self::Record;
}

/// Broadcast the stage's `node_id` onto every record while converting it.
fn with_node<S: IntoWriteRecord>(records: Vec<S>, node_id: i32) -> Vec<S::Record> {
    records
        .into_iter()
        .map(|r| r.into_write_record(node_id))
        .collect()
}

impl From<SimulationStageResult> for StageWritePayload {
    fn from(src: SimulationStageResult) -> Self {
        let node_id = src.node_id.0;
        Self {
            stage_id: src.stage_id,
            node_id,
            costs: with_node(src.costs, node_id),
            hydros: with_node(src.hydros, node_id),
            hydro_bus_generation: with_node(src.hydro_bus_generation, node_id),
            thermals: with_node(src.thermals, node_id),
            exchanges: with_node(src.exchanges, node_id),
            buses: with_node(src.buses, node_id),
            pumping_stations: with_node(src.pumping_stations, node_id),
            contracts: with_node(src.contracts, node_id),
            non_controllables: with_node(src.non_controllables, node_id),
            inflow_lags: with_node(src.inflow_lags, node_id),
            transit_buckets: with_node(src.transit_buckets, node_id),
            generic_violations: with_node(src.generic_violations, node_id),
            anticipated_lanes: with_node(src.anticipated_lanes, node_id),
        }
    }
}

impl From<SimulationScenarioResult> for ScenarioWritePayload {
    fn from(src: SimulationScenarioResult) -> Self {
        Self {
            scenario_id: src.scenario_id,
            stages: src.stages.into_iter().map(Into::into).collect(),
            transit_seed: src.transit_seed.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;
    use crate::simulation::ScenarioCategoryCosts;
    use cobre_io::output::simulation_writer::ScenarioWritePayload;

    fn make_cost(stage_id: u32, block_id: u32) -> SimulationCostResult {
        SimulationCostResult {
            stage_id,
            block_id: Some(block_id),
            total_cost: 1.0,
            immediate_cost: 2.0,
            future_cost: 3.0,
            discount_factor: 0.95,
            thermal_cost: 4.0,
            anticipated_thermal_cost: 0.0,
            contract_cost: 5.0,
            deficit_cost: 6.0,
            excess_cost: 7.0,
            storage_violation_cost: 8.0,
            filling_target_cost: 9.0,
            hydro_violation_cost: 10.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            turbined_violation_cost: 0.0,
            generation_violation_cost: 0.0,
            evaporation_violation_cost: 0.0,
            withdrawal_violation_cost: 0.0,
            inflow_penalty_cost: 11.0,
            generic_violation_cost: 12.0,
            spillage_cost: 13.0,
            turbined_cost: 14.0,
            curtailment_cost: 15.0,
            exchange_cost: 16.0,
            pumping_cost: 17.0,
        }
    }

    fn make_hydro(stage_id: u32, block_id: u32) -> SimulationHydroResult {
        SimulationHydroResult {
            stage_id,
            block_id: Some(block_id),
            hydro_id: 1,
            turbined_m3s: 100.0,
            spillage_m3s: 10.0,
            evaporation_m3s: Some(1.0),
            diverted_inflow_m3s: Some(2.0),
            diverted_outflow_m3s: Some(3.0),
            incremental_inflow_m3s: 50.0,
            inflow_m3s: 55.0,
            storage_initial_hm3: 500.0,
            storage_final_hm3: 490.0,
            generation_mw: 200.0,
            equivalent_productivity_mw_per_m3s: 0.0,
            accumulated_productivity_mw_per_m3s: 0.0,
            incremental_inflow_energy_mw: 0.0,
            stored_energy_initial_mwh: 0.0,
            stored_energy_final_mwh: 0.0,
            spillage_cost: 0.0,
            water_value_per_hm3: 30.0,
            storage_binding_code: 0,
            operative_state_code: 1,
            turbined_slack_m3s: 0.0,
            outflow_slack_below_m3s: 0.0,
            outflow_slack_above_m3s: 0.0,
            generation_slack_mw: 0.0,
            storage_violation_below_hm3: 0.0,
            filling_target_violation_hm3: 0.0,
            evaporation_violation_pos_m3s: 0.0,
            evaporation_violation_neg_m3s: 0.0,
            inflow_nonnegativity_slack_m3s: 0.0,
            water_withdrawal_violation_pos_m3s: 0.0,
            water_withdrawal_violation_neg_m3s: 0.0,
            integrated_equivalent_productivity_mw_per_m3s: 0.0,
            integrated_accumulated_productivity_mw_per_m3s: 0.0,
            stored_energy_initial_mw: 0.0,
            stored_energy_final_mw: 0.0,
        }
    }

    fn make_hydro_bus(stage_id: u32, block_id: u32) -> SimulationHydroBusResult {
        SimulationHydroBusResult {
            stage_id,
            block_id: Some(block_id),
            hydro_id: 1,
            bus_id: 9,
            turbined_m3s: 40.0,
            generation_mw: 80.0,
        }
    }

    fn make_thermal(stage_id: u32, block_id: u32) -> SimulationThermalResult {
        SimulationThermalResult {
            stage_id,
            block_id: Some(block_id),
            thermal_id: 1,
            generation_mw: 150.0,
            generation_cost: 300.0,
            is_anticipated: false,
            anticipated_committed_mw: None,
            anticipated_decision_mw: None,
            operative_state_code: 1,
        }
    }

    fn make_exchange(stage_id: u32, block_id: u32) -> SimulationExchangeResult {
        SimulationExchangeResult {
            stage_id,
            block_id: Some(block_id),
            line_id: 1,
            direct_flow_mw: 50.0,
            reverse_flow_mw: 0.0,
            exchange_cost: 5.0,
            operative_state_code: 1,
        }
    }

    fn make_bus(stage_id: u32, block_id: u32) -> SimulationBusResult {
        SimulationBusResult {
            stage_id,
            block_id: Some(block_id),
            bus_id: 1,
            load_mw: 400.0,
            deficit_mw: 0.0,
            excess_mw: 0.0,
            spot_price: 75.0,
        }
    }

    fn make_pumping(stage_id: u32, block_id: u32) -> SimulationPumpingResult {
        SimulationPumpingResult {
            stage_id,
            block_id: Some(block_id),
            pumping_station_id: 1,
            pumped_flow_m3s: 20.0,
            power_consumption_mw: 10.0,
            pumping_cost: 50.0,
            operative_state_code: 1,
        }
    }

    fn make_contract(stage_id: u32, block_id: u32) -> SimulationContractResult {
        SimulationContractResult {
            stage_id,
            block_id: Some(block_id),
            contract_id: 1,
            power_mw: 100.0,
            price_per_mwh: 80.0,
            total_cost: 192.0,
            operative_state_code: 1,
        }
    }

    fn make_non_controllable(stage_id: u32, block_id: u32) -> SimulationNonControllableResult {
        SimulationNonControllableResult {
            stage_id,
            block_id: Some(block_id),
            non_controllable_id: 1,
            generation_mw: 60.0,
            available_mw: 70.0,
            curtailment_mw: 10.0,
            curtailment_cost: 100.0,
            operative_state_code: 1,
        }
    }

    fn make_inflow_lag(stage_id: u32) -> SimulationInflowLagResult {
        SimulationInflowLagResult {
            stage_id,
            hydro_id: 1,
            lag_index: 0,
            inflow_m3s: 45.0,
        }
    }

    fn make_transit_bucket(stage_id: u32, lag: u32) -> SimulationTransitBucketResult {
        SimulationTransitBucketResult {
            stage_id,
            hydro_id: 1,
            lag,
            in_transit_volume_hm3: 12.0,
            delayed_arrival_hm3: if lag == 1 { 4.0 } else { 0.0 },
        }
    }

    fn make_generic_violation(stage_id: u32, block_id: u32) -> SimulationGenericViolationResult {
        SimulationGenericViolationResult {
            stage_id,
            block_id: Some(block_id),
            constraint_id: 1,
            slack_value: 0.5,
            slack_cost: 1000.0,
        }
    }

    fn make_stage(stage_id: u32) -> SimulationStageResult {
        SimulationStageResult {
            stage_id,
            node_id: NodeId(stage_id as i32),
            costs: vec![make_cost(stage_id, 0)],
            hydros: vec![make_hydro(stage_id, 0)],
            hydro_bus_generation: vec![make_hydro_bus(stage_id, 0)],
            thermals: vec![make_thermal(stage_id, 0)],
            exchanges: vec![make_exchange(stage_id, 0)],
            buses: vec![make_bus(stage_id, 0)],
            pumping_stations: vec![make_pumping(stage_id, 0)],
            contracts: vec![make_contract(stage_id, 0)],
            non_controllables: vec![make_non_controllable(stage_id, 0)],
            inflow_lags: vec![make_inflow_lag(stage_id)],
            transit_buckets: vec![
                make_transit_bucket(stage_id, 1),
                make_transit_bucket(stage_id, 2),
            ],
            generic_violations: vec![make_generic_violation(stage_id, 0)],
            anticipated_lanes: vec![],
        }
    }

    fn make_category_costs() -> ScenarioCategoryCosts {
        ScenarioCategoryCosts {
            resource_cost: 100.0,
            recourse_cost: 50.0,
            violation_cost: 10.0,
            regularization_cost: 5.0,
            imputed_cost: 2.0,
        }
    }

    #[test]
    fn convert_scenario_result_to_write_payload_round_trip() {
        let scenario = SimulationScenarioResult {
            scenario_id: 7,
            total_cost: 167.0,
            per_category_costs: make_category_costs(),
            stages: vec![make_stage(0), make_stage(1)],
            transit_seed: vec![SimulationTransitSeedResult {
                hydro_id: 2,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
                value_m3s: 12.5,
            }],
        };

        let payload = ScenarioWritePayload::from(scenario);

        assert_eq!(payload.scenario_id, 7);
        assert_eq!(payload.stages.len(), 2);
        assert_eq!(payload.transit_seed.len(), 1);
        assert_eq!(payload.transit_seed[0].hydro_id, 2);
        assert_eq!(payload.transit_seed[0].value_m3s, 12.5);

        let stage0 = &payload.stages[0];
        assert_eq!(stage0.stage_id, 0);
        assert_eq!(stage0.costs[0].total_cost, 1.0);
        assert_eq!(stage0.costs[0].discount_factor, 0.95);
        assert_eq!(stage0.hydros[0].turbined_m3s, 100.0);
        assert_eq!(stage0.hydros[0].storage_initial_hm3, 500.0);
        assert_eq!(stage0.hydro_bus_generation[0].bus_id, 9);
        assert_eq!(stage0.hydro_bus_generation[0].turbined_m3s, 40.0);
        assert_eq!(stage0.thermals[0].generation_mw, 150.0);
        assert_eq!(stage0.exchanges[0].direct_flow_mw, 50.0);
        assert_eq!(stage0.buses[0].spot_price, 75.0);
        assert_eq!(stage0.pumping_stations[0].pumped_flow_m3s, 20.0);
        assert_eq!(stage0.contracts[0].price_per_mwh, 80.0);
        assert_eq!(stage0.non_controllables[0].curtailment_mw, 10.0);
        assert_eq!(stage0.inflow_lags[0].inflow_m3s, 45.0);
        assert_eq!(stage0.transit_buckets.len(), 2);
        assert_eq!(stage0.transit_buckets[0].lag, 1);
        assert_eq!(stage0.transit_buckets[0].delayed_arrival_hm3, 4.0);
        assert_eq!(stage0.transit_buckets[1].lag, 2);
        assert_eq!(stage0.transit_buckets[1].delayed_arrival_hm3, 0.0);
        assert_eq!(stage0.generic_violations[0].slack_cost, 1000.0);

        let stage1 = &payload.stages[1];
        assert_eq!(stage1.stage_id, 1);
    }

    #[test]
    fn convert_stage_result_preserves_all_entity_types() {
        let stage = make_stage(3);
        let payload = StageWritePayload::from(stage);

        assert_eq!(payload.stage_id, 3);
        assert!(!payload.costs.is_empty(), "costs must be non-empty");
        assert!(!payload.hydros.is_empty(), "hydros must be non-empty");
        assert!(
            !payload.hydro_bus_generation.is_empty(),
            "hydro_bus_generation must be non-empty"
        );
        assert!(!payload.thermals.is_empty(), "thermals must be non-empty");
        assert!(!payload.exchanges.is_empty(), "exchanges must be non-empty");
        assert!(!payload.buses.is_empty(), "buses must be non-empty");
        assert!(
            !payload.pumping_stations.is_empty(),
            "pumping_stations must be non-empty"
        );
        assert!(!payload.contracts.is_empty(), "contracts must be non-empty");
        assert!(
            !payload.non_controllables.is_empty(),
            "non_controllables must be non-empty"
        );
        assert!(
            !payload.inflow_lags.is_empty(),
            "inflow_lags must be non-empty"
        );
        assert!(
            !payload.transit_buckets.is_empty(),
            "transit_buckets must be non-empty"
        );
        assert!(
            !payload.generic_violations.is_empty(),
            "generic_violations must be non-empty"
        );

        assert_eq!(payload.costs[0].stage_id, 3);
        assert_eq!(payload.hydros[0].hydro_id, 1);
        assert_eq!(payload.hydro_bus_generation[0].hydro_id, 1);
        assert_eq!(payload.hydro_bus_generation[0].generation_mw, 80.0);
        assert_eq!(payload.thermals[0].thermal_id, 1);
        assert_eq!(payload.exchanges[0].line_id, 1);
        assert_eq!(payload.buses[0].bus_id, 1);
        assert_eq!(payload.pumping_stations[0].pumping_station_id, 1);
        assert_eq!(payload.contracts[0].contract_id, 1);
        assert_eq!(payload.non_controllables[0].non_controllable_id, 1);
        assert_eq!(payload.inflow_lags[0].lag_index, 0);
        assert_eq!(payload.transit_buckets[0].lag, 1);
        assert_eq!(payload.transit_buckets[0].in_transit_volume_hm3, 12.0);
        assert_eq!(payload.generic_violations[0].constraint_id, 1);
    }
}
