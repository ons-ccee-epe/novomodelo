//! Construction-time validation helpers for [`SystemBuilder::build()`](crate::SystemBuilder::build).

use std::collections::HashMap;

use crate::{
    Bus, EnergyContract, EntityId, Hydro, HydroUnit, HydroUnitGroup, HydroUnitGroupViolation,
    HydroUnitInitialState, HydroUnitStateViolation, HydroUnitStatus, HydroUnitViolation, Line,
    NeumaierAccumulator, NonControllableSource, PumpingStation, Stage, Thermal, ThermalUnit,
    ThermalUnitInitialState, ThermalUnitStateViolation, ThermalUnitStatus, ThermalUnitTransition,
    ThermalUnitViolation, ValidationError,
};

pub(crate) trait HasId {
    fn entity_id(&self) -> EntityId;
}

impl HasId for Bus {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Line {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Hydro {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Thermal {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for PumpingStation {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for EnergyContract {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for NonControllableSource {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}

pub(crate) fn build_index<T: HasId>(entities: &[T]) -> HashMap<EntityId, usize> {
    let mut index = HashMap::with_capacity(entities.len());
    for (i, entity) in entities.iter().enumerate() {
        index.insert(entity.entity_id(), i);
    }
    index
}

/// Keys are `i32` stage IDs, negative for pre-study stages.
pub(crate) fn build_stage_index(stages: &[Stage]) -> HashMap<i32, usize> {
    let mut index = HashMap::with_capacity(stages.len());
    for (i, stage) in stages.iter().enumerate() {
        index.insert(stage.id, i);
    }
    index
}

/// Detect duplicate entity IDs in a **sorted** slice by scanning adjacent pairs.
///
/// Contract: the caller must sort `entities` by `entity_id()` ascending before
/// calling (the builder sorts every collection in `build`). On unsorted input
/// the adjacent-pair scan silently misses non-adjacent duplicates, letting a
/// colliding id slip past validation.
pub(crate) fn check_duplicates<T: HasId>(
    entities: &[T],
    entity_type: &'static str,
    errors: &mut Vec<ValidationError>,
) {
    for window in entities.windows(2) {
        if window[0].entity_id() == window[1].entity_id() {
            errors.push(ValidationError::DuplicateId {
                entity_type,
                id: window[0].entity_id(),
            });
        }
    }
}

/// Parallel i32-keyed duplicate scan for stages, which key on a domain `i32` id
/// (negative for pre-study) and so cannot reuse the [`HasId`]-generic [`check_duplicates`].
///
/// Contract: `stages` must be sorted ascending by `id` before calling. The
/// adjacent-pair scan misses non-adjacent duplicates on unsorted input, letting
/// `build_stage_index` silently overwrite a colliding stage with no validation error.
pub(crate) fn check_duplicate_stages(stages: &[Stage], errors: &mut Vec<ValidationError>) {
    for window in stages.windows(2) {
        if window[0].id == window[1].id {
            errors.push(ValidationError::DuplicateId {
                entity_type: "Stage",
                id: EntityId(window[0].id),
            });
        }
    }
}

/// Duplicate unit ids within each plant; units must be id-sorted
/// ([`Thermal::sort_units`]), and output is in `(thermal_id, unit_id)` order whatever the
/// order of same-id plants.
pub(crate) fn check_duplicate_thermal_unit_ids(
    thermals: &[Thermal],
    errors: &mut Vec<ValidationError>,
) {
    let mut duplicates: Vec<(EntityId, EntityId)> = thermals
        .iter()
        .flat_map(|thermal| {
            thermal
                .units
                .windows(2)
                .filter(|pair| pair[0].id == pair[1].id)
                .map(|pair| (thermal.id, pair[0].id))
        })
        .collect();
    duplicates.sort_by_key(|(thermal_id, unit_id)| (thermal_id.0, unit_id.0));
    errors.extend(duplicates.into_iter().map(|(thermal_id, unit_id)| {
        ValidationError::InvalidThermalUnit {
            thermal_id,
            unit_id,
            violation: ThermalUnitViolation::DuplicateId,
        }
    }));
}

/// Duplicate `(thermal_id, unit_id)` records, one error per equal-key adjacent pair.
///
/// Contract: `states` must be sorted by `(thermal_id, unit_id)` before calling, as `build`
/// sorts them; on unsorted input the adjacent-pair scan misses non-adjacent duplicates.
pub(crate) fn check_duplicate_thermal_unit_states(
    states: &[ThermalUnitInitialState],
    errors: &mut Vec<ValidationError>,
) {
    for pair in states.windows(2) {
        if pair[0].thermal_id == pair[1].thermal_id && pair[0].unit_id == pair[1].unit_id {
            errors.push(ValidationError::InvalidThermalUnitInitialState {
                thermal_id: pair[0].thermal_id,
                unit_id: pair[0].unit_id,
                violation: ThermalUnitStateViolation::DuplicateRecord,
            });
        }
    }
}

/// Duplicate group ids within each plant; groups must be id-sorted
/// ([`Hydro::sort_unit_groups`]), and output is in `(hydro_id, group_id)` order whatever the
/// order of same-id plants.
pub(crate) fn check_duplicate_hydro_unit_group_ids(
    hydros: &[Hydro],
    errors: &mut Vec<ValidationError>,
) {
    let mut duplicates: Vec<(EntityId, EntityId)> = hydros
        .iter()
        .flat_map(|hydro| {
            hydro
                .unit_groups
                .windows(2)
                .filter(|pair| pair[0].id == pair[1].id)
                .map(|pair| (hydro.id, pair[0].id))
        })
        .collect();
    duplicates.sort_by_key(|(hydro_id, group_id)| (hydro_id.0, group_id.0));
    errors.extend(duplicates.into_iter().map(|(hydro_id, group_id)| {
        ValidationError::InvalidHydroUnitGroup {
            hydro_id,
            group_id,
            violation: HydroUnitGroupViolation::DuplicateId,
        }
    }));
}

/// Duplicate unit ids within each group; units must be id-sorted
/// ([`HydroUnitGroup::sort_units`](crate::HydroUnitGroup::sort_units)), and output is in
/// `(hydro_id, group_id, unit_id)` order whatever the order of same-id plants and groups.
pub(crate) fn check_duplicate_hydro_unit_ids(hydros: &[Hydro], errors: &mut Vec<ValidationError>) {
    let mut duplicates: Vec<(EntityId, EntityId, EntityId)> = hydros
        .iter()
        .flat_map(|hydro| {
            hydro.unit_groups.iter().flat_map(move |group| {
                group
                    .units
                    .windows(2)
                    .filter(|pair| pair[0].id == pair[1].id)
                    .map(move |pair| (hydro.id, group.id, pair[0].id))
            })
        })
        .collect();
    duplicates.sort_by_key(|(hydro_id, group_id, unit_id)| (hydro_id.0, group_id.0, unit_id.0));
    errors.extend(duplicates.into_iter().map(|(hydro_id, group_id, unit_id)| {
        ValidationError::InvalidHydroUnit {
            hydro_id,
            group_id,
            unit_id,
            violation: HydroUnitViolation::DuplicateId,
        }
    }));
}

/// Duplicate `(hydro_id, group_id, unit_id)` records, one error per equal-key adjacent pair.
///
/// Contract: `states` must be sorted by `(hydro_id, group_id, unit_id)` before calling, as
/// `build` sorts them; on unsorted input the adjacent-pair scan misses non-adjacent duplicates.
pub(crate) fn check_duplicate_hydro_unit_states(
    states: &[HydroUnitInitialState],
    errors: &mut Vec<ValidationError>,
) {
    for pair in states.windows(2) {
        if pair[0].hydro_id == pair[1].hydro_id
            && pair[0].group_id == pair[1].group_id
            && pair[0].unit_id == pair[1].unit_id
        {
            errors.push(ValidationError::InvalidHydroUnitInitialState {
                hydro_id: pair[0].hydro_id,
                group_id: pair[0].group_id,
                unit_id: pair[0].unit_id,
                violation: HydroUnitStateViolation::DuplicateRecord,
            });
        }
    }
}

/// Bundles entity slices needed for cross-reference validation.
pub(crate) struct CrossRefEntities<'a> {
    pub(crate) lines: &'a [Line],
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) pumping_stations: &'a [PumpingStation],
    pub(crate) contracts: &'a [EnergyContract],
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
}

/// Validate every [`EntityId`] cross-reference field against the appropriate index,
/// appending all violations to `errors` — no short-circuiting on the first error.
pub(crate) fn validate_cross_references(
    entities: &CrossRefEntities<'_>,
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    validate_line_refs(entities.lines, bus_index, errors);
    validate_hydro_refs(entities.hydros, hydro_index, errors);
    validate_thermal_refs(entities.thermals, bus_index, errors);
    validate_pumping_station_refs(entities.pumping_stations, bus_index, hydro_index, errors);
    validate_contract_refs(entities.contracts, bus_index, errors);
    validate_ncs_refs(entities.non_controllable_sources, bus_index, errors);
}

fn validate_line_refs(
    lines: &[Line],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for line in lines {
        if !bus_index.contains_key(&line.source_bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Line",
                source_id: line.id,
                field_name: "source_bus_id",
                referenced_id: line.source_bus_id,
                expected_type: "Bus",
            });
        }
        if !bus_index.contains_key(&line.target_bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Line",
                source_id: line.id,
                field_name: "target_bus_id",
                referenced_id: line.target_bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_hydro_refs(
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for hydro in hydros {
        if let Some(downstream_id) = hydro.downstream_id
            && !hydro_index.contains_key(&downstream_id)
        {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "downstream_id",
                referenced_id: downstream_id,
                expected_type: "Hydro",
            });
        }
        if let Some(ref diversion) = hydro.diversion
            && !hydro_index.contains_key(&diversion.downstream_id)
        {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "diversion.downstream_id",
                referenced_id: diversion.downstream_id,
                expected_type: "Hydro",
            });
        }
    }
}

fn validate_thermal_refs(
    thermals: &[Thermal],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for thermal in thermals {
        if !bus_index.contains_key(&thermal.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Thermal",
                source_id: thermal.id,
                field_name: "bus_id",
                referenced_id: thermal.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_pumping_station_refs(
    pumping_stations: &[PumpingStation],
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for ps in pumping_stations {
        if !bus_index.contains_key(&ps.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "bus_id",
                referenced_id: ps.bus_id,
                expected_type: "Bus",
            });
        }
        if !hydro_index.contains_key(&ps.source_hydro_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "source_hydro_id",
                referenced_id: ps.source_hydro_id,
                expected_type: "Hydro",
            });
        }
        if !hydro_index.contains_key(&ps.destination_hydro_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "destination_hydro_id",
                referenced_id: ps.destination_hydro_id,
                expected_type: "Hydro",
            });
        }
    }
}

fn validate_contract_refs(
    contracts: &[EnergyContract],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for contract in contracts {
        if !bus_index.contains_key(&contract.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "EnergyContract",
                source_id: contract.id,
                field_name: "bus_id",
                referenced_id: contract.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_ncs_refs(
    non_controllable_sources: &[NonControllableSource],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for ncs in non_controllable_sources {
        if !bus_index.contains_key(&ncs.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "NonControllableSource",
                source_id: ncs.id,
                field_name: "bus_id",
                referenced_id: ncs.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

/// Validate the filling config of every hydro that has one, appending all
/// violations to `errors` (no short-circuiting on the first).
///
/// For each `filling: Some(config)`:
/// - `filling_min_rate_m3s` must be non-negative: zero is valid (no minimum
///   accumulation required); only a negative or NaN rate is rejected.
/// - `entry_stage_id` must be `Some`.
/// - `start_stage_id < entry_stage_id`. Mirrors the novomodelo-io ordering guard here so
///   a `System` built directly via the public [`SystemBuilder`](super::SystemBuilder)
///   still rejects an inverted config, which otherwise mis-phases the reservoir in
///   the solver with no error. Validating `start_stage_id` against the study stage
///   set stays in novomodelo-io, which alone holds the horizon.
pub(crate) fn validate_filling_configs(hydros: &[Hydro], errors: &mut Vec<ValidationError>) {
    for hydro in hydros {
        if let Some(filling) = &hydro.filling {
            if filling.filling_min_rate_m3s.is_nan() || filling.filling_min_rate_m3s < 0.0 {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling_min_rate_m3s must be non-negative".to_string(),
                });
            }
            if hydro.entry_stage_id.is_none() {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling requires entry_stage_id to be set".to_string(),
                });
            }
            if let Some(entry) = hydro.entry_stage_id
                && filling.start_stage_id >= entry
            {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling start_stage_id must be less than entry_stage_id".to_string(),
                });
            }
        }
    }
}

fn is_finite_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

pub(crate) fn bounds_are_valid(min: f64, max: f64) -> bool {
    min.is_finite() && max.is_finite() && 0.0 <= min && min <= max
}

fn validate_ramp(
    levels: &[f64],
    transition: ThermalUnitTransition,
    min_generation_mw: f64,
    reject: &mut impl FnMut(ThermalUnitViolation),
) {
    for (index, &level_mw) in levels.iter().enumerate() {
        let in_range = 0.0 <= level_mw && level_mw <= min_generation_mw;
        if !in_range {
            reject(ThermalUnitViolation::RampLevelOutOfRange {
                transition,
                index,
                level_mw,
            });
        }
    }
    for (previous_index, pair) in levels.windows(2).enumerate() {
        let ordered = match transition {
            ThermalUnitTransition::StartingUp => pair[0] <= pair[1],
            ThermalUnitTransition::ShuttingDown => pair[0] >= pair[1],
        };
        if !ordered {
            reject(ThermalUnitViolation::RampNotMonotone {
                transition,
                index: previous_index + 1,
            });
        }
    }
    let endpoint = match transition {
        ThermalUnitTransition::StartingUp => levels.last(),
        ThermalUnitTransition::ShuttingDown => levels.first(),
    };
    if let Some(&level_mw) = endpoint {
        #[allow(clippy::float_cmp)] // exact by contract: a tolerance would admit a wrong endpoint
        let at_minimum = level_mw == min_generation_mw;
        if !at_minimum {
            reject(ThermalUnitViolation::RampEndpointNotAtMinimum {
                transition,
                level_mw,
            });
        }
    }
}

/// The tolerance is scaled by `max(1, envelope max)`, so summation rounding never rejects an
/// envelope its units cover.
const UNIT_CAPACITY_RELATIVE_TOLERANCE: f64 = 1e-9;

/// Compensated sum of `unit_maxima` when it falls short of `envelope_max`, else `None`.
pub(crate) fn capacity_shortfall(
    envelope_max: f64,
    unit_maxima: impl Iterator<Item = f64>,
) -> Option<f64> {
    let mut sum = NeumaierAccumulator::new();
    for unit_max in unit_maxima {
        sum.add(unit_max);
    }
    let units_max = sum.value();
    let tolerance = UNIT_CAPACITY_RELATIVE_TOLERANCE * envelope_max.max(1.0);
    // An infinite envelope maximum makes the tolerance infinite, which would accept any sum.
    let covered = envelope_max.is_finite() && envelope_max - units_max <= tolerance;
    (!covered).then_some(units_max)
}

fn validate_unit_capacity(thermal: &Thermal, errors: &mut Vec<ValidationError>) {
    let generation_bounds_hold = thermal
        .units
        .iter()
        .all(|unit| bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw));
    if thermal.units.is_empty() || !generation_bounds_hold {
        return;
    }
    if let Some(units_max_mw) = capacity_shortfall(
        thermal.max_generation_mw,
        thermal.units.iter().map(|unit| unit.max_generation_mw),
    ) {
        errors.push(ValidationError::InsufficientThermalUnitCapacity {
            thermal_id: thermal.id,
            units_max_mw,
            plant_max_mw: thermal.max_generation_mw,
        });
    }
}

/// Appends every unit violation and plant capacity shortfall to `errors`. A unit with invalid
/// generation bounds skips the ramp rules and exempts its plant from the capacity rule, so one bad
/// bound reports once; a plant without units has no capacity rule.
pub(crate) fn validate_thermal_units(thermals: &[Thermal], errors: &mut Vec<ValidationError>) {
    for thermal in thermals {
        for unit in &thermal.units {
            let mut reject = |violation| {
                errors.push(ValidationError::InvalidThermalUnit {
                    thermal_id: thermal.id,
                    unit_id: unit.id,
                    violation,
                });
            };
            if !bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw) {
                reject(ThermalUnitViolation::InvalidGenerationBounds {
                    min_generation_mw: unit.min_generation_mw,
                    max_generation_mw: unit.max_generation_mw,
                });
            }
            if !is_finite_non_negative(unit.min_up_time_hours) {
                reject(ThermalUnitViolation::InvalidMinUpTime {
                    min_up_time_hours: unit.min_up_time_hours,
                });
            }
            if !is_finite_non_negative(unit.min_down_time_hours) {
                reject(ThermalUnitViolation::InvalidMinDownTime {
                    min_down_time_hours: unit.min_down_time_hours,
                });
            }
            if bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw) {
                validate_ramp(
                    &unit.startup_ramp_mw,
                    ThermalUnitTransition::StartingUp,
                    unit.min_generation_mw,
                    &mut reject,
                );
                validate_ramp(
                    &unit.shutdown_ramp_mw,
                    ThermalUnitTransition::ShuttingDown,
                    unit.min_generation_mw,
                    &mut reject,
                );
            }
        }
        validate_unit_capacity(thermal, errors);
    }
}

fn status_generation_violation(
    state: &ThermalUnitInitialState,
    unit: &ThermalUnit,
) -> Option<ThermalUnitStateViolation> {
    let g = state.previous_generation_mw;
    let (min, max) = (unit.min_generation_mw, unit.max_generation_mw);
    let off_trajectory = |transition| {
        state.ramp_position(unit).is_none().then_some(
            ThermalUnitStateViolation::GenerationNotOnTrajectory {
                transition,
                previous_generation_mw: g,
            },
        )
    };
    match state.status {
        ThermalUnitStatus::Off => {
            (g != 0.0).then_some(ThermalUnitStateViolation::OffWithGeneration {
                previous_generation_mw: g,
            })
        }
        ThermalUnitStatus::On => {
            let within = min <= g && g <= max;
            (!within).then_some(ThermalUnitStateViolation::OnOutsideBounds {
                previous_generation_mw: g,
                min_generation_mw: min,
                max_generation_mw: max,
            })
        }
        ThermalUnitStatus::StartingUp => off_trajectory(ThermalUnitTransition::StartingUp),
        ThermalUnitStatus::ShuttingDown => off_trajectory(ThermalUnitTransition::ShuttingDown),
    }
}

/// Appends every initial-state violation to `errors`. The status rule runs only when the unit
/// resolves, `previous_generation_mw` is valid and the unit's generation bounds are valid, so
/// each bad input reports once.
///
/// Contract: each plant's units must be id-sorted ([`Thermal::sort_units`]) and free of duplicate
/// ids, and `thermal_index` must be [`build_index`] of `thermals`.
pub(crate) fn validate_thermal_unit_states(
    states: &[ThermalUnitInitialState],
    thermals: &[Thermal],
    thermal_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for state in states {
        let mut reject = |violation| {
            errors.push(ValidationError::InvalidThermalUnitInitialState {
                thermal_id: state.thermal_id,
                unit_id: state.unit_id,
                violation,
            });
        };
        let plant = thermal_index
            .get(&state.thermal_id)
            .map(|&position| &thermals[position]);
        let unit = plant.and_then(|plant| {
            plant
                .units
                .binary_search_by_key(&state.unit_id.0, |unit| unit.id.0)
                .ok()
                .map(|position| &plant.units[position])
        });
        if plant.is_none() {
            reject(ThermalUnitStateViolation::UnknownThermal);
        } else if unit.is_none() {
            reject(ThermalUnitStateViolation::UnknownUnit);
        }
        if !is_finite_non_negative(state.time_in_state_hours) {
            reject(ThermalUnitStateViolation::InvalidTimeInState {
                time_in_state_hours: state.time_in_state_hours,
            });
        }
        let generation_is_valid = is_finite_non_negative(state.previous_generation_mw);
        if !generation_is_valid {
            reject(ThermalUnitStateViolation::InvalidPreviousGeneration {
                previous_generation_mw: state.previous_generation_mw,
            });
        }
        if let Some(unit) = unit
            && generation_is_valid
            && bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw)
            && let Some(violation) = status_generation_violation(state, unit)
        {
            reject(violation);
        }
    }
}

/// Appends every unit violation and group capacity shortfall to `errors`. A unit with invalid
/// generation or turbined bounds exempts its group from both capacity rules, so one bad bound
/// reports once; a group without units has no capacity rule.
pub(crate) fn validate_hydro_units(hydros: &[Hydro], errors: &mut Vec<ValidationError>) {
    for hydro in hydros {
        for group in &hydro.unit_groups {
            for unit in &group.units {
                let mut reject = |violation| {
                    errors.push(ValidationError::InvalidHydroUnit {
                        hydro_id: hydro.id,
                        group_id: group.id,
                        unit_id: unit.id,
                        violation,
                    });
                };
                if !bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw) {
                    reject(HydroUnitViolation::InvalidGenerationBounds {
                        min_generation_mw: unit.min_generation_mw,
                        max_generation_mw: unit.max_generation_mw,
                    });
                }
                if !bounds_are_valid(unit.min_turbined_m3s, unit.max_turbined_m3s) {
                    reject(HydroUnitViolation::InvalidTurbinedBounds {
                        min_turbined_m3s: unit.min_turbined_m3s,
                        max_turbined_m3s: unit.max_turbined_m3s,
                    });
                }
                if !is_finite_non_negative(unit.min_up_time_hours) {
                    reject(HydroUnitViolation::InvalidMinUpTime {
                        min_up_time_hours: unit.min_up_time_hours,
                    });
                }
                if !is_finite_non_negative(unit.min_down_time_hours) {
                    reject(HydroUnitViolation::InvalidMinDownTime {
                        min_down_time_hours: unit.min_down_time_hours,
                    });
                }
            }
            validate_hydro_unit_capacity(hydro.id, group, errors);
        }
    }
}

fn validate_hydro_unit_capacity(
    hydro_id: EntityId,
    group: &HydroUnitGroup,
    errors: &mut Vec<ValidationError>,
) {
    let unit_bounds_hold = group.units.iter().all(|unit| {
        bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw)
            && bounds_are_valid(unit.min_turbined_m3s, unit.max_turbined_m3s)
    });
    if group.units.is_empty() || !unit_bounds_hold {
        return;
    }
    let mut reject = |violation| {
        errors.push(ValidationError::InvalidHydroUnitGroup {
            hydro_id,
            group_id: group.id,
            violation,
        });
    };
    if let Some(units_max_mw) = capacity_shortfall(
        group.max_generation_mw,
        group.units.iter().map(|unit| unit.max_generation_mw),
    ) {
        reject(HydroUnitGroupViolation::InsufficientGenerationCapacity {
            units_max_mw,
            group_max_mw: group.max_generation_mw,
        });
    }
    if let Some(units_max_m3s) = capacity_shortfall(
        group.max_turbined_m3s,
        group.units.iter().map(|unit| unit.max_turbined_m3s),
    ) {
        reject(HydroUnitGroupViolation::InsufficientTurbinedCapacity {
            units_max_m3s,
            group_max_m3s: group.max_turbined_m3s,
        });
    }
}

fn hydro_generation_status_violation(
    state: &HydroUnitInitialState,
    unit: &HydroUnit,
) -> Option<HydroUnitStateViolation> {
    let g = state.previous_generation_mw;
    let (min, max) = (unit.min_generation_mw, unit.max_generation_mw);
    match state.status {
        HydroUnitStatus::Off => (g != 0.0).then_some(HydroUnitStateViolation::OffWithGeneration {
            previous_generation_mw: g,
        }),
        HydroUnitStatus::On => {
            let within = min <= g && g <= max;
            (!within).then_some(HydroUnitStateViolation::OnGenerationOutsideBounds {
                previous_generation_mw: g,
                min_generation_mw: min,
                max_generation_mw: max,
            })
        }
    }
}

fn hydro_turbined_status_violation(
    state: &HydroUnitInitialState,
    unit: &HydroUnit,
) -> Option<HydroUnitStateViolation> {
    let q = state.previous_turbined_m3s;
    let (min, max) = (unit.min_turbined_m3s, unit.max_turbined_m3s);
    match state.status {
        HydroUnitStatus::Off => {
            (q != 0.0).then_some(HydroUnitStateViolation::OffWithTurbinedFlow {
                previous_turbined_m3s: q,
            })
        }
        HydroUnitStatus::On => {
            let within = min <= q && q <= max;
            (!within).then_some(HydroUnitStateViolation::OnTurbinedOutsideBounds {
                previous_turbined_m3s: q,
                min_turbined_m3s: min,
                max_turbined_m3s: max,
            })
        }
    }
}

/// Appends every initial-state violation to `errors`. Each quantity's status rule runs only when
/// the unit resolves and both that quantity's previous value and the unit's bounds for it are
/// valid, so each bad input reports once.
///
/// Contract: each plant's groups and each group's units must be id-sorted and free of duplicate
/// ids, and `hydro_index` must be [`build_index`] of `hydros`.
pub(crate) fn validate_hydro_unit_states(
    states: &[HydroUnitInitialState],
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for state in states {
        let mut reject = |violation| {
            errors.push(ValidationError::InvalidHydroUnitInitialState {
                hydro_id: state.hydro_id,
                group_id: state.group_id,
                unit_id: state.unit_id,
                violation,
            });
        };
        let plant = hydro_index
            .get(&state.hydro_id)
            .map(|&position| &hydros[position]);
        let group = plant.and_then(|plant| {
            plant
                .unit_groups
                .binary_search_by_key(&state.group_id.0, |group| group.id.0)
                .ok()
                .map(|position| &plant.unit_groups[position])
        });
        let unit = group.and_then(|group| {
            group
                .units
                .binary_search_by_key(&state.unit_id.0, |unit| unit.id.0)
                .ok()
                .map(|position| &group.units[position])
        });
        if plant.is_none() {
            reject(HydroUnitStateViolation::UnknownHydro);
        } else if group.is_none() {
            reject(HydroUnitStateViolation::UnknownGroup);
        } else if unit.is_none() {
            reject(HydroUnitStateViolation::UnknownUnit);
        }
        if !is_finite_non_negative(state.time_in_state_hours) {
            reject(HydroUnitStateViolation::InvalidTimeInState {
                time_in_state_hours: state.time_in_state_hours,
            });
        }
        let generation_is_valid = is_finite_non_negative(state.previous_generation_mw);
        if !generation_is_valid {
            reject(HydroUnitStateViolation::InvalidPreviousGeneration {
                previous_generation_mw: state.previous_generation_mw,
            });
        }
        let turbined_is_valid = is_finite_non_negative(state.previous_turbined_m3s);
        if !turbined_is_valid {
            reject(HydroUnitStateViolation::InvalidPreviousTurbined {
                previous_turbined_m3s: state.previous_turbined_m3s,
            });
        }
        if let Some(unit) = unit
            && generation_is_valid
            && bounds_are_valid(unit.min_generation_mw, unit.max_generation_mw)
            && let Some(violation) = hydro_generation_status_violation(state, unit)
        {
            reject(violation);
        }
        if let Some(unit) = unit
            && turbined_is_valid
            && bounds_are_valid(unit.min_turbined_m3s, unit.max_turbined_m3s)
            && let Some(violation) = hydro_turbined_status_violation(state, unit)
        {
            reject(violation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_index, check_duplicate_hydro_unit_group_ids, check_duplicate_hydro_unit_ids,
        check_duplicate_hydro_unit_states, check_duplicate_thermal_unit_ids,
        check_duplicate_thermal_unit_states, validate_hydro_unit_states, validate_hydro_units,
        validate_thermal_unit_states, validate_thermal_units,
    };
    use crate::test_support::{
        HydroSpec, HydroUnitSpec, HydroUnitStateSpec, MirrorUnitGroup, ThermalSpec,
        ThermalUnitSpec, ThermalUnitStateSpec, UnitGroupSpec, f64_bits_eq, make_hydro,
        make_hydro_unit, make_hydro_unit_state, make_thermal, make_thermal_unit,
        make_thermal_unit_state, make_unit_group,
    };
    use crate::{
        EntityId, Hydro, HydroUnit, HydroUnitGroup, HydroUnitGroupViolation, HydroUnitInitialState,
        HydroUnitStateViolation, HydroUnitStatus, HydroUnitViolation, Thermal, ThermalUnit,
        ThermalUnitInitialState, ThermalUnitStateViolation, ThermalUnitStatus,
        ThermalUnitTransition, ThermalUnitViolation, ValidationError,
    };

    fn plant_without_capacity_floor(units: Vec<ThermalUnit>) -> Thermal {
        make_thermal(ThermalSpec {
            id: 1,
            bus_id: 1,
            max_generation_mw: 0.0,
            units,
            ..Default::default()
        })
    }

    fn validated(units: Vec<ThermalUnit>) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        validate_thermal_units(&[plant_without_capacity_floor(units)], &mut errors);
        errors
    }

    fn plant_errors(max_generation_mw: f64, units: Vec<ThermalUnit>) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        validate_thermal_units(
            &[make_thermal(ThermalSpec {
                id: 1,
                bus_id: 1,
                max_generation_mw,
                units,
                ..Default::default()
            })],
            &mut errors,
        );
        errors
    }

    fn units_with_maxima(min_generation_mw: f64, maxima: &[f64]) -> Vec<ThermalUnit> {
        maxima
            .iter()
            .zip(1..)
            .map(|(&max_generation_mw, id)| {
                make_thermal_unit(ThermalUnitSpec {
                    id,
                    min_generation_mw,
                    max_generation_mw,
                    ..Default::default()
                })
            })
            .collect()
    }

    #[test]
    fn thermal_unit_scalar_rules_accept_zero_maximum_and_zero_durations() {
        let errors = validated(vec![
            make_thermal_unit(ThermalUnitSpec {
                id: 1,
                ..Default::default()
            }),
            make_thermal_unit(ThermalUnitSpec {
                id: 2,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                ..Default::default()
            }),
            make_thermal_unit(ThermalUnitSpec {
                id: 3,
                min_generation_mw: -0.0,
                max_generation_mw: 0.0,
                ..Default::default()
            }),
        ]);

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn thermal_unit_generation_bounds_reject_negative_inverted_and_non_finite() {
        let bounds = [
            (-1.0, 10.0),
            (50.0, 40.0),
            (0.0, f64::INFINITY),
            (f64::NAN, 10.0),
            (0.0, f64::NAN),
        ];
        let errors = validated(
            bounds
                .into_iter()
                .zip(1..)
                .map(|((min_generation_mw, max_generation_mw), id)| {
                    make_thermal_unit(ThermalUnitSpec {
                        id,
                        min_generation_mw,
                        max_generation_mw,
                        ..Default::default()
                    })
                })
                .collect(),
        );

        assert!(
            matches!(
                errors.as_slice(),
                [
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(1),
                        violation: ThermalUnitViolation::InvalidGenerationBounds { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(2),
                        violation: ThermalUnitViolation::InvalidGenerationBounds { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(3),
                        violation: ThermalUnitViolation::InvalidGenerationBounds { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(4),
                        violation: ThermalUnitViolation::InvalidGenerationBounds { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(5),
                        violation: ThermalUnitViolation::InvalidGenerationBounds { .. },
                        ..
                    },
                ]
            ),
            "unexpected errors: {errors:?}"
        );
    }

    #[test]
    fn thermal_unit_durations_reject_negative_and_non_finite() {
        let errors = validated(vec![
            make_thermal_unit(ThermalUnitSpec {
                id: 1,
                min_up_time_hours: -1.0,
                ..Default::default()
            }),
            make_thermal_unit(ThermalUnitSpec {
                id: 2,
                min_down_time_hours: f64::INFINITY,
                ..Default::default()
            }),
            make_thermal_unit(ThermalUnitSpec {
                id: 3,
                min_up_time_hours: f64::NAN,
                min_down_time_hours: -0.5,
                ..Default::default()
            }),
        ]);

        assert!(
            matches!(
                errors.as_slice(),
                [
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(1),
                        violation: ThermalUnitViolation::InvalidMinUpTime { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(2),
                        violation: ThermalUnitViolation::InvalidMinDownTime { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(3),
                        violation: ThermalUnitViolation::InvalidMinUpTime { .. },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        unit_id: EntityId(3),
                        violation: ThermalUnitViolation::InvalidMinDownTime { .. },
                        ..
                    },
                ]
            ),
            "unexpected errors: {errors:?}"
        );
    }

    fn ramped_unit(id: i32, startup_ramp_mw: &[f64], shutdown_ramp_mw: &[f64]) -> ThermalUnit {
        make_thermal_unit(ThermalUnitSpec {
            id,
            min_generation_mw: 20.0,
            max_generation_mw: 60.0,
            startup_ramp_mw: startup_ramp_mw.to_vec(),
            shutdown_ramp_mw: shutdown_ramp_mw.to_vec(),
            ..Default::default()
        })
    }

    fn invalid_unit_1(violation: ThermalUnitViolation) -> ValidationError {
        ValidationError::InvalidThermalUnit {
            thermal_id: EntityId(1),
            unit_id: EntityId(1),
            violation,
        }
    }

    #[test]
    fn thermal_unit_ramps_accept_empty_flat_and_endpoint_at_minimum() {
        let errors = validated(vec![
            ramped_unit(1, &[], &[]),
            ramped_unit(2, &[0.0, 20.0], &[20.0, 0.0]),
            ramped_unit(3, &[20.0, 20.0], &[20.0]),
            ramped_unit(4, &[5.0, 10.0, 15.0, 20.0], &[20.0, 5.0]),
        ]);

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn thermal_unit_startup_ramp_violations_are_reported_in_rule_order() {
        let errors = validated(vec![ramped_unit(1, &[10.0, 25.0, 5.0], &[])]);

        assert_eq!(
            errors,
            vec![
                invalid_unit_1(ThermalUnitViolation::RampLevelOutOfRange {
                    transition: ThermalUnitTransition::StartingUp,
                    index: 1,
                    level_mw: 25.0,
                }),
                invalid_unit_1(ThermalUnitViolation::RampNotMonotone {
                    transition: ThermalUnitTransition::StartingUp,
                    index: 2,
                }),
                invalid_unit_1(ThermalUnitViolation::RampEndpointNotAtMinimum {
                    transition: ThermalUnitTransition::StartingUp,
                    level_mw: 5.0,
                }),
            ]
        );
    }

    #[test]
    fn thermal_unit_shutdown_ramp_violations_are_reported_in_rule_order() {
        let errors = validated(vec![ramped_unit(1, &[], &[15.0, 18.0, -1.0])]);

        assert_eq!(
            errors,
            vec![
                invalid_unit_1(ThermalUnitViolation::RampLevelOutOfRange {
                    transition: ThermalUnitTransition::ShuttingDown,
                    index: 2,
                    level_mw: -1.0,
                }),
                invalid_unit_1(ThermalUnitViolation::RampNotMonotone {
                    transition: ThermalUnitTransition::ShuttingDown,
                    index: 1,
                }),
                invalid_unit_1(ThermalUnitViolation::RampEndpointNotAtMinimum {
                    transition: ThermalUnitTransition::ShuttingDown,
                    level_mw: 15.0,
                }),
            ]
        );
    }

    #[test]
    fn thermal_unit_ramp_with_nan_level_is_rejected() {
        let errors = validated(vec![ramped_unit(1, &[f64::NAN, 20.0], &[])]);

        assert!(
            matches!(
                errors.as_slice(),
                [
                    ValidationError::InvalidThermalUnit {
                        violation: ThermalUnitViolation::RampLevelOutOfRange {
                            transition: ThermalUnitTransition::StartingUp,
                            index: 0,
                            level_mw,
                        },
                        ..
                    },
                    ValidationError::InvalidThermalUnit {
                        violation: ThermalUnitViolation::RampNotMonotone {
                            transition: ThermalUnitTransition::StartingUp,
                            index: 1,
                        },
                        ..
                    },
                ] if level_mw.is_nan()
            ),
            "unexpected errors: {errors:?}"
        );
    }

    #[test]
    fn thermal_unit_ramp_rules_are_skipped_when_generation_bounds_fail() {
        let errors = validated(vec![make_thermal_unit(ThermalUnitSpec {
            id: 1,
            min_generation_mw: 50.0,
            max_generation_mw: 40.0,
            startup_ramp_mw: vec![60.0],
            ..Default::default()
        })]);

        assert_eq!(
            errors,
            vec![invalid_unit_1(
                ThermalUnitViolation::InvalidGenerationBounds {
                    min_generation_mw: 50.0,
                    max_generation_mw: 40.0,
                }
            )]
        );
    }

    #[test]
    fn duplicate_thermal_unit_ids_are_reported_in_key_order() {
        let plant = |id, unit_ids: &[i32]| {
            make_thermal(ThermalSpec {
                id,
                bus_id: 1,
                units: unit_ids
                    .iter()
                    .map(|&unit_id| {
                        make_thermal_unit(ThermalUnitSpec {
                            id: unit_id,
                            ..Default::default()
                        })
                    })
                    .collect(),
                ..Default::default()
            })
        };
        let duplicate = |thermal_id, unit_id| ValidationError::InvalidThermalUnit {
            thermal_id: EntityId(thermal_id),
            unit_id: EntityId(unit_id),
            violation: ThermalUnitViolation::DuplicateId,
        };

        let mut errors = Vec::new();
        check_duplicate_thermal_unit_ids(
            &[plant(2, &[4, 4]), plant(2, &[3, 3]), plant(1, &[5, 5, 5])],
            &mut errors,
        );

        assert_eq!(
            errors,
            vec![
                duplicate(1, 5),
                duplicate(1, 5),
                duplicate(2, 3),
                duplicate(2, 4),
            ]
        );
    }

    #[test]
    fn unit_capacity_accepts_a_sum_within_tolerance_of_the_plant_maximum() {
        let cases: [(f64, [f64; 2]); 2] = [(300.3, [100.1, 200.2]), (300.0, [150.0, 150.0])];
        for (plant_max_mw, maxima) in cases {
            let errors = plant_errors(plant_max_mw, units_with_maxima(30.0, &maxima));
            assert!(errors.is_empty(), "plant max {plant_max_mw}: {errors:?}");
        }
    }

    #[test]
    fn unit_capacity_rejects_a_shortfall_beyond_tolerance() {
        let errors = plant_errors(100.0, units_with_maxima(30.0, &[40.0, 59.0]));

        assert_eq!(
            errors,
            vec![ValidationError::InsufficientThermalUnitCapacity {
                thermal_id: EntityId(1),
                units_max_mw: 99.0,
                plant_max_mw: 100.0,
            }]
        );
    }

    #[test]
    fn unit_capacity_tolerance_scales_with_the_plant_maximum() {
        let cases: [(f64, &[f64], bool); 5] = [
            (1e6, &[5e5, 5e5 - 1e-4], true),
            (1e6, &[5e5, 5e5 - 1e-2], false),
            (0.5, &[0.5 - 1e-10], true),
            (0.5, &[0.5 - 1e-8], false),
            // 2e-9 is exactly twice 1e-9 in f64, so the shortfall equals the floored tolerance.
            (2e-9, &[1e-9], true),
        ];
        for (plant_max_mw, maxima, accepted) in cases {
            let errors = plant_errors(plant_max_mw, units_with_maxima(0.0, maxima));
            assert_eq!(
                errors.is_empty(),
                accepted,
                "plant max {plant_max_mw}, unit maxima {maxima:?}: {errors:?}"
            );
        }
    }

    #[test]
    fn unit_capacity_rejects_a_nan_plant_maximum() {
        let errors = plant_errors(f64::NAN, units_with_maxima(30.0, &[100.0]));

        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::InsufficientThermalUnitCapacity {
                    thermal_id: EntityId(1),
                    ..
                }]
            ),
            "{errors:?}"
        );
    }

    #[test]
    fn unit_capacity_rejects_an_infinite_plant_maximum() {
        for plant_max_mw in [f64::INFINITY, f64::NEG_INFINITY] {
            let errors = plant_errors(plant_max_mw, units_with_maxima(30.0, &[100.0]));

            assert!(
                matches!(
                    errors.as_slice(),
                    [ValidationError::InsufficientThermalUnitCapacity {
                        thermal_id: EntityId(1),
                        ..
                    }]
                ),
                "plant max {plant_max_mw}: {errors:?}"
            );
        }
    }

    #[test]
    fn unit_capacity_skips_plants_without_units_or_with_invalid_unit_bounds() {
        assert!(plant_errors(100.0, Vec::new()).is_empty());

        let errors = plant_errors(
            100.0,
            vec![make_thermal_unit(ThermalUnitSpec {
                id: 1,
                min_generation_mw: 0.0,
                max_generation_mw: f64::NAN,
                ..Default::default()
            })],
        );

        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::InvalidThermalUnit {
                    violation: ThermalUnitViolation::InvalidGenerationBounds { .. },
                    ..
                }]
            ),
            "{errors:?}"
        );
    }

    fn state(thermal_id: i32, unit_id: i32) -> ThermalUnitInitialState {
        make_thermal_unit_state(ThermalUnitStateSpec {
            thermal_id,
            unit_id,
            ..Default::default()
        })
    }

    fn invalid_state(
        thermal_id: i32,
        unit_id: i32,
        violation: ThermalUnitStateViolation,
    ) -> ValidationError {
        ValidationError::InvalidThermalUnitInitialState {
            thermal_id: EntityId(thermal_id),
            unit_id: EntityId(unit_id),
            violation,
        }
    }

    fn state_errors(
        states: &[ThermalUnitInitialState],
        thermals: &[Thermal],
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        validate_thermal_unit_states(states, thermals, &build_index(thermals), &mut errors);
        errors
    }

    #[test]
    fn duplicate_thermal_unit_states_are_reported_once_per_adjacent_pair() {
        let mut errors = Vec::new();
        check_duplicate_thermal_unit_states(
            &[
                state(1, 1),
                state(1, 1),
                state(1, 1),
                state(2, 3),
                state(2, 3),
            ],
            &mut errors,
        );

        assert_eq!(
            errors,
            vec![
                invalid_state(1, 1, ThermalUnitStateViolation::DuplicateRecord),
                invalid_state(1, 1, ThermalUnitStateViolation::DuplicateRecord),
                invalid_state(2, 3, ThermalUnitStateViolation::DuplicateRecord),
            ]
        );
    }

    #[test]
    fn thermal_unit_state_references_resolve_against_plant_units() {
        let thermals = [
            make_thermal(ThermalSpec {
                id: 1,
                bus_id: 1,
                units: units_with_maxima(30.0, &[100.0, 100.0]),
                ..Default::default()
            }),
            make_thermal(ThermalSpec {
                id: 2,
                bus_id: 1,
                ..Default::default()
            }),
        ];

        let errors = state_errors(
            &[state(1, 1), state(1, 3), state(2, 1), state(9, 1)],
            &thermals,
        );

        assert_eq!(
            errors,
            vec![
                invalid_state(1, 3, ThermalUnitStateViolation::UnknownUnit),
                invalid_state(2, 1, ThermalUnitStateViolation::UnknownUnit),
                invalid_state(9, 1, ThermalUnitStateViolation::UnknownThermal),
            ]
        );
    }

    #[test]
    fn thermal_unit_state_scalars_reject_negative_and_non_finite() {
        let thermals = [plant_without_capacity_floor(units_with_maxima(
            30.0,
            &[100.0],
        ))];
        let violations = |time_in_state_hours, previous_generation_mw| {
            let record = make_thermal_unit_state(ThermalUnitStateSpec {
                thermal_id: 1,
                unit_id: 1,
                time_in_state_hours,
                previous_generation_mw,
                ..Default::default()
            });
            state_errors(&[record], &thermals)
                .into_iter()
                .map(|error| match error {
                    ValidationError::InvalidThermalUnitInitialState { violation, .. } => violation,
                    other => panic!("expected InvalidThermalUnitInitialState, got {other:?}"),
                })
                .collect::<Vec<_>>()
        };

        assert!(matches!(
            violations(-1.0, 0.0).as_slice(),
            [ThermalUnitStateViolation::InvalidTimeInState {
                time_in_state_hours: -1.0
            }]
        ));
        assert!(matches!(
            violations(f64::INFINITY, 0.0).as_slice(),
            [ThermalUnitStateViolation::InvalidTimeInState { time_in_state_hours }]
                if time_in_state_hours.is_infinite()
        ));
        assert!(matches!(
            violations(0.0, f64::NAN).as_slice(),
            [ThermalUnitStateViolation::InvalidPreviousGeneration { previous_generation_mw }]
                if previous_generation_mw.is_nan()
        ));
        assert!(matches!(
            violations(0.0, -0.5).as_slice(),
            [ThermalUnitStateViolation::InvalidPreviousGeneration {
                previous_generation_mw: -0.5
            }]
        ));
    }

    fn ramped_units(count: i32) -> Vec<ThermalUnit> {
        (1..=count)
            .map(|id| ramped_unit(id, &[10.0, 20.0], &[20.0, 5.0]))
            .collect()
    }

    fn status_state(
        thermal_id: i32,
        unit_id: i32,
        status: ThermalUnitStatus,
        previous_generation_mw: f64,
    ) -> ThermalUnitInitialState {
        make_thermal_unit_state(ThermalUnitStateSpec {
            thermal_id,
            unit_id,
            status,
            previous_generation_mw,
            ..Default::default()
        })
    }

    #[test]
    fn thermal_unit_state_consistency_accepts_matching_statuses() {
        let thermals = [plant_without_capacity_floor(ramped_units(6))];

        let errors = state_errors(
            &[
                status_state(1, 1, ThermalUnitStatus::Off, 0.0),
                status_state(1, 2, ThermalUnitStatus::Off, -0.0),
                status_state(1, 3, ThermalUnitStatus::On, 20.0),
                status_state(1, 4, ThermalUnitStatus::On, 60.0),
                status_state(1, 5, ThermalUnitStatus::StartingUp, 10.0),
                status_state(1, 6, ThermalUnitStatus::ShuttingDown, 5.0),
            ],
            &thermals,
        );

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn thermal_unit_state_consistency_rejects_mismatched_statuses() {
        let mut units = ramped_units(6);
        units[4] = ramped_unit(5, &[], &[20.0, 5.0]);
        let thermals = [plant_without_capacity_floor(units)];

        let errors = state_errors(
            &[
                status_state(1, 1, ThermalUnitStatus::Off, 5.0),
                status_state(1, 2, ThermalUnitStatus::On, 10.0),
                status_state(1, 3, ThermalUnitStatus::On, 61.0),
                status_state(1, 4, ThermalUnitStatus::StartingUp, 20.0),
                status_state(1, 5, ThermalUnitStatus::StartingUp, 10.0),
                status_state(1, 6, ThermalUnitStatus::ShuttingDown, 7.0),
            ],
            &thermals,
        );

        assert_eq!(
            errors,
            vec![
                invalid_state(
                    1,
                    1,
                    ThermalUnitStateViolation::OffWithGeneration {
                        previous_generation_mw: 5.0
                    }
                ),
                invalid_state(
                    1,
                    2,
                    ThermalUnitStateViolation::OnOutsideBounds {
                        previous_generation_mw: 10.0,
                        min_generation_mw: 20.0,
                        max_generation_mw: 60.0,
                    }
                ),
                invalid_state(
                    1,
                    3,
                    ThermalUnitStateViolation::OnOutsideBounds {
                        previous_generation_mw: 61.0,
                        min_generation_mw: 20.0,
                        max_generation_mw: 60.0,
                    }
                ),
                invalid_state(
                    1,
                    4,
                    ThermalUnitStateViolation::GenerationNotOnTrajectory {
                        transition: ThermalUnitTransition::StartingUp,
                        previous_generation_mw: 20.0,
                    }
                ),
                invalid_state(
                    1,
                    5,
                    ThermalUnitStateViolation::GenerationNotOnTrajectory {
                        transition: ThermalUnitTransition::StartingUp,
                        previous_generation_mw: 10.0,
                    }
                ),
                invalid_state(
                    1,
                    6,
                    ThermalUnitStateViolation::GenerationNotOnTrajectory {
                        transition: ThermalUnitTransition::ShuttingDown,
                        previous_generation_mw: 7.0,
                    }
                ),
            ]
        );
    }

    #[test]
    fn thermal_unit_state_consistency_is_skipped_when_inputs_fail_their_own_rules() {
        let thermals = [
            plant_without_capacity_floor(ramped_units(3)),
            make_thermal(ThermalSpec {
                id: 2,
                bus_id: 1,
                max_generation_mw: 0.0,
                units: vec![make_thermal_unit(ThermalUnitSpec {
                    id: 1,
                    min_generation_mw: 50.0,
                    max_generation_mw: 40.0,
                    ..Default::default()
                })],
                ..Default::default()
            }),
        ];
        let mut negative_time = status_state(1, 3, ThermalUnitStatus::Off, 5.0);
        negative_time.time_in_state_hours = -1.0;

        let errors = state_errors(
            &[
                status_state(1, 1, ThermalUnitStatus::On, -5.0),
                status_state(1, 2, ThermalUnitStatus::Off, f64::INFINITY),
                negative_time,
                status_state(1, 9, ThermalUnitStatus::StartingUp, 3.0),
                status_state(2, 1, ThermalUnitStatus::On, 45.0),
            ],
            &thermals,
        );

        assert_eq!(
            errors,
            vec![
                invalid_state(
                    1,
                    1,
                    ThermalUnitStateViolation::InvalidPreviousGeneration {
                        previous_generation_mw: -5.0
                    }
                ),
                invalid_state(
                    1,
                    2,
                    ThermalUnitStateViolation::InvalidPreviousGeneration {
                        previous_generation_mw: f64::INFINITY
                    }
                ),
                invalid_state(
                    1,
                    3,
                    ThermalUnitStateViolation::InvalidTimeInState {
                        time_in_state_hours: -1.0
                    }
                ),
                invalid_state(
                    1,
                    3,
                    ThermalUnitStateViolation::OffWithGeneration {
                        previous_generation_mw: 5.0
                    }
                ),
                invalid_state(1, 9, ThermalUnitStateViolation::UnknownUnit),
            ]
        );
    }

    fn hydro_with_groups(id: i32, unit_groups: Vec<HydroUnitGroup>) -> Hydro {
        let mut hydro = make_hydro(HydroSpec {
            id,
            bus_id: 1,
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        });
        hydro.unit_groups = unit_groups;
        hydro
    }

    fn default_group(id: i32, unit_ids: &[i32]) -> HydroUnitGroup {
        make_unit_group(UnitGroupSpec {
            id,
            bus_id: 1,
            units: unit_ids
                .iter()
                .map(|&unit_id| {
                    make_hydro_unit(HydroUnitSpec {
                        id: unit_id,
                        ..Default::default()
                    })
                })
                .collect(),
            ..Default::default()
        })
    }

    fn hydro_state(hydro_id: i32, group_id: i32, unit_id: i32) -> HydroUnitInitialState {
        make_hydro_unit_state(HydroUnitStateSpec {
            hydro_id,
            group_id,
            unit_id,
            ..Default::default()
        })
    }

    fn invalid_hydro_state(
        hydro_id: i32,
        group_id: i32,
        unit_id: i32,
        violation: HydroUnitStateViolation,
    ) -> ValidationError {
        ValidationError::InvalidHydroUnitInitialState {
            hydro_id: EntityId(hydro_id),
            group_id: EntityId(group_id),
            unit_id: EntityId(unit_id),
            violation,
        }
    }

    #[test]
    fn duplicate_hydro_unit_group_ids_are_reported_in_key_order() {
        let hydros = [
            hydro_with_groups(2, vec![default_group(4, &[1]), default_group(4, &[1])]),
            hydro_with_groups(2, vec![default_group(3, &[1]), default_group(3, &[1])]),
            hydro_with_groups(
                1,
                vec![
                    default_group(5, &[1]),
                    default_group(5, &[1]),
                    default_group(5, &[1]),
                ],
            ),
        ];
        let duplicate_group = |hydro_id, group_id| ValidationError::InvalidHydroUnitGroup {
            hydro_id: EntityId(hydro_id),
            group_id: EntityId(group_id),
            violation: HydroUnitGroupViolation::DuplicateId,
        };
        let mut errors = Vec::new();

        check_duplicate_hydro_unit_group_ids(&hydros, &mut errors);

        assert_eq!(
            errors,
            vec![
                duplicate_group(1, 5),
                duplicate_group(1, 5),
                duplicate_group(2, 3),
                duplicate_group(2, 4),
            ]
        );
    }

    #[test]
    fn hydro_unit_group_ids_may_repeat_across_plants() {
        let hydros = [
            hydro_with_groups(1, vec![default_group(0, &[1]), default_group(1, &[1])]),
            hydro_with_groups(2, vec![default_group(0, &[1]), default_group(1, &[1])]),
        ];
        let mut errors = Vec::new();

        check_duplicate_hydro_unit_group_ids(&hydros, &mut errors);

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn duplicate_hydro_unit_ids_are_reported_in_key_order() {
        let hydros = [
            hydro_with_groups(2, vec![default_group(0, &[4, 4])]),
            hydro_with_groups(2, vec![default_group(0, &[3, 3])]),
            hydro_with_groups(
                1,
                vec![default_group(0, &[6]), default_group(1, &[5, 5, 5])],
            ),
        ];
        let duplicate_unit = |hydro_id, group_id, unit_id| ValidationError::InvalidHydroUnit {
            hydro_id: EntityId(hydro_id),
            group_id: EntityId(group_id),
            unit_id: EntityId(unit_id),
            violation: HydroUnitViolation::DuplicateId,
        };
        let mut errors = Vec::new();

        check_duplicate_hydro_unit_ids(&hydros, &mut errors);

        assert_eq!(
            errors,
            vec![
                duplicate_unit(1, 1, 5),
                duplicate_unit(1, 1, 5),
                duplicate_unit(2, 0, 3),
                duplicate_unit(2, 0, 4),
            ]
        );
    }

    #[test]
    fn hydro_unit_ids_may_repeat_across_groups() {
        let hydros = [
            hydro_with_groups(
                1,
                vec![default_group(0, &[1, 2]), default_group(1, &[1, 2])],
            ),
            hydro_with_groups(2, vec![default_group(0, &[1, 2])]),
        ];
        let mut errors = Vec::new();

        check_duplicate_hydro_unit_ids(&hydros, &mut errors);

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn duplicate_hydro_unit_states_are_reported_once_per_adjacent_pair() {
        let states = [
            hydro_state(1, 0, 1),
            hydro_state(1, 0, 1),
            hydro_state(1, 0, 1),
            hydro_state(1, 0, 2),
            hydro_state(1, 1, 2),
            hydro_state(2, 1, 2),
            hydro_state(2, 1, 2),
        ];
        let mut errors = Vec::new();

        check_duplicate_hydro_unit_states(&states, &mut errors);

        assert_eq!(
            errors,
            vec![
                invalid_hydro_state(1, 0, 1, HydroUnitStateViolation::DuplicateRecord),
                invalid_hydro_state(1, 0, 1, HydroUnitStateViolation::DuplicateRecord),
                invalid_hydro_state(2, 1, 2, HydroUnitStateViolation::DuplicateRecord),
            ]
        );
    }

    fn group_without_capacity_floor(id: i32, units: Vec<HydroUnit>) -> HydroUnitGroup {
        make_unit_group(UnitGroupSpec {
            id,
            bus_id: 1,
            max_generation_mw: 0.0,
            max_turbined_m3s: 0.0,
            units,
            ..Default::default()
        })
    }

    fn hydro_unit_errors(units: Vec<HydroUnit>) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        validate_hydro_units(
            &[hydro_with_groups(
                1,
                vec![group_without_capacity_floor(0, units)],
            )],
            &mut errors,
        );
        errors
    }

    fn single_unit_violation(errors: &[ValidationError]) -> HydroUnitViolation {
        match errors {
            [
                ValidationError::InvalidHydroUnit {
                    hydro_id: EntityId(1),
                    group_id: EntityId(0),
                    unit_id: EntityId(1),
                    violation,
                },
            ] => *violation,
            other => panic!("expected one InvalidHydroUnit on (1, 0, 1), got {other:?}"),
        }
    }

    const REJECTED_BOUNDS: [(f64, f64); 9] = [
        (f64::NAN, 10.0),
        (f64::INFINITY, 10.0),
        (f64::NEG_INFINITY, 10.0),
        (-1.0, 10.0),
        (0.0, f64::NAN),
        (0.0, f64::INFINITY),
        (0.0, f64::NEG_INFINITY),
        (0.0, -1.0),
        (50.0, 40.0),
    ];

    #[test]
    fn hydro_unit_generation_bounds_reject_negative_inverted_and_non_finite() {
        for (min, max) in REJECTED_BOUNDS {
            let violation =
                single_unit_violation(&hydro_unit_errors(vec![make_hydro_unit(HydroUnitSpec {
                    id: 1,
                    min_generation_mw: min,
                    max_generation_mw: max,
                    ..Default::default()
                })]));

            assert!(
                matches!(
                    violation,
                    HydroUnitViolation::InvalidGenerationBounds {
                        min_generation_mw,
                        max_generation_mw,
                    } if f64_bits_eq(min_generation_mw, min) && f64_bits_eq(max_generation_mw, max)
                ),
                "bounds ({min}, {max}): {violation:?}"
            );
        }
    }

    #[test]
    fn hydro_unit_turbined_bounds_reject_negative_inverted_and_non_finite() {
        for (min, max) in REJECTED_BOUNDS {
            let violation =
                single_unit_violation(&hydro_unit_errors(vec![make_hydro_unit(HydroUnitSpec {
                    id: 1,
                    min_turbined_m3s: min,
                    max_turbined_m3s: max,
                    ..Default::default()
                })]));

            assert!(
                matches!(
                    violation,
                    HydroUnitViolation::InvalidTurbinedBounds {
                        min_turbined_m3s,
                        max_turbined_m3s,
                    } if f64_bits_eq(min_turbined_m3s, min) && f64_bits_eq(max_turbined_m3s, max)
                ),
                "bounds ({min}, {max}): {violation:?}"
            );
        }
    }

    #[test]
    fn hydro_unit_durations_reject_negative_and_non_finite() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let up =
                single_unit_violation(&hydro_unit_errors(vec![make_hydro_unit(HydroUnitSpec {
                    id: 1,
                    min_up_time_hours: value,
                    ..Default::default()
                })]));
            let down =
                single_unit_violation(&hydro_unit_errors(vec![make_hydro_unit(HydroUnitSpec {
                    id: 1,
                    min_down_time_hours: value,
                    ..Default::default()
                })]));

            assert!(
                matches!(
                    up,
                    HydroUnitViolation::InvalidMinUpTime { min_up_time_hours }
                        if f64_bits_eq(min_up_time_hours, value)
                ),
                "min up time {value}: {up:?}"
            );
            assert!(
                matches!(
                    down,
                    HydroUnitViolation::InvalidMinDownTime { min_down_time_hours }
                        if f64_bits_eq(min_down_time_hours, value)
                ),
                "min down time {value}: {down:?}"
            );
        }
    }

    #[test]
    fn hydro_unit_scalar_rules_accept_equal_and_zero_bounds_and_zero_durations() {
        let errors = hydro_unit_errors(vec![
            make_hydro_unit(HydroUnitSpec {
                id: 1,
                min_generation_mw: 5.0,
                max_generation_mw: 5.0,
                min_turbined_m3s: 3.0,
                max_turbined_m3s: 3.0,
                ..Default::default()
            }),
            make_hydro_unit(HydroUnitSpec {
                id: 2,
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 0.0,
                ..Default::default()
            }),
            make_hydro_unit(HydroUnitSpec {
                id: 3,
                min_generation_mw: -0.0,
                max_generation_mw: 0.0,
                min_turbined_m3s: -0.0,
                max_turbined_m3s: 0.0,
                min_up_time_hours: -0.0,
                min_down_time_hours: 0.0,
                ..Default::default()
            }),
        ]);

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    #[test]
    fn hydro_unit_violations_are_reported_in_rule_order() {
        let errors = hydro_unit_errors(vec![
            make_hydro_unit(HydroUnitSpec {
                id: 1,
                min_generation_mw: 50.0,
                max_generation_mw: 40.0,
                min_turbined_m3s: -1.0,
                max_turbined_m3s: 10.0,
                min_up_time_hours: -2.0,
                min_down_time_hours: f64::INFINITY,
                ..Default::default()
            }),
            make_hydro_unit(HydroUnitSpec {
                id: 2,
                ..Default::default()
            }),
        ]);
        let invalid = |violation| ValidationError::InvalidHydroUnit {
            hydro_id: EntityId(1),
            group_id: EntityId(0),
            unit_id: EntityId(1),
            violation,
        };

        assert_eq!(
            errors,
            vec![
                invalid(HydroUnitViolation::InvalidGenerationBounds {
                    min_generation_mw: 50.0,
                    max_generation_mw: 40.0,
                }),
                invalid(HydroUnitViolation::InvalidTurbinedBounds {
                    min_turbined_m3s: -1.0,
                    max_turbined_m3s: 10.0,
                }),
                invalid(HydroUnitViolation::InvalidMinUpTime {
                    min_up_time_hours: -2.0,
                }),
                invalid(HydroUnitViolation::InvalidMinDownTime {
                    min_down_time_hours: f64::INFINITY,
                }),
            ]
        );
    }

    fn sized_units(maxima: &[(f64, f64)]) -> Vec<HydroUnit> {
        maxima
            .iter()
            .zip(1..)
            .map(|(&(max_generation_mw, max_turbined_m3s), id)| {
                make_hydro_unit(HydroUnitSpec {
                    id,
                    max_generation_mw,
                    max_turbined_m3s,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn group_capacity_errors(
        max_generation_mw: f64,
        max_turbined_m3s: f64,
        units: Vec<HydroUnit>,
    ) -> Vec<ValidationError> {
        let group = make_unit_group(UnitGroupSpec {
            id: 0,
            bus_id: 1,
            max_generation_mw,
            max_turbined_m3s,
            units,
            ..Default::default()
        });
        let mut errors = Vec::new();
        validate_hydro_units(&[hydro_with_groups(1, vec![group])], &mut errors);
        errors
    }

    fn short_group(violation: HydroUnitGroupViolation) -> ValidationError {
        ValidationError::InvalidHydroUnitGroup {
            hydro_id: EntityId(1),
            group_id: EntityId(0),
            violation,
        }
    }

    #[test]
    fn hydro_unit_capacity_accepts_a_sum_within_tolerance_of_the_group_maximum() {
        let accepts = |group_max_mw: f64, group_max_m3s: f64, maxima: &[(f64, f64)]| {
            let errors = group_capacity_errors(group_max_mw, group_max_m3s, sized_units(maxima));
            assert!(
                errors.is_empty(),
                "group max ({group_max_mw}, {group_max_m3s}): {errors:?}"
            );
        };

        accepts(300.3, 300.3, &[(100.1, 100.1), (200.2, 200.2)]);
        accepts(300.0, 150.0, &[(150.0, 75.0), (150.0, 75.0)]);
        accepts(-5.0, -5.0, &[(1.0, 1.0)]);
    }

    #[test]
    fn hydro_unit_capacity_rejects_a_generation_shortfall() {
        let errors = group_capacity_errors(100.0, 10.0, sized_units(&[(40.0, 5.0), (59.0, 5.0)]));

        assert_eq!(
            errors,
            vec![short_group(
                HydroUnitGroupViolation::InsufficientGenerationCapacity {
                    units_max_mw: 99.0,
                    group_max_mw: 100.0,
                }
            )]
        );
    }

    #[test]
    fn hydro_unit_capacity_rejects_a_turbined_shortfall() {
        let errors = group_capacity_errors(10.0, 100.0, sized_units(&[(5.0, 40.0), (5.0, 59.0)]));

        assert_eq!(
            errors,
            vec![short_group(
                HydroUnitGroupViolation::InsufficientTurbinedCapacity {
                    units_max_m3s: 99.0,
                    group_max_m3s: 100.0,
                }
            )]
        );
    }

    #[test]
    fn hydro_unit_capacity_reports_both_quantities_independently() {
        let errors =
            group_capacity_errors(100.0, 100.0, sized_units(&[(40.0, 40.0), (59.0, 59.0)]));

        assert_eq!(
            errors,
            vec![
                short_group(HydroUnitGroupViolation::InsufficientGenerationCapacity {
                    units_max_mw: 99.0,
                    group_max_mw: 100.0,
                }),
                short_group(HydroUnitGroupViolation::InsufficientTurbinedCapacity {
                    units_max_m3s: 99.0,
                    group_max_m3s: 100.0,
                }),
            ]
        );
    }

    #[test]
    fn hydro_unit_capacity_rejects_a_non_finite_group_maximum() {
        for group_max in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let generation = group_capacity_errors(group_max, 1.0, sized_units(&[(1.0, 1.0)]));
            let turbined = group_capacity_errors(1.0, group_max, sized_units(&[(1.0, 1.0)]));

            assert!(
                matches!(
                    generation.as_slice(),
                    [ValidationError::InvalidHydroUnitGroup {
                        hydro_id: EntityId(1),
                        group_id: EntityId(0),
                        violation: HydroUnitGroupViolation::InsufficientGenerationCapacity {
                            units_max_mw,
                            group_max_mw,
                        },
                    }] if f64_bits_eq(*units_max_mw, 1.0) && f64_bits_eq(*group_max_mw, group_max)
                ),
                "generation group max {group_max}: {generation:?}"
            );
            assert!(
                matches!(
                    turbined.as_slice(),
                    [ValidationError::InvalidHydroUnitGroup {
                        hydro_id: EntityId(1),
                        group_id: EntityId(0),
                        violation: HydroUnitGroupViolation::InsufficientTurbinedCapacity {
                            units_max_m3s,
                            group_max_m3s,
                        },
                    }] if f64_bits_eq(*units_max_m3s, 1.0) && f64_bits_eq(*group_max_m3s, group_max)
                ),
                "turbined group max {group_max}: {turbined:?}"
            );
        }
    }

    #[test]
    fn hydro_unit_capacity_skips_both_quantities_when_any_unit_bound_is_invalid() {
        let invalid_unit_2 = |violation| ValidationError::InvalidHydroUnit {
            hydro_id: EntityId(1),
            group_id: EntityId(0),
            unit_id: EntityId(2),
            violation,
        };
        let units_with = |unit_2: HydroUnitSpec| {
            let mut units = sized_units(&[(1.0, 1.0)]);
            units.push(make_hydro_unit(unit_2));
            units
        };

        let generation = group_capacity_errors(
            100.0,
            100.0,
            units_with(HydroUnitSpec {
                id: 2,
                min_generation_mw: 50.0,
                max_generation_mw: 40.0,
                ..Default::default()
            }),
        );
        let turbined = group_capacity_errors(
            100.0,
            100.0,
            units_with(HydroUnitSpec {
                id: 2,
                min_turbined_m3s: 50.0,
                max_turbined_m3s: 40.0,
                ..Default::default()
            }),
        );

        assert_eq!(
            generation,
            vec![invalid_unit_2(
                HydroUnitViolation::InvalidGenerationBounds {
                    min_generation_mw: 50.0,
                    max_generation_mw: 40.0,
                }
            )]
        );
        assert_eq!(
            turbined,
            vec![invalid_unit_2(HydroUnitViolation::InvalidTurbinedBounds {
                min_turbined_m3s: 50.0,
                max_turbined_m3s: 40.0,
            })]
        );
    }

    #[test]
    fn hydro_unit_capacity_skips_groups_without_units() {
        assert!(group_capacity_errors(100.0, 100.0, Vec::new()).is_empty());
    }

    fn boxed_unit(id: i32, generation: (f64, f64), turbined: (f64, f64)) -> HydroUnit {
        make_hydro_unit(HydroUnitSpec {
            id,
            min_generation_mw: generation.0,
            max_generation_mw: generation.1,
            min_turbined_m3s: turbined.0,
            max_turbined_m3s: turbined.1,
            ..Default::default()
        })
    }

    fn hydro_status_state(
        (hydro_id, group_id, unit_id): (i32, i32, i32),
        status: HydroUnitStatus,
        previous_generation_mw: f64,
        previous_turbined_m3s: f64,
    ) -> HydroUnitInitialState {
        make_hydro_unit_state(HydroUnitStateSpec {
            hydro_id,
            group_id,
            unit_id,
            status,
            previous_generation_mw,
            previous_turbined_m3s,
            ..Default::default()
        })
    }

    fn hydro_state_errors(
        states: &[HydroUnitInitialState],
        hydros: &[Hydro],
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        validate_hydro_unit_states(states, hydros, &build_index(hydros), &mut errors);
        errors
    }

    #[test]
    fn hydro_unit_state_references_resolve_hydro_then_group_then_unit() {
        let hydros = [
            hydro_with_groups(1, vec![default_group(0, &[1, 2]), default_group(3, &[1])]),
            hydro_with_groups(2, vec![default_group(0, &[5])]),
        ];

        let errors = hydro_state_errors(
            &[
                hydro_state(1, 0, 1),
                hydro_state(1, 0, 4),
                hydro_state(1, 1, 1),
                hydro_state(1, 3, 1),
                hydro_state(2, 0, 1),
                hydro_state(2, 3, 1),
                hydro_state(9, 0, 1),
            ],
            &hydros,
        );

        assert_eq!(
            errors,
            vec![
                invalid_hydro_state(1, 0, 4, HydroUnitStateViolation::UnknownUnit),
                invalid_hydro_state(1, 1, 1, HydroUnitStateViolation::UnknownGroup),
                invalid_hydro_state(2, 0, 1, HydroUnitStateViolation::UnknownUnit),
                invalid_hydro_state(2, 3, 1, HydroUnitStateViolation::UnknownGroup),
                invalid_hydro_state(9, 0, 1, HydroUnitStateViolation::UnknownHydro),
            ]
        );
    }

    #[test]
    fn hydro_unit_state_scalars_reject_negative_and_non_finite() {
        let hydros = [hydro_with_groups(
            1,
            vec![group_without_capacity_floor(
                0,
                vec![boxed_unit(1, (0.0, 10.0), (0.0, 10.0))],
            )],
        )];
        let single_violation = |record| match hydro_state_errors(&[record], &hydros).as_slice() {
            [
                ValidationError::InvalidHydroUnitInitialState {
                    hydro_id: EntityId(1),
                    group_id: EntityId(0),
                    unit_id: EntityId(1),
                    violation,
                },
            ] => *violation,
            other => {
                panic!("expected one InvalidHydroUnitInitialState on (1, 0, 1), got {other:?}")
            }
        };

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let time = single_violation(HydroUnitInitialState {
                time_in_state_hours: value,
                ..hydro_state(1, 0, 1)
            });
            let generation = single_violation(hydro_status_state(
                (1, 0, 1),
                HydroUnitStatus::Off,
                value,
                0.0,
            ));
            let turbined = single_violation(hydro_status_state(
                (1, 0, 1),
                HydroUnitStatus::Off,
                0.0,
                value,
            ));

            assert!(
                matches!(
                    time,
                    HydroUnitStateViolation::InvalidTimeInState { time_in_state_hours }
                        if f64_bits_eq(time_in_state_hours, value)
                ),
                "time {value}: {time:?}"
            );
            assert!(
                matches!(
                    generation,
                    HydroUnitStateViolation::InvalidPreviousGeneration { previous_generation_mw }
                        if f64_bits_eq(previous_generation_mw, value)
                ),
                "generation {value}: {generation:?}"
            );
            assert!(
                matches!(
                    turbined,
                    HydroUnitStateViolation::InvalidPreviousTurbined { previous_turbined_m3s }
                        if f64_bits_eq(previous_turbined_m3s, value)
                ),
                "turbined {value}: {turbined:?}"
            );
        }
    }

    #[test]
    fn hydro_unit_state_off_requires_zero_previous_values() {
        let units = (1..=5)
            .map(|id| boxed_unit(id, (0.0, 10.0), (0.0, 10.0)))
            .collect();
        let hydros = [hydro_with_groups(
            1,
            vec![group_without_capacity_floor(0, units)],
        )];
        let off = |unit_id, generation, turbined| {
            hydro_status_state((1, 0, unit_id), HydroUnitStatus::Off, generation, turbined)
        };

        let errors = hydro_state_errors(
            &[
                off(1, 0.0, 0.0),
                off(2, -0.0, -0.0),
                off(3, 3.0, 0.0),
                off(4, 0.0, 2.5),
                off(5, 3.0, 2.5),
            ],
            &hydros,
        );

        assert_eq!(
            errors,
            vec![
                invalid_hydro_state(
                    1,
                    0,
                    3,
                    HydroUnitStateViolation::OffWithGeneration {
                        previous_generation_mw: 3.0
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    4,
                    HydroUnitStateViolation::OffWithTurbinedFlow {
                        previous_turbined_m3s: 2.5
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    5,
                    HydroUnitStateViolation::OffWithGeneration {
                        previous_generation_mw: 3.0
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    5,
                    HydroUnitStateViolation::OffWithTurbinedFlow {
                        previous_turbined_m3s: 2.5
                    }
                ),
            ]
        );
    }

    #[test]
    fn hydro_unit_state_on_requires_previous_values_inside_unit_bounds() {
        let units = (1..=5)
            .map(|id| boxed_unit(id, (20.0, 60.0), (10.0, 30.0)))
            .collect();
        let hydros = [hydro_with_groups(
            1,
            vec![group_without_capacity_floor(0, units)],
        )];
        let on = |unit_id, generation, turbined| {
            hydro_status_state((1, 0, unit_id), HydroUnitStatus::On, generation, turbined)
        };

        let errors = hydro_state_errors(
            &[
                on(1, 20.0, 10.0),
                on(2, 60.0, 30.0),
                on(3, 19.0, 20.0),
                on(4, 40.0, 31.0),
                on(5, 61.0, 9.0),
            ],
            &hydros,
        );

        assert_eq!(
            errors,
            vec![
                invalid_hydro_state(
                    1,
                    0,
                    3,
                    HydroUnitStateViolation::OnGenerationOutsideBounds {
                        previous_generation_mw: 19.0,
                        min_generation_mw: 20.0,
                        max_generation_mw: 60.0,
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    4,
                    HydroUnitStateViolation::OnTurbinedOutsideBounds {
                        previous_turbined_m3s: 31.0,
                        min_turbined_m3s: 10.0,
                        max_turbined_m3s: 30.0,
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    5,
                    HydroUnitStateViolation::OnGenerationOutsideBounds {
                        previous_generation_mw: 61.0,
                        min_generation_mw: 20.0,
                        max_generation_mw: 60.0,
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    5,
                    HydroUnitStateViolation::OnTurbinedOutsideBounds {
                        previous_turbined_m3s: 9.0,
                        min_turbined_m3s: 10.0,
                        max_turbined_m3s: 30.0,
                    }
                ),
            ]
        );
    }

    #[test]
    fn hydro_unit_state_gates_each_quantity_independently() {
        let hydros = [hydro_with_groups(
            1,
            vec![group_without_capacity_floor(
                0,
                vec![
                    boxed_unit(1, (50.0, 40.0), (0.0, 10.0)),
                    boxed_unit(2, (0.0, 10.0), (50.0, 40.0)),
                    boxed_unit(3, (0.0, 10.0), (0.0, 10.0)),
                    boxed_unit(4, (0.0, 10.0), (0.0, 10.0)),
                ],
            )],
        )];

        let errors = hydro_state_errors(
            &[
                hydro_status_state((1, 0, 1), HydroUnitStatus::On, 45.0, 20.0),
                hydro_status_state((1, 0, 2), HydroUnitStatus::On, 20.0, 45.0),
                hydro_status_state((1, 0, 3), HydroUnitStatus::Off, -1.0, 5.0),
                hydro_status_state((1, 0, 4), HydroUnitStatus::Off, 5.0, f64::INFINITY),
                hydro_status_state((1, 0, 9), HydroUnitStatus::Off, 7.0, 7.0),
            ],
            &hydros,
        );

        assert_eq!(
            errors,
            vec![
                invalid_hydro_state(
                    1,
                    0,
                    1,
                    HydroUnitStateViolation::OnTurbinedOutsideBounds {
                        previous_turbined_m3s: 20.0,
                        min_turbined_m3s: 0.0,
                        max_turbined_m3s: 10.0,
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    2,
                    HydroUnitStateViolation::OnGenerationOutsideBounds {
                        previous_generation_mw: 20.0,
                        min_generation_mw: 0.0,
                        max_generation_mw: 10.0,
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    3,
                    HydroUnitStateViolation::InvalidPreviousGeneration {
                        previous_generation_mw: -1.0
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    3,
                    HydroUnitStateViolation::OffWithTurbinedFlow {
                        previous_turbined_m3s: 5.0
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    4,
                    HydroUnitStateViolation::InvalidPreviousTurbined {
                        previous_turbined_m3s: f64::INFINITY
                    }
                ),
                invalid_hydro_state(
                    1,
                    0,
                    4,
                    HydroUnitStateViolation::OffWithGeneration {
                        previous_generation_mw: 5.0
                    }
                ),
                invalid_hydro_state(1, 0, 9, HydroUnitStateViolation::UnknownUnit),
            ]
        );
    }
}
