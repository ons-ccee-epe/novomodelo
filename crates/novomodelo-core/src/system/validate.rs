//! Construction-time validation helpers for [`SystemBuilder::build()`](crate::SystemBuilder::build).

use std::collections::HashMap;

use crate::{
    Bus, EnergyContract, EntityId, Hydro, Line, NeumaierAccumulator, NonControllableSource,
    PumpingStation, Stage, Thermal, ThermalUnit, ThermalUnitInitialState,
    ThermalUnitStateViolation, ThermalUnitStatus, ThermalUnitTransition, ThermalUnitViolation,
    ValidationError,
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

fn generation_bounds_are_valid(unit: &ThermalUnit) -> bool {
    let (min, max) = (unit.min_generation_mw, unit.max_generation_mw);
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

/// The tolerance is scaled by `max(1, plant max)`, so summation rounding never rejects a plant
/// its units cover.
const UNIT_CAPACITY_RELATIVE_TOLERANCE: f64 = 1e-9;

fn validate_unit_capacity(thermal: &Thermal, errors: &mut Vec<ValidationError>) {
    if thermal.units.is_empty() || !thermal.units.iter().all(generation_bounds_are_valid) {
        return;
    }
    let mut sum = NeumaierAccumulator::new();
    for unit in &thermal.units {
        sum.add(unit.max_generation_mw);
    }
    let units_max_mw = sum.value();
    let plant_max_mw = thermal.max_generation_mw;
    let tolerance = UNIT_CAPACITY_RELATIVE_TOLERANCE * plant_max_mw.max(1.0);
    // An infinite plant maximum makes the tolerance infinite, which would accept any sum.
    let covered = plant_max_mw.is_finite() && plant_max_mw - units_max_mw <= tolerance;
    if !covered {
        errors.push(ValidationError::InsufficientThermalUnitCapacity {
            thermal_id: thermal.id,
            units_max_mw,
            plant_max_mw,
        });
    }
}

/// Validate every thermal unit, appending all violations to `errors` (no
/// short-circuiting on the first).
///
/// Per unit, in order: generation bounds, minimum up time, minimum down time,
/// startup ramp, shutdown ramp. The ramp rules read `min_generation_mw` and run
/// only when the generation bounds are valid.
///
/// Per plant, after its units: the units' summed maximum must cover the plant
/// maximum. Plants without units, or with a unit whose generation bounds fail, are skipped.
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
            if !generation_bounds_are_valid(unit) {
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
            if generation_bounds_are_valid(unit) {
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

/// Validate every thermal unit initial state, appending all violations to `errors` (no
/// short-circuiting on the first).
///
/// Per record, in slice order: the plant and unit it names, `time_in_state_hours`,
/// `previous_generation_mw`, then the status against `previous_generation_mw`. The scalar rules
/// do not depend on the reference resolving; the status rule runs only when the unit resolves,
/// `previous_generation_mw` passed its rule, and the unit's generation bounds are valid.
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
            && generation_bounds_are_valid(unit)
            && let Some(violation) = status_generation_violation(state, unit)
        {
            reject(violation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_index, check_duplicate_thermal_unit_ids, check_duplicate_thermal_unit_states,
        validate_thermal_unit_states, validate_thermal_units,
    };
    use crate::test_support::{
        ThermalSpec, ThermalUnitSpec, ThermalUnitStateSpec, make_thermal, make_thermal_unit,
        make_thermal_unit_state,
    };
    use crate::{
        EntityId, Thermal, ThermalUnit, ThermalUnitInitialState, ThermalUnitStateViolation,
        ThermalUnitStatus, ThermalUnitTransition, ThermalUnitViolation, ValidationError,
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
}
