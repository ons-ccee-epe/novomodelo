use novomodelo_core::{EntityId, Line, VariableRef};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use super::parser::{Arg, VariableCall};
use crate::LoadError;

// ── Line bus-pair addressing ────────────────────────────────────────────────

/// Direction of a `(source_bus, target_bus)` pair relative to a line's declared
/// source→target orientation. `Reversed` folds `-1.0` into the resolved term's scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinePairOrientation {
    Forward,
    Reversed,
}

/// Resolves a `(source_bus, target_bus)` pair to the `line_exchange` line it addresses
/// and the sign that pair takes relative to the line's declared direction.
///
/// Built from `&[Line]` via [`build_line_bus_pair_index`]. The default is empty (no pair
/// resolves) — the form a direct
/// [`parse_generic_constraints`](crate::constraints::parse_generic_constraints) caller with
/// no line topology passes.
#[derive(Debug, Clone, Default)]
pub struct LineBusPairIndex {
    by_pair: HashMap<(EntityId, EntityId), (EntityId, LinePairOrientation)>,
}

impl LineBusPairIndex {
    pub(crate) fn resolve(
        &self,
        source_bus: EntityId,
        target_bus: EntityId,
    ) -> Option<(EntityId, LinePairOrientation)> {
        self.by_pair.get(&(source_bus, target_bus)).copied()
    }
}

/// Build the `(source_bus, target_bus) → (line_id, orientation)` index from the loaded
/// lines. Each line contributes its declared pair (`Forward`) and the reversed pair
/// (`Reversed`), so an author may address a line from either endpoint.
///
/// # Errors
///
/// [`LoadError::SchemaError`] when two distinct lines share one unordered bus pair: the
/// pair no longer identifies a single line, so it is rejected loudly (pointing the author
/// at a named-expression sum) rather than resolved to an arbitrary line.
pub fn build_line_bus_pair_index(lines: &[Line]) -> Result<LineBusPairIndex, LoadError> {
    let mut by_pair: HashMap<(EntityId, EntityId), (EntityId, LinePairOrientation)> =
        HashMap::with_capacity(lines.len().saturating_mul(2));
    for line in lines {
        insert_line_pair(
            &mut by_pair,
            (line.source_bus_id, line.target_bus_id),
            line.id,
            LinePairOrientation::Forward,
        )?;
        insert_line_pair(
            &mut by_pair,
            (line.target_bus_id, line.source_bus_id),
            line.id,
            LinePairOrientation::Reversed,
        )?;
    }
    Ok(LineBusPairIndex { by_pair })
}

/// Insert one directed pair, rejecting a key already claimed by a different line. A key
/// re-inserted by the same line (a degenerate `source == target` self-loop) keeps its
/// first entry rather than erroring.
fn insert_line_pair(
    by_pair: &mut HashMap<(EntityId, EntityId), (EntityId, LinePairOrientation)>,
    key: (EntityId, EntityId),
    line_id: EntityId,
    orientation: LinePairOrientation,
) -> Result<(), LoadError> {
    match by_pair.get(&key) {
        Some((existing, _)) if *existing != line_id => Err(duplicate_line_pair_error(key)),
        Some(_) => Ok(()),
        None => {
            by_pair.insert(key, (line_id, orientation));
            Ok(())
        }
    }
}

/// Order-invariant duplicate-pair error: names the buses sorted so the message is
/// identical regardless of which line or direction triggered the collision.
fn duplicate_line_pair_error(key: (EntityId, EntityId)) -> LoadError {
    let (a, b) = if key.0.0 <= key.1.0 {
        (key.0, key.1)
    } else {
        (key.1, key.0)
    };
    LoadError::SchemaError {
        path: PathBuf::from("system/lines.json"),
        field: format!("lines[source_bus={a},target_bus={b}]"),
        message: format!(
            "buses {a} and {b} are connected by more than one line, so \
             (source_bus={a}, target_bus={b}) does not identify a single line; \
             reference the intended line by its id, or sum the lines with a named expression"
        ),
    }
}

// ── Integer conversion helpers ────────────────────────────────────────────────

/// Convert an `f64` token value to `i32` if it represents an exact integer in `[0, i32::MAX]`.
///
/// Entity IDs are non-negative integers. The tokenizer stores all numeric literals
/// as `f64`, so we need to verify the value round-trips exactly through `i32`.
pub(crate) fn token_f64_to_i32(v: f64) -> Option<i32> {
    if v < 0.0 || v > f64::from(i32::MAX) || v.fract() != 0.0 {
        return None;
    }
    // SAFETY: v is in [0, i32::MAX] with zero fractional part; the cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(v as i32)
}

/// Convert an `f64` token value to `usize` if it represents an exact non-negative integer.
///
/// Block IDs are non-negative integers. Same rationale as [`token_f64_to_i32`].
pub(crate) fn token_f64_to_usize(v: f64) -> Option<usize> {
    if v < 0.0 || v.fract() != 0.0 {
        return None;
    }
    // SAFETY: v >= 0 and has zero fractional part; usize can represent all
    // non-negative f64 values that fit within platform pointer width.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(v as usize)
}

/// Build a [`VariableRef`] from the parsed variable name, entity ID, optional block
/// ID, and optional bus selector.
///
/// Returns `Err(String)` if the variable name is not a recognised LP variable name, if a
/// block argument is provided for a stage-only variable, or if a `bus=` selector is
/// provided for any variable other than `hydro_turbined`/`hydro_generation`.
// Rationale: one exhaustive match over the LP variable catalog; splitting would scatter the
// canonical catalog and drop the compile-time exhaustiveness check.
#[allow(clippy::too_many_lines)]
pub(crate) fn build_variable_ref(
    name: &str,
    entity_id: EntityId,
    block_id: Option<usize>,
    bus_id: Option<EntityId>,
) -> Result<VariableRef, String> {
    let variable = match name {
        // Stage-only variables (block_id must be None).
        "hydro_storage" => {
            reject_block_argument(name, block_id)?;
            Ok(VariableRef::HydroStorage {
                hydro_id: entity_id,
            })
        }
        "hydro_withdrawal" => {
            reject_block_argument(name, block_id)?;
            Ok(VariableRef::HydroWithdrawal {
                hydro_id: entity_id,
            })
        }
        "hydro_outflow_volume" => {
            reject_block_argument(name, block_id)?;
            Ok(VariableRef::HydroOutflowVolume {
                hydro_id: entity_id,
            })
        }
        "hydro_diversion_volume" => {
            reject_block_argument(name, block_id)?;
            Ok(VariableRef::HydroDiversionVolume {
                hydro_id: entity_id,
            })
        }
        "pumping_volume" => {
            reject_block_argument(name, block_id)?;
            Ok(VariableRef::PumpingVolume {
                station_id: entity_id,
            })
        }
        // Block-capable variables.
        "hydro_evaporation" => Ok(VariableRef::HydroEvaporation {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_inflow" => Ok(VariableRef::HydroInflow {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_storage_initial" => Ok(VariableRef::HydroStorageInitial {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_storage_final" => Ok(VariableRef::HydroStorageFinal {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_useful_volume_initial" => Ok(VariableRef::HydroUsefulVolumeInitial {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_useful_volume_final" => Ok(VariableRef::HydroUsefulVolumeFinal {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_turbined" => Ok(VariableRef::HydroTurbined {
            hydro_id: entity_id,
            block_id,
            bus_id,
        }),
        "hydro_spillage" => Ok(VariableRef::HydroSpillage {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_diversion" => Ok(VariableRef::HydroDiversion {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_outflow" => Ok(VariableRef::HydroOutflow {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_generation" => Ok(VariableRef::HydroGeneration {
            hydro_id: entity_id,
            block_id,
            bus_id,
        }),
        "thermal_generation" => Ok(VariableRef::ThermalGeneration {
            thermal_id: entity_id,
            block_id,
        }),
        "line_direct" => Ok(VariableRef::LineDirect {
            line_id: entity_id,
            block_id,
        }),
        "line_reverse" => Ok(VariableRef::LineReverse {
            line_id: entity_id,
            block_id,
        }),
        "line_exchange" => Ok(VariableRef::LineExchange {
            line_id: entity_id,
            block_id,
        }),
        "bus_deficit" => Ok(VariableRef::BusDeficit {
            bus_id: entity_id,
            block_id,
        }),
        "bus_excess" => Ok(VariableRef::BusExcess {
            bus_id: entity_id,
            block_id,
        }),
        "pumping_flow" => Ok(VariableRef::PumpingFlow {
            station_id: entity_id,
            block_id,
        }),
        "pumping_power" => Ok(VariableRef::PumpingPower {
            station_id: entity_id,
            block_id,
        }),
        "contract_import" => Ok(VariableRef::ContractImport {
            contract_id: entity_id,
            block_id,
        }),
        "contract_export" => Ok(VariableRef::ContractExport {
            contract_id: entity_id,
            block_id,
        }),
        "non_controllable_generation" => Ok(VariableRef::NonControllableGeneration {
            source_id: entity_id,
            block_id,
        }),
        "non_controllable_curtailment" => Ok(VariableRef::NonControllableCurtailment {
            source_id: entity_id,
            block_id,
        }),
        "hydro_availability" => Ok(VariableRef::HydroAvailability {
            hydro_id: entity_id,
            block_id,
        }),
        "anticipated_decision" => {
            if block_id.is_some() {
                return Err(format!(
                    "variable \"anticipated_decision\" is a stage-level scalar and \
                     does not accept a block_id — write \"anticipated_decision({})\", \
                     not \"anticipated_decision({}, ...)\"",
                    entity_id.0, entity_id.0,
                ));
            }
            Ok(VariableRef::AnticipatedDecision {
                thermal_id: entity_id,
            })
        }
        other => Err(format!(
            "unknown variable name \"{other}\": not a recognised LP variable name"
        )),
    }?;

    if bus_id.is_some() && !matches!(name, "hydro_turbined" | "hydro_generation") {
        return Err(bus_selector_error(name));
    }

    Ok(variable)
}

fn reject_block_argument(name: &str, block_id: Option<usize>) -> Result<(), String> {
    if block_id.is_some() {
        return Err(format!(
            "variable \"{name}\" does not accept a block argument"
        ));
    }
    Ok(())
}

/// The call `build_variable_ref` maps back to the same variant.
pub(crate) fn variable_call(variable: &VariableRef) -> String {
    let (name, id, block_id, bus_id) = match *variable {
        VariableRef::HydroStorage { hydro_id } => ("hydro_storage", hydro_id, None, None),
        VariableRef::HydroWithdrawal { hydro_id } => ("hydro_withdrawal", hydro_id, None, None),
        VariableRef::HydroEvaporation { hydro_id, block_id } => {
            ("hydro_evaporation", hydro_id, block_id, None)
        }
        VariableRef::HydroInflow { hydro_id, block_id } => {
            ("hydro_inflow", hydro_id, block_id, None)
        }
        VariableRef::HydroStorageInitial { hydro_id, block_id } => {
            ("hydro_storage_initial", hydro_id, block_id, None)
        }
        VariableRef::HydroStorageFinal { hydro_id, block_id } => {
            ("hydro_storage_final", hydro_id, block_id, None)
        }
        VariableRef::HydroUsefulVolumeInitial { hydro_id, block_id } => {
            ("hydro_useful_volume_initial", hydro_id, block_id, None)
        }
        VariableRef::HydroUsefulVolumeFinal { hydro_id, block_id } => {
            ("hydro_useful_volume_final", hydro_id, block_id, None)
        }
        VariableRef::HydroTurbined {
            hydro_id,
            block_id,
            bus_id,
        } => ("hydro_turbined", hydro_id, block_id, bus_id),
        VariableRef::HydroSpillage { hydro_id, block_id } => {
            ("hydro_spillage", hydro_id, block_id, None)
        }
        VariableRef::HydroDiversion { hydro_id, block_id } => {
            ("hydro_diversion", hydro_id, block_id, None)
        }
        VariableRef::HydroOutflow { hydro_id, block_id } => {
            ("hydro_outflow", hydro_id, block_id, None)
        }
        VariableRef::HydroGeneration {
            hydro_id,
            block_id,
            bus_id,
        } => ("hydro_generation", hydro_id, block_id, bus_id),
        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => ("thermal_generation", thermal_id, block_id, None),
        VariableRef::LineDirect { line_id, block_id } => ("line_direct", line_id, block_id, None),
        VariableRef::LineReverse { line_id, block_id } => ("line_reverse", line_id, block_id, None),
        VariableRef::LineExchange { line_id, block_id } => {
            ("line_exchange", line_id, block_id, None)
        }
        VariableRef::BusDeficit { bus_id, block_id } => ("bus_deficit", bus_id, block_id, None),
        VariableRef::BusExcess { bus_id, block_id } => ("bus_excess", bus_id, block_id, None),
        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => ("pumping_flow", station_id, block_id, None),
        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => ("pumping_power", station_id, block_id, None),
        VariableRef::ContractImport {
            contract_id,
            block_id,
        } => ("contract_import", contract_id, block_id, None),
        VariableRef::ContractExport {
            contract_id,
            block_id,
        } => ("contract_export", contract_id, block_id, None),
        VariableRef::NonControllableGeneration {
            source_id,
            block_id,
        } => ("non_controllable_generation", source_id, block_id, None),
        VariableRef::NonControllableCurtailment {
            source_id,
            block_id,
        } => ("non_controllable_curtailment", source_id, block_id, None),
        VariableRef::AnticipatedDecision { thermal_id } => {
            ("anticipated_decision", thermal_id, None, None)
        }
        VariableRef::HydroOutflowVolume { hydro_id } => {
            ("hydro_outflow_volume", hydro_id, None, None)
        }
        VariableRef::HydroDiversionVolume { hydro_id } => {
            ("hydro_diversion_volume", hydro_id, None, None)
        }
        VariableRef::PumpingVolume { station_id } => ("pumping_volume", station_id, None, None),
        VariableRef::HydroAvailability { hydro_id, block_id } => {
            ("hydro_availability", hydro_id, block_id, None)
        }
        VariableRef::HydroGroupGeneration {
            hydro_id,
            group_id,
            block_id,
        } => {
            let block = block_id.map_or_else(String::new, |k| format!(", {k}"));
            return format!("hydro_generation({hydro_id}{block}, unit_group={group_id})");
        }
    };
    let block = block_id.map_or_else(String::new, |k| format!(", {k}"));
    let bus = bus_id.map_or_else(String::new, |b| format!(", bus={b}"));
    format!("{name}({id}{block}{bus})")
}

/// A value a constraint reads as data: a scenario value. A constraint scope stands each distinct
/// atom for one scalar-parameter id, so a bound holds it as a parameter reference and the value
/// is supplied when the bounds are evaluated.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DataAtom {
    /// Load of a bus, read as data.
    BusLoadValue {
        /// The bus.
        bus_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
}

/// Whether `name` is a value read from the scenario, not an LP variable.
pub(crate) fn is_data_term(name: &str) -> bool {
    name == "bus_load_value"
}

fn bus_selector_error(name: &str) -> String {
    format!(
        "variable \"{name}\" does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" accept a \"bus=\" argument"
    )
}

fn unit_group_selector_error(name: &str) -> String {
    format!(
        "variable \"{name}\" does not accept a unit-group selector; only \"hydro_generation\" accepts a \"unit_group=\" argument"
    )
}

fn entity_id_arg(value: f64, name: &str) -> Result<i32, String> {
    token_f64_to_i32(value).ok_or_else(|| {
        format!("entity_id must be a non-negative integer, got {value} in variable \"{name}\"")
    })
}

fn block_id_arg(value: f64, name: &str) -> Result<usize, String> {
    token_f64_to_usize(value).ok_or_else(|| {
        format!("block_id must be a non-negative integer, got {value} in variable \"{name}\"")
    })
}

/// The arguments of a one-entity call: the entity, an optional block, and an optional `bus=`
/// and `unit_group=`.
struct EntityCall {
    entity_id: EntityId,
    block_id: Option<usize>,
    bus_id: Option<EntityId>,
    unit_group: Option<EntityId>,
}

fn read_entity_call(call: &VariableCall) -> Result<EntityCall, String> {
    let name = call.name.as_str();
    let Some((first, rest)) = call.args.split_first() else {
        return Err(format!("variable \"{name}\" has an empty argument list"));
    };
    let entity_id = match first {
        Arg::Named { .. } => {
            return Err(format!("expected integer entity_id in variable \"{name}\""));
        }
        Arg::Positional { value, .. } => EntityId::from(entity_id_arg(*value, name)?),
    };

    let mut block_id: Option<usize> = None;
    let mut bus_id: Option<EntityId> = None;
    let mut unit_group: Option<EntityId> = None;
    for arg in rest {
        match arg {
            Arg::Positional { value, .. } => {
                if bus_id.is_some() {
                    return Err(format!(
                        "positional block argument must precede the named \"bus=\" argument in variable \"{name}\""
                    ));
                }
                if unit_group.is_some() {
                    return Err(format!(
                        "positional block argument must precede the named \"unit_group=\" argument in variable \"{name}\""
                    ));
                }
                if block_id.is_some() {
                    return Err(format!("repeated block argument in variable \"{name}\""));
                }
                block_id = Some(block_id_arg(*value, name)?);
            }
            Arg::Named { key, value, .. } if key == "bus" => {
                if bus_id.is_some() {
                    return Err(format!("repeated \"bus=\" argument in variable \"{name}\""));
                }
                let bus = token_f64_to_i32(*value).ok_or_else(|| {
                    format!(
                        "bus_id must be a non-negative integer, got {value} in variable \"{name}\""
                    )
                })?;
                bus_id = Some(EntityId::from(bus));
            }
            Arg::Named { key, value, .. } if key == "unit_group" => {
                if unit_group.is_some() {
                    return Err(format!(
                        "repeated \"unit_group=\" argument in variable \"{name}\""
                    ));
                }
                let group = token_f64_to_i32(*value).ok_or_else(|| {
                    format!(
                        "unit_group must be a non-negative integer, got {value} in variable \"{name}\""
                    )
                })?;
                unit_group = Some(EntityId::from(group));
            }
            Arg::Named { key, .. } if key == "source_bus" || key == "target_bus" => {
                return Err(format!(
                    "the (source_bus=, target_bus=) pair form is only accepted on \"line_exchange\" or \"line_flow\" as its sole argument, e.g. \"line_exchange(source_bus=3, target_bus=7)\"; it cannot follow a positional argument in \"{name}\""
                ));
            }
            Arg::Named { key, .. } => {
                return Err(format!(
                    "unknown named argument \"{key}\" in variable \"{name}\": only \"bus\" and \"unit_group\" are supported"
                ));
            }
        }
    }
    Ok(EntityCall {
        entity_id,
        block_id,
        bus_id,
        unit_group,
    })
}

/// A one-entity call of a name that takes neither a bus nor a unit-group selector.
fn read_entity_call_without_bus(call: &VariableCall) -> Result<EntityCall, String> {
    let entity_call = read_entity_call(call)?;
    if entity_call.bus_id.is_some() {
        return Err(bus_selector_error(&call.name));
    }
    if entity_call.unit_group.is_some() {
        return Err(unit_group_selector_error(&call.name));
    }
    Ok(entity_call)
}

/// Resolves a parsed variable call: a positional entity id, an optional positional block and
/// an optional `bus=` or, on `hydro_generation`, `unit_group=` after them, or the
/// `(source_bus=, target_bus=)` pair form of `line_exchange` and `line_flow`, which reads its line
/// and orientation from `line_index`. The direct forms are `Forward`.
pub(crate) fn variable_from_call(
    call: &VariableCall,
    line_index: &LineBusPairIndex,
) -> Result<(VariableRef, LinePairOrientation), String> {
    let name = call.name.as_str();
    if is_data_term(name) {
        return Err(format!("\"{name}\" is a data term, not an LP variable"));
    }
    if name == "line_flow" {
        return line_pair_from_call(call, line_index);
    }
    if let Some(Arg::Named { key, .. }) = call.args.first()
        && (key == "source_bus" || key == "target_bus")
    {
        return line_pair_from_call(call, line_index);
    }

    let EntityCall {
        entity_id,
        block_id,
        bus_id,
        unit_group,
    } = read_entity_call(call)?;
    if let Some(group_id) = unit_group {
        if name != "hydro_generation" {
            return Err(unit_group_selector_error(name));
        }
        if bus_id.is_some() {
            return Err(format!(
                "variable \"{name}\" takes \"bus=\" or \"unit_group=\", not both"
            ));
        }
        return Ok((
            VariableRef::HydroGroupGeneration {
                hydro_id: entity_id,
                group_id,
                block_id,
            },
            LinePairOrientation::Forward,
        ));
    }
    Ok((
        build_variable_ref(name, entity_id, block_id, bus_id)?,
        LinePairOrientation::Forward,
    ))
}

/// The refusal [`variable_from_call`] gives a call whose name is neither a data term nor in the
/// LP variable catalog; `None` for a name the catalog knows. `build_variable_ref` accepts every
/// catalog name given an entity and no block or bus, except `line_flow`, which only the pair
/// form carries.
pub(crate) fn unknown_name_refusal(call: &VariableCall) -> Option<String> {
    let name = call.name.as_str();
    if is_data_term(name)
        || name == "line_flow"
        || build_variable_ref(name, EntityId(0), None, None).is_ok()
    {
        return None;
    }
    variable_from_call(call, &LineBusPairIndex::default()).err()
}

/// The data atom a call reads: a name [`is_data_term`] accepts.
pub(crate) fn data_atom(call: &VariableCall) -> Result<DataAtom, String> {
    let name = call.name.as_str();
    match name {
        "bus_load_value" => bus_load_value(call),
        _ => Err(format!("\"{name}\" is not a data term")),
    }
}

fn bus_load_value(call: &VariableCall) -> Result<DataAtom, String> {
    let EntityCall {
        entity_id,
        block_id,
        ..
    } = read_entity_call_without_bus(call)?;
    Ok(DataAtom::BusLoadValue {
        bus_id: entity_id,
        block_id,
    })
}

/// The scalar-parameter id standing for the data atom [`data_atom`] reads from a call.
pub(crate) fn data_param(
    call: &VariableCall,
    data_ids: Option<&BTreeMap<DataAtom, EntityId>>,
) -> Result<EntityId, String> {
    let name = call.name.as_str();
    let data_ids = data_ids.ok_or_else(|| {
        format!("\"{name}\" reads scenario data, which a scalar parameter cannot read")
    })?;
    data_ids
        .get(&data_atom(call)?)
        .copied()
        .ok_or_else(|| format!("data term \"{name}\" has no parameter id"))
}

/// The pair form: `line_exchange` is the net flow of the line, signed by the pair's orientation;
/// `line_flow` is the flow in the pair's direction, whose variable carries the sign, so it is
/// `Forward`.
fn line_pair_from_call(
    call: &VariableCall,
    line_index: &LineBusPairIndex,
) -> Result<(VariableRef, LinePairOrientation), String> {
    let name = call.name.as_str();
    if !matches!(name, "line_exchange" | "line_flow") {
        return Err(format!(
            "the (source_bus=, target_bus=) pair form addresses a line by its endpoint buses and is only accepted on \"line_exchange\" or \"line_flow\", not \"{name}\""
        ));
    }
    let (source_bus, target_bus) = read_bus_pair(call)?;
    let (line_id, orientation) = line_index.resolve(source_bus, target_bus).ok_or_else(|| {
        format!("no line connects buses source_bus={source_bus} and target_bus={target_bus}")
    })?;
    if name == "line_flow" {
        return Ok((
            directional_line(line_id, orientation),
            LinePairOrientation::Forward,
        ));
    }
    Ok((
        VariableRef::LineExchange {
            line_id,
            block_id: None,
        },
        orientation,
    ))
}

fn read_bus_pair(call: &VariableCall) -> Result<(EntityId, EntityId), String> {
    let name = call.name.as_str();
    let mut source_bus: Option<EntityId> = None;
    let mut target_bus: Option<EntityId> = None;
    for arg in &call.args {
        let (slot, key, value) = match arg {
            Arg::Named { key, value, .. } if key == "source_bus" => (&mut source_bus, key, value),
            Arg::Named { key, value, .. } if key == "target_bus" => (&mut target_bus, key, value),
            _ => {
                return Err(format!(
                    "expected \"source_bus\" or \"target_bus\" in \"{name}\" pair form; write \"{name}(source_bus=X, target_bus=Y)\""
                ));
            }
        };
        let bus = token_f64_to_i32(*value).ok_or_else(|| {
            format!("{key} must be a non-negative integer, got {value} in \"{name}\"")
        })?;
        if slot.is_some() {
            return Err(format!("repeated \"{key}=\" argument in \"{name}\""));
        }
        *slot = Some(EntityId::from(bus));
    }

    let (Some(source_bus), Some(target_bus)) = (source_bus, target_bus) else {
        return Err(format!(
            "the \"{name}\" pair form requires both \"source_bus=\" and \"target_bus=\"; write \"{name}(source_bus=X, target_bus=Y)\""
        ));
    };
    Ok((source_bus, target_bus))
}

fn directional_line(line_id: EntityId, orientation: LinePairOrientation) -> VariableRef {
    match orientation {
        LinePairOrientation::Forward => VariableRef::LineDirect {
            line_id,
            block_id: None,
        },
        LinePairOrientation::Reversed => VariableRef::LineReverse {
            line_id,
            block_id: None,
        },
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::expression::parser::{ExprKind, parse};
    use crate::test_support::make_line;

    /// Build a one-line index: a line `id` connecting `source_bus_id`→`target_bus_id`.
    fn line(id: i32, source: i32, target: i32) -> Line {
        let mut line = make_line(id, source, target);
        line.name = format!("L{id}");
        line
    }

    /// `hydro_inflow` is block-capable: a bare entity id parses with `block_id: None`.
    #[test]
    fn test_build_hydro_inflow_no_block() {
        let var = build_variable_ref("hydro_inflow", EntityId(3), None, None).unwrap();
        assert_eq!(
            var,
            VariableRef::HydroInflow {
                hydro_id: EntityId(3),
                block_id: None,
            }
        );
    }

    /// `hydro_inflow` accepts a block argument: the upstream-release terms are per-block,
    /// so `hydro_inflow(h, k)` threads `block_id: Some(k)` through.
    #[test]
    fn test_build_hydro_inflow_with_block() {
        let var = build_variable_ref("hydro_inflow", EntityId(3), Some(0), None).unwrap();
        assert_eq!(
            var,
            VariableRef::HydroInflow {
                hydro_id: EntityId(3),
                block_id: Some(0),
            }
        );
    }

    /// `hydro_storage_initial` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_storage_initial_block_none_and_some() {
        let none = build_variable_ref("hydro_storage_initial", EntityId(4), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(4),
                block_id: None,
            }
        );
        let some = build_variable_ref("hydro_storage_initial", EntityId(4), Some(2), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(4),
                block_id: Some(2),
            }
        );
    }

    /// `hydro_storage_final` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_storage_final_block_none_and_some() {
        let none = build_variable_ref("hydro_storage_final", EntityId(4), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(4),
                block_id: None,
            }
        );
        let some = build_variable_ref("hydro_storage_final", EntityId(4), Some(1), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(4),
                block_id: Some(1),
            }
        );
    }

    /// `hydro_useful_volume_initial` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_useful_volume_initial_block_none_and_some() {
        let none =
            build_variable_ref("hydro_useful_volume_initial", EntityId(4), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(4),
                block_id: None,
            }
        );
        let some =
            build_variable_ref("hydro_useful_volume_initial", EntityId(4), Some(2), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(4),
                block_id: Some(2),
            }
        );
    }

    /// `hydro_useful_volume_final` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_useful_volume_final_block_none_and_some() {
        let none =
            build_variable_ref("hydro_useful_volume_final", EntityId(3), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(3),
                block_id: None,
            }
        );
        let some =
            build_variable_ref("hydro_useful_volume_final", EntityId(3), Some(0), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(3),
                block_id: Some(0),
            }
        );
    }

    /// Neither useful-volume boundary variable accepts a `bus=` selector — the
    /// post-match guard rejects it the same as for the storage-pair variables.
    #[test]
    fn test_build_hydro_useful_volume_bus_selector_rejected() {
        for name in ["hydro_useful_volume_final", "hydro_useful_volume_initial"] {
            assert!(build_variable_ref(name, EntityId(3), None, Some(EntityId(1))).is_err());
        }
    }

    #[test]
    fn stage_volume_names_resolve_without_a_block_or_bus() {
        let id = EntityId(4);
        assert_eq!(
            build_variable_ref("hydro_outflow_volume", id, None, None),
            Ok(VariableRef::HydroOutflowVolume { hydro_id: id })
        );
        assert_eq!(
            build_variable_ref("hydro_diversion_volume", id, None, None),
            Ok(VariableRef::HydroDiversionVolume { hydro_id: id })
        );
        assert_eq!(
            build_variable_ref("pumping_volume", id, None, None),
            Ok(VariableRef::PumpingVolume { station_id: id })
        );
        for name in [
            "hydro_outflow_volume",
            "hydro_diversion_volume",
            "pumping_volume",
        ] {
            assert_eq!(
                build_variable_ref(name, id, Some(1), None),
                Err(format!(
                    "variable \"{name}\" does not accept a block argument"
                ))
            );
            let err = build_variable_ref(name, id, None, Some(EntityId(1))).unwrap_err();
            assert!(
                err.starts_with(&format!(
                    "variable \"{name}\" does not accept a bus selector"
                )),
                "{err}"
            );
        }
    }

    #[test]
    fn variable_call_round_trips_through_the_parser() {
        const NAMES: [&str; 30] = [
            "hydro_storage",
            "hydro_withdrawal",
            "hydro_outflow_volume",
            "hydro_diversion_volume",
            "pumping_volume",
            "hydro_evaporation",
            "hydro_inflow",
            "hydro_storage_initial",
            "hydro_storage_final",
            "hydro_useful_volume_initial",
            "hydro_useful_volume_final",
            "hydro_turbined",
            "hydro_spillage",
            "hydro_diversion",
            "hydro_outflow",
            "hydro_generation",
            "thermal_generation",
            "line_direct",
            "line_reverse",
            "line_exchange",
            "bus_deficit",
            "bus_excess",
            "pumping_flow",
            "pumping_power",
            "contract_import",
            "contract_export",
            "non_controllable_generation",
            "non_controllable_curtailment",
            "anticipated_decision",
            "hydro_availability",
        ];
        for name in NAMES {
            assert_eq!(unknown_name_refusal(&call(name, vec![])), None, "{name}");
            let mut accepted = false;
            for block in [None, Some(2)] {
                for bus in [None, Some(EntityId(3))] {
                    let Ok(variable) = build_variable_ref(name, EntityId(7), block, bus) else {
                        continue;
                    };
                    accepted = true;
                    let call = variable_call(&variable);
                    let ExprKind::Variable(parsed) = parse(&call).unwrap().lhs.kind else {
                        panic!("`{call}` did not parse to a bare variable call");
                    };
                    assert_eq!(
                        variable_from_call(&parsed, &LineBusPairIndex::default()),
                        Ok((variable, LinePairOrientation::Forward)),
                        "`{call}` did not resolve back to its variant"
                    );
                }
            }
            assert!(accepted, "{name}: no block and bus combination is accepted");
        }
        for block_id in [None, Some(2)] {
            let variable = VariableRef::HydroGroupGeneration {
                hydro_id: EntityId(7),
                group_id: EntityId(1),
                block_id,
            };
            let call = variable_call(&variable);
            let ExprKind::Variable(parsed) = parse(&call).unwrap().lhs.kind else {
                panic!("`{call}` did not parse to a bare variable call");
            };
            assert_eq!(
                variable_from_call(&parsed, &LineBusPairIndex::default()),
                Ok((variable, LinePairOrientation::Forward)),
                "`{call}` did not resolve back to its variant"
            );
        }
    }

    #[test]
    fn test_build_load_and_availability_terms_bus_selector_rejected() {
        assert!(
            build_variable_ref("hydro_availability", EntityId(3), None, Some(EntityId(1))).is_err()
        );
    }

    fn positional(value: f64) -> Arg {
        Arg::Positional { value, span: 0..0 }
    }

    fn call(name: &str, args: Vec<Arg>) -> VariableCall {
        VariableCall {
            name: name.to_owned(),
            args,
        }
    }

    /// A removed load name resolves as an unknown name with the same arguments does.
    #[test]
    fn removed_load_names_are_unknown_names() {
        let index = LineBusPairIndex::default();
        for (removed, args) in [
            ("bus_load", vec![positional(0.0)]),
            ("system_load", vec![]),
            ("system_load_value", vec![]),
        ] {
            let named = call(removed, args.clone());
            let unknown = call("no_such_term", args);
            let renamed = |message: String| message.replace("no_such_term", removed);
            assert!(variable_from_call(&named, &index).is_err(), "{removed}");
            assert_eq!(
                variable_from_call(&named, &index),
                variable_from_call(&unknown, &index).map_err(renamed),
                "{removed}"
            );
            assert_eq!(
                data_atom(&named),
                data_atom(&unknown).map_err(renamed),
                "{removed}"
            );
            assert_eq!(
                is_data_term(removed),
                is_data_term(&unknown.name),
                "{removed}"
            );
        }
    }

    /// A name outside the catalog is refused as `variable_from_call` refuses it, with any
    /// arguments; a data term and `line_flow`, which `build_variable_ref` does not hold, are not
    /// unknown.
    #[test]
    fn an_unknown_name_is_refused_as_variable_from_call_refuses_it() {
        let index = LineBusPairIndex::default();
        for args in [
            vec![],
            vec![positional(3.0)],
            vec![positional(3.0), positional(1.0)],
            vec![named("bus", 2.0)],
            vec![named("source_bus", 1.0), named("target_bus", 2.0)],
        ] {
            let unknown = call("no_such_term", args);
            let refusal = variable_from_call(&unknown, &index).unwrap_err();
            assert_eq!(unknown_name_refusal(&unknown), Some(refusal));
        }
        assert_eq!(
            unknown_name_refusal(&call("no_such_term", vec![positional(3.0)])),
            Some(
                "unknown variable name \"no_such_term\": not a recognised LP variable name"
                    .to_owned()
            )
        );
        for name in ["line_flow", "bus_load_value"] {
            assert_eq!(unknown_name_refusal(&call(name, vec![])), None, "{name}");
        }
    }

    fn named(key: &str, value: f64) -> Arg {
        Arg::Named {
            key: key.to_owned(),
            value,
            span: 0..0,
        }
    }

    /// `line_flow(source_bus=X, target_bus=Y)` is the flow X to Y: the line's direct variable
    /// when it is declared X to Y, its reverse variable when it is declared Y to X.
    #[test]
    fn line_flow_follows_the_declared_direction() {
        let index = build_line_bus_pair_index(&[line(10, 3, 7)]).unwrap();
        let (line_id, block_id) = (EntityId(10), None);
        let flow = |args| variable_from_call(&call("line_flow", args), &index);
        assert_eq!(
            flow(vec![named("source_bus", 3.0), named("target_bus", 7.0)]),
            Ok((
                VariableRef::LineDirect { line_id, block_id },
                LinePairOrientation::Forward
            ))
        );
        assert_eq!(
            flow(vec![named("target_bus", 3.0), named("source_bus", 7.0)]),
            Ok((
                VariableRef::LineReverse { line_id, block_id },
                LinePairOrientation::Forward
            ))
        );
    }

    /// `line_flow` takes only the bus pair: a pair no line connects, a positional argument, a
    /// block, and a missing, repeated or unknown key are each rejected.
    #[test]
    fn line_flow_takes_only_the_bus_pair() {
        let index = build_line_bus_pair_index(&[line(10, 3, 7)]).unwrap();
        let unexpected = "expected \"source_bus\" or \"target_bus\" in \"line_flow\" pair form; \
                          write \"line_flow(source_bus=X, target_bus=Y)\"";
        let incomplete = "the \"line_flow\" pair form requires both \"source_bus=\" and \
                          \"target_bus=\"; write \"line_flow(source_bus=X, target_bus=Y)\"";
        for (args, message) in [
            (
                vec![named("source_bus", 3.0), named("target_bus", 9.0)],
                "no line connects buses source_bus=3 and target_bus=9",
            ),
            (vec![positional(3.0), positional(7.0)], unexpected),
            (vec![named("source_bus", 3.0)], incomplete),
            (vec![], incomplete),
            (
                vec![
                    named("source_bus", 3.0),
                    named("target_bus", 7.0),
                    positional(0.0),
                ],
                unexpected,
            ),
            (
                vec![named("source_bus", 3.0), named("bus", 7.0)],
                unexpected,
            ),
            (
                vec![named("source_bus", 3.0), named("source_bus", 7.0)],
                "repeated \"source_bus=\" argument in \"line_flow\"",
            ),
        ] {
            let flow = call("line_flow", args);
            assert_eq!(
                variable_from_call(&flow, &index),
                Err(message.to_owned()),
                "{flow:?}"
            );
        }
    }

    /// `hydro_generation(X, unit_group=G)` is the generation of unit group G of hydro X, at the
    /// block when one precedes the selector. Any other name, `bus=` beside it, a repeat, a value
    /// that is not a non-negative integer and a block after it are each rejected.
    #[test]
    fn hydro_generation_reads_a_unit_group_selector() {
        let index = LineBusPairIndex::default();
        let group = |block_id| VariableRef::HydroGroupGeneration {
            hydro_id: EntityId(14),
            group_id: EntityId(0),
            block_id,
        };
        let resolve = |args| variable_from_call(&call("hydro_generation", args), &index);
        assert_eq!(
            resolve(vec![positional(14.0), named("unit_group", 0.0)]),
            Ok((group(None), LinePairOrientation::Forward))
        );
        assert_eq!(
            resolve(vec![
                positional(14.0),
                positional(2.0),
                named("unit_group", 0.0)
            ]),
            Ok((group(Some(2)), LinePairOrientation::Forward))
        );
        assert_eq!(
            resolve(vec![positional(14.0), named("bus", 1.0)]),
            Ok((
                VariableRef::HydroGeneration {
                    hydro_id: EntityId(14),
                    block_id: None,
                    bus_id: Some(EntityId(1)),
                },
                LinePairOrientation::Forward
            ))
        );

        let other_name = |name: &str| {
            format!(
                "variable \"{name}\" does not accept a unit-group selector; only \
                 \"hydro_generation\" accepts a \"unit_group=\" argument"
            )
        };
        let both = "variable \"hydro_generation\" takes \"bus=\" or \"unit_group=\", not both";
        for (name, args, message) in [
            (
                "hydro_turbined",
                vec![positional(14.0), named("unit_group", 0.0)],
                other_name("hydro_turbined"),
            ),
            (
                "hydro_generation",
                vec![
                    positional(14.0),
                    named("bus", 1.0),
                    named("unit_group", 0.0),
                ],
                both.to_owned(),
            ),
            (
                "hydro_generation",
                vec![
                    positional(14.0),
                    named("unit_group", 0.0),
                    named("bus", 1.0),
                ],
                both.to_owned(),
            ),
            (
                "hydro_generation",
                vec![
                    positional(14.0),
                    named("unit_group", 0.0),
                    named("unit_group", 1.0),
                ],
                "repeated \"unit_group=\" argument in variable \"hydro_generation\"".to_owned(),
            ),
            (
                "hydro_generation",
                vec![positional(14.0), named("unit_group", 1.5)],
                "unit_group must be a non-negative integer, got 1.5 in variable \
                 \"hydro_generation\""
                    .to_owned(),
            ),
            (
                "hydro_generation",
                vec![positional(14.0), named("unit_group", -1.0)],
                "unit_group must be a non-negative integer, got -1 in variable \
                 \"hydro_generation\""
                    .to_owned(),
            ),
            (
                "hydro_generation",
                vec![positional(14.0), named("unit_group", 0.0), positional(2.0)],
                "positional block argument must precede the named \"unit_group=\" argument in \
                 variable \"hydro_generation\""
                    .to_owned(),
            ),
        ] {
            let call = call(name, args);
            assert_eq!(variable_from_call(&call, &index), Err(message), "{call:?}");
        }
        let selected = call(
            "bus_load_value",
            vec![positional(4.0), named("unit_group", 0.0)],
        );
        assert_eq!(data_atom(&selected), Err(other_name("bus_load_value")));
    }

    /// The load and availability names take ids, as variables, as data atoms and in a call
    /// shape; a data term is no LP variable and a scalar parameter cannot read it.
    #[test]
    fn load_and_availability_terms_take_ids() {
        let index = LineBusPairIndex::default();
        let availability = |block_id| VariableRef::HydroAvailability {
            hydro_id: EntityId(14),
            block_id,
        };
        let variable =
            |name: &str, args: Vec<Arg>| variable_from_call(&call(name, args), &index).map(|r| r.0);
        for (name, args, expected) in [
            (
                "hydro_availability",
                vec![positional(14.0)],
                availability(None),
            ),
            (
                "hydro_availability",
                vec![positional(14.0), positional(2.0)],
                availability(Some(2)),
            ),
        ] {
            assert_eq!(variable(name, args), Ok(expected), "{name}");
        }

        assert_eq!(
            data_atom(&call("bus_load_value", vec![positional(0.0)])),
            Ok(DataAtom::BusLoadValue {
                bus_id: EntityId(0),
                block_id: None,
            })
        );

        for name in ["bus_load_value", "hydro_availability"] {
            assert_eq!(is_data_term(name), name == "bus_load_value", "{name}");
        }

        for well_formed in [
            call("bus_load_value", vec![positional(0.0)]),
            call("bus_load_value", vec![positional(0.0), positional(1.0)]),
        ] {
            assert_eq!(data_atom(&well_formed).map(drop), Ok(()), "{well_formed:?}");
        }
        for bad in [
            call("bus_load_value", vec![]),
            call("bus_load_value", vec![positional(0.0), named("bus", 1.0)]),
        ] {
            assert!(data_atom(&bad).is_err(), "{bad:?}");
        }
        let data = call("bus_load_value", vec![positional(0.0)]);
        assert_eq!(
            data_param(&data, None),
            Err(format!(
                "\"{}\" reads scenario data, which a scalar parameter cannot read",
                data.name
            ))
        );
        assert_eq!(
            variable_from_call(&data, &index),
            Err(format!(
                "\"{}\" is a data term, not an LP variable",
                data.name
            ))
        );
    }

    /// Two distinct lines sharing one unordered bus pair make the index build fail loudly,
    /// naming the buses and pointing at named expressions — regardless of the lines'
    /// declared orientations or their order in the input.
    #[test]
    fn line_bus_pair_duplicate_pair_is_loud_error() {
        for lines in [
            [line(1, 3, 7), line(2, 3, 7)],
            [line(2, 7, 3), line(1, 3, 7)],
        ] {
            let err = build_line_bus_pair_index(&lines).unwrap_err();
            match err {
                LoadError::SchemaError { message, .. } => {
                    assert!(
                        message.contains("buses 3 and 7"),
                        "message must name buses 3 and 7: {message}"
                    );
                    assert!(
                        message.contains("named expression"),
                        "message must point at named expressions: {message}"
                    );
                }
                other => panic!("expected SchemaError, got: {other:?}"),
            }
        }
    }
}
