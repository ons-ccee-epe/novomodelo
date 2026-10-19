use novomodelo_core::{EntityId, Line, VariableRef};
use std::collections::HashMap;
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
            if block_id.is_some() {
                return Err(format!(
                    "variable \"{name}\" does not accept a block argument"
                ));
            }
            Ok(VariableRef::HydroStorage {
                hydro_id: entity_id,
            })
        }
        "hydro_withdrawal" => {
            if block_id.is_some() {
                return Err(format!(
                    "variable \"{name}\" does not accept a block argument"
                ));
            }
            Ok(VariableRef::HydroWithdrawal {
                hydro_id: entity_id,
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

    if bus_id.is_some()
        && !matches!(
            variable,
            VariableRef::HydroTurbined { .. } | VariableRef::HydroGeneration { .. }
        )
    {
        return Err(format!(
            "variable \"{name}\" does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" accept a \"bus=\" argument"
        ));
    }

    Ok(variable)
}

/// Resolves a parsed variable call: a positional entity id, an optional positional block and
/// an optional `bus=` after them, or the `line_exchange(source_bus=, target_bus=)` pair form,
/// which reads its line and orientation from `line_index`. The direct forms are `Forward`.
pub(crate) fn variable_from_call(
    call: &VariableCall,
    line_index: &LineBusPairIndex,
) -> Result<(VariableRef, LinePairOrientation), String> {
    let name = call.name.as_str();
    let Some((first, rest)) = call.args.split_first() else {
        return Err(format!("variable \"{name}\" has an empty argument list"));
    };
    let entity_id = match first {
        Arg::Named { key, .. } if key == "source_bus" || key == "target_bus" => {
            return line_pair_from_call(call, line_index);
        }
        Arg::Named { .. } => {
            return Err(format!("expected integer entity_id in variable \"{name}\""));
        }
        Arg::Positional { value, .. } => token_f64_to_i32(*value).ok_or_else(|| {
            format!("entity_id must be a non-negative integer, got {value} in variable \"{name}\"")
        })?,
    };

    let mut block_id: Option<usize> = None;
    let mut bus_id: Option<EntityId> = None;
    for arg in rest {
        match arg {
            Arg::Positional { value, .. } => {
                if bus_id.is_some() {
                    return Err(format!(
                        "positional block argument must precede the named \"bus=\" argument in variable \"{name}\""
                    ));
                }
                if block_id.is_some() {
                    return Err(format!("repeated block argument in variable \"{name}\""));
                }
                block_id = Some(token_f64_to_usize(*value).ok_or_else(|| {
                    format!(
                        "block_id must be a non-negative integer, got {value} in variable \"{name}\""
                    )
                })?);
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
            Arg::Named { key, .. } if key == "source_bus" || key == "target_bus" => {
                return Err(format!(
                    "the (source_bus=, target_bus=) pair form is only accepted on \"line_exchange\" as its sole argument, e.g. \"line_exchange(source_bus=3, target_bus=7)\"; it cannot follow a positional argument in \"{name}\""
                ));
            }
            Arg::Named { key, .. } => {
                return Err(format!(
                    "unknown named argument \"{key}\" in variable \"{name}\": only \"bus\" is supported"
                ));
            }
        }
    }

    build_variable_ref(name, EntityId::from(entity_id), block_id, bus_id)
        .map(|variable| (variable, LinePairOrientation::Forward))
}

/// The pair form: exactly one `source_bus=` and one `target_bus=`, on `line_exchange` only.
fn line_pair_from_call(
    call: &VariableCall,
    line_index: &LineBusPairIndex,
) -> Result<(VariableRef, LinePairOrientation), String> {
    let name = call.name.as_str();
    if name != "line_exchange" {
        return Err(format!(
            "the (source_bus=, target_bus=) pair form addresses a line by its endpoint buses and is only accepted on \"line_exchange\", not \"{name}\""
        ));
    }

    let mut source_bus: Option<EntityId> = None;
    let mut target_bus: Option<EntityId> = None;
    for arg in &call.args {
        let (slot, key, value) = match arg {
            Arg::Named { key, value, .. } if key == "source_bus" => (&mut source_bus, key, value),
            Arg::Named { key, value, .. } if key == "target_bus" => (&mut target_bus, key, value),
            _ => {
                return Err(format!(
                    "expected \"source_bus\" or \"target_bus\" in \"{name}\" pair form"
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
            "the \"{name}\" pair form requires both \"source_bus=\" and \"target_bus=\"; write \"line_exchange(source_bus=X, target_bus=Y)\""
        ));
    };
    let (line_id, orientation) = line_index.resolve(source_bus, target_bus).ok_or_else(|| {
        format!("no line connects buses source_bus={source_bus} and target_bus={target_bus}")
    })?;
    Ok((
        VariableRef::LineExchange {
            line_id,
            block_id: None,
        },
        orientation,
    ))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Build a one-line index: a line `id` connecting `source_bus_id`→`target_bus_id`.
    fn line(id: i32, source: i32, target: i32) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .expect("valid date"),
            source_bus_id: EntityId(source),
            target_bus_id: EntityId(target),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
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
