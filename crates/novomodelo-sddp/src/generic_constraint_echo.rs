//! Assemble the resolved generic-constraint echo from the built stage templates.
//!
//! Projects `Vec<GenericConstraintEchoRow>` from the data the LP builder actually
//! consumed: the per-stage row entries (bounds, block, slack), the resolved
//! generic coefficients (term values, per-solve sides, conditions, plant capacities read), and
//! the system's generic constraints (LHS terms). One row per resolved LHS term of each active
//! `(stage, block)` row, then one per plant capacity its constraint reads; a row with neither
//! contributes a single placeholder row. Output is in canonical `(constraint, stage, block)`
//! order, a row's terms by term index before its capacity reads, and declaration-order
//! invariant.

use novomodelo_core::{EntityId, GenericConstraint, PenaltyBasis, System, VariableRef};
use novomodelo_io::GenericConstraintEchoRow;

use crate::StudySetup;
use crate::lp::builder::{GenericConstraintRowEntry, RowOrigin};
use crate::policy::resolved_generic_coefficients::{BoundSide, ResolvedGenericCoefficients};

/// Build the resolved generic-constraint echo rows for `setup`/`system` in
/// canonical `(constraint, stage, block, term)` order.
#[must_use]
pub fn build_generic_constraint_echo_rows(
    setup: &StudySetup,
    system: &System,
) -> Vec<GenericConstraintEchoRow> {
    build_echo_rows_from_parts(
        &setup
            .inputs
            .stage_data
            .stage_templates
            .generic_constraint_row_entries,
        &setup.inputs.study_stage_ids,
        &setup.inputs.stage_data.generic_rows,
        system.generic_constraints(),
    )
}

/// Assemble the echo rows from the raw parts (slices in, canonically-ordered `Vec`
/// out) — decoupled from `StudySetup` so the assembly is unit-testable without a
/// full setup. `entries_per_stage` is indexed `[stage][row]`; `stage_ids[s]` is
/// the study stage id; `constraints` is joined by a row's `constraint_idx`.
// Rationale (cast lints): block and term indices are stage-local counts far below
// i32::MAX; the echo's i32 columns are Parquet Int32.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_echo_rows_from_parts(
    entries_per_stage: &[Vec<GenericConstraintRowEntry>],
    stage_ids: &[i32],
    coefficients: &ResolvedGenericCoefficients,
    constraints: &[GenericConstraint],
) -> Vec<GenericConstraintEchoRow> {
    let mut rows = Vec::new();

    for (stage_idx, entries) in entries_per_stage.iter().enumerate() {
        let stage_id = stage_ids[stage_idx];
        for entry in entries {
            let RowOrigin::Generic { constraint_idx } = entry.origin;
            let constraint = &constraints[constraint_idx];
            let block_id = if entry.is_stage_level {
                None
            } else {
                Some(entry.block_idx as i32)
            };
            let lower_per_solve = entry.bound_lower.is_some()
                && coefficients.side_is_per_solve(constraint_idx, BoundSide::Lower);
            let upper_per_solve = entry.bound_upper.is_some()
                && coefficients.side_is_per_solve(constraint_idx, BoundSide::Upper);
            let marker = GenericConstraintEchoRow {
                stage_id,
                block_id,
                constraint_id: constraint.id.0,
                constraint_name: constraint.name.clone(),
                term_index: None,
                variable_kind: None,
                variable: None,
                coefficient: None,
                bound_lower: entry.bound_lower.filter(|_| !lower_per_solve),
                bound_upper: entry.bound_upper.filter(|_| !upper_per_solve),
                // From the layout's endpoints, where a per-solve side is the infinity of its side.
                derived_shape: derived_shape(entry.bound_lower, entry.bound_upper).to_string(),
                slack_enabled: entry.slack.is_some(),
                slack_penalty: entry.slack.map(|s| s.penalty),
                slack_penalty_basis: entry
                    .slack
                    .map(|s| penalty_basis_label(s.penalty_basis).to_string()),
                value: None,
                lower_per_solve,
                upper_per_solve,
                has_active_when: coefficients.is_gated(constraint_idx),
            };
            let first = rows.len();

            for (term_idx, term) in constraint.expression.terms.iter().enumerate() {
                let (kind, rendered) = render_variable(&term.variable);
                rows.push(GenericConstraintEchoRow {
                    term_index: Some(term_idx as i32),
                    variable_kind: Some(kind.to_string()),
                    variable: Some(rendered),
                    coefficient: Some(coefficients.term_value(
                        constraint_idx,
                        term_idx,
                        stage_idx,
                        entry.block_idx,
                    )),
                    ..marker.clone()
                });
            }
            for &(atom, id) in coefficients.capacity_reads(constraint_idx) {
                let (kind, rendered) = render_variable(&VariableRef::HydroAvailability {
                    hydro_id: atom.hydro_id,
                    block_id: atom.block_id,
                });
                let value = coefficients.setup_value(id, stage_idx, entry.block_idx);
                debug_assert!(
                    value.is_some(),
                    "a capacity a constraint reads has no setup value"
                );
                rows.push(GenericConstraintEchoRow {
                    variable_kind: Some(kind.to_string()),
                    variable: Some(rendered),
                    value,
                    ..marker.clone()
                });
            }
            if rows.len() == first {
                rows.push(marker);
            }
        }
    }

    // Canonical order is a function of the resolved model, so it does not depend on
    // the stage-major walk order — the declaration-order-invariance guarantee. The sort is
    // stable, so a row's capacity reads keep their order after its terms.
    rows.sort_by_key(|r| {
        (
            r.constraint_id,
            r.stage_id,
            r.block_id.unwrap_or(-1),
            r.term_index.unwrap_or(i32::MAX),
        )
    });
    rows
}

/// Shape label from the bound null-pattern (never an authored sense).
fn derived_shape(bound_lower: Option<f64>, bound_upper: Option<f64>) -> &'static str {
    match (bound_lower, bound_upper) {
        (Some(l), Some(u)) => {
            // Exact (bit-identical) equality, never a tolerance — a narrow band
            // must stay "band".
            if l.to_bits() == u.to_bits() {
                "equality"
            } else {
                "band"
            }
        }
        (Some(_), None) => "floor",
        (None, Some(_)) => "cap",
        (None, None) => {
            debug_assert!(
                false,
                "generic-constraint row has neither bound endpoint; referential validation guarantees at least one"
            );
            "open"
        }
    }
}

/// The `penalty_basis` spelling of `generic_constraints.json`.
fn penalty_basis_label(basis: PenaltyBasis) -> &'static str {
    match basis {
        PenaltyBasis::PerHour => "per_hour",
        PenaltyBasis::PerUnit => "per_unit",
    }
}

/// Return `(kind_tag, rendered_label)` for a variable reference — the `"kind"`
/// discriminant and the `"kind(args)"` display string. The exhaustive `match`
/// makes a new [`VariableRef`] variant a compile error.
// Rationale: one exhaustive match over the variable catalog; splitting would scatter the
// catalog and drop the compile-time exhaustiveness check.
#[allow(clippy::too_many_lines)]
pub(crate) fn render_variable(v: &VariableRef) -> (&'static str, String) {
    let (kind, args) = match *v {
        VariableRef::HydroStorage { hydro_id } => ("hydro_storage", format!("id={}", hydro_id.0)),
        VariableRef::HydroTurbined {
            hydro_id,
            block_id,
            bus_id,
        } => (
            "hydro_turbined",
            format!(
                "id={}, bus={}, block={}",
                hydro_id.0,
                bus_label(bus_id),
                block_label(block_id)
            ),
        ),
        VariableRef::HydroSpillage { hydro_id, block_id } => (
            "hydro_spillage",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroDiversion { hydro_id, block_id } => (
            "hydro_diversion",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroOutflow { hydro_id, block_id } => (
            "hydro_outflow",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroGeneration {
            hydro_id,
            block_id,
            bus_id,
        } => (
            "hydro_generation",
            format!(
                "id={}, bus={}, block={}",
                hydro_id.0,
                bus_label(bus_id),
                block_label(block_id)
            ),
        ),
        VariableRef::HydroEvaporation { hydro_id, block_id } => (
            "hydro_evaporation",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroWithdrawal { hydro_id } => {
            ("hydro_withdrawal", format!("id={}", hydro_id.0))
        }
        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => (
            "thermal_generation",
            format!("id={}, block={}", thermal_id.0, block_label(block_id)),
        ),
        VariableRef::LineDirect { line_id, block_id } => (
            "line_direct",
            format!("id={}, block={}", line_id.0, block_label(block_id)),
        ),
        VariableRef::LineReverse { line_id, block_id } => (
            "line_reverse",
            format!("id={}, block={}", line_id.0, block_label(block_id)),
        ),
        VariableRef::LineExchange { line_id, block_id } => (
            "line_exchange",
            format!("id={}, block={}", line_id.0, block_label(block_id)),
        ),
        VariableRef::BusDeficit { bus_id, block_id } => (
            "bus_deficit",
            format!("id={}, block={}", bus_id.0, block_label(block_id)),
        ),
        VariableRef::BusExcess { bus_id, block_id } => (
            "bus_excess",
            format!("id={}, block={}", bus_id.0, block_label(block_id)),
        ),
        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => (
            "pumping_flow",
            format!("id={}, block={}", station_id.0, block_label(block_id)),
        ),
        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => (
            "pumping_power",
            format!("id={}, block={}", station_id.0, block_label(block_id)),
        ),
        VariableRef::ContractImport {
            contract_id,
            block_id,
        } => (
            "contract_import",
            format!("id={}, block={}", contract_id.0, block_label(block_id)),
        ),
        VariableRef::ContractExport {
            contract_id,
            block_id,
        } => (
            "contract_export",
            format!("id={}, block={}", contract_id.0, block_label(block_id)),
        ),
        VariableRef::NonControllableGeneration {
            source_id,
            block_id,
        } => (
            "non_controllable_generation",
            format!("id={}, block={}", source_id.0, block_label(block_id)),
        ),
        VariableRef::NonControllableCurtailment {
            source_id,
            block_id,
        } => (
            "non_controllable_curtailment",
            format!("id={}, block={}", source_id.0, block_label(block_id)),
        ),
        VariableRef::AnticipatedDecision { thermal_id } => {
            ("anticipated_decision", format!("id={}", thermal_id.0))
        }
        VariableRef::HydroInflow { hydro_id, block_id } => (
            "hydro_inflow",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroStorageInitial { hydro_id, block_id } => (
            "hydro_storage_initial",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroStorageFinal { hydro_id, block_id } => (
            "hydro_storage_final",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroUsefulVolumeInitial { hydro_id, block_id } => (
            "hydro_useful_volume_initial",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroUsefulVolumeFinal { hydro_id, block_id } => (
            "hydro_useful_volume_final",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroOutflowVolume { hydro_id } => {
            ("hydro_outflow_volume", format!("id={}", hydro_id.0))
        }
        VariableRef::HydroDiversionVolume { hydro_id } => {
            ("hydro_diversion_volume", format!("id={}", hydro_id.0))
        }
        VariableRef::PumpingVolume { station_id } => {
            ("pumping_volume", format!("id={}", station_id.0))
        }
        VariableRef::HydroAvailability { hydro_id, block_id } => (
            "hydro_availability",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroGroupGeneration {
            hydro_id,
            group_id,
            block_id,
        } => (
            "hydro_generation",
            format!(
                "id={}, unit_group={}, block={}",
                hydro_id.0,
                group_id.0,
                block_label(block_id)
            ),
        ),
    };
    (kind, format!("{kind}({args})"))
}

/// `Some(k)` renders the block index; `None` renders `all`.
fn block_label(block_id: Option<usize>) -> String {
    block_id.map_or_else(|| "all".to_string(), |k| k.to_string())
}

/// `Some(b)` renders the bus id; `None` renders `all` (the plant's cells summed).
fn bus_label(bus_id: Option<EntityId>) -> String {
    bus_id.map_or_else(|| "all".to_string(), |b| b.0.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use novomodelo_core::test_support::{HydroSpec, make_hydro};
    use novomodelo_core::{
        ConstraintExpression, HydroUnitGroupBoundsCountsSpec, LinearTerm,
        ResolvedHydroUnitGroupBounds, ScalarExpr, SlackConfig, StageId,
    };
    use novomodelo_io::constraints::scenario::{
        AvailabilityAtom, CompareOp, DataAtom, ResolvedCondition,
    };
    use novomodelo_io::constraints::{ConstraintScenario, ScenarioLayer};

    use crate::ResolvedParameters;
    use crate::policy::resolved_generic_coefficients::{LoadBuses, ScenarioSources};

    const PER_UNIT_5000: Option<SlackConfig> = Some(SlackConfig {
        penalty: 5000.0,
        penalty_basis: PenaltyBasis::PerUnit,
    });

    fn entry(
        constraint_idx: usize,
        entity_id: i32,
        block_idx: usize,
        is_stage_level: bool,
        bound_lower: Option<f64>,
        bound_upper: Option<f64>,
        slack: Option<SlackConfig>,
    ) -> GenericConstraintRowEntry {
        GenericConstraintRowEntry {
            origin: RowOrigin::Generic { constraint_idx },
            entity_id,
            block_idx,
            is_stage_level,
            bound_lower,
            bound_upper,
            slack,
            slack_plus_col: None,
            slack_minus_col: None,
        }
    }

    fn constraint(id: i32, name: &str, terms: Vec<LinearTerm>) -> GenericConstraint {
        GenericConstraint {
            id: EntityId(id),
            name: name.to_string(),
            description: None,
            expression: ConstraintExpression { terms },
            slack: None,
            bound_lower: None,
            bound_upper: None,
        }
    }

    fn coefficients(
        constraints: &[GenericConstraint],
        resolved: &ResolvedParameters,
        stage_ids: &[i32],
        entries: &[Vec<GenericConstraintRowEntry>],
    ) -> ResolvedGenericCoefficients {
        let study_stage_ids: Vec<StageId> = stage_ids.iter().copied().map(StageId).collect();
        let block_counts: Vec<usize> = entries
            .iter()
            .map(|stage| stage.iter().map(|e| e.block_idx + 1).fold(1, usize::max))
            .collect();
        ResolvedGenericCoefficients::build(constraints, resolved, &study_stage_ids, &block_counts)
            .unwrap()
    }

    fn thermal(id: i32) -> VariableRef {
        VariableRef::ThermalGeneration {
            thermal_id: EntityId(id),
            block_id: None,
        }
    }

    const PLANT_14_MW: f64 = 300.0;
    const BUS_1: [EntityId; 1] = [EntityId(1)];

    /// The owner of `constraints` over one study stage of one block, with plant 14, one unit
    /// group of [`PLANT_14_MW`], behind the capacities of `layer`, and bus 1, whose load is drawn
    /// as noise, behind its loads.
    fn scenario_coefficients(
        constraints: &[GenericConstraint],
        layer: &ScenarioLayer,
    ) -> ResolvedGenericCoefficients {
        let hydros = [make_hydro(HydroSpec {
            id: 14,
            max_generation_mw: PLANT_14_MW,
            ..HydroSpec::default()
        })];
        let group_bounds = ResolvedHydroUnitGroupBounds::new(&HydroUnitGroupBoundsCountsSpec {
            groups_per_plant: &[1],
            n_stages: 1,
            max_blocks: 1,
        });
        ResolvedGenericCoefficients::build_with_scenario(
            constraints,
            &ScenarioSources {
                layer,
                hydros: &hydros,
                group_bounds: &group_bounds,
                loads: LoadBuses {
                    buses_with_load: &BUS_1,
                    noise_members: &BUS_1,
                    bus_ids: &BUS_1,
                },
            },
            &ResolvedParameters::default(),
            &[StageId(7)],
            &[1],
        )
        .unwrap()
    }

    /// The one row of constraint 0 at the one block, with the endpoints the layout takes.
    fn layout_entry(
        rgc: &ResolvedGenericCoefficients,
        entity_id: i32,
    ) -> GenericConstraintRowEntry {
        let endpoints = rgc.endpoints(0, 0, 0, None, None);
        entry(
            0,
            entity_id,
            0,
            false,
            endpoints.lower,
            endpoints.upper,
            None,
        )
    }

    /// derived: the upper bound reads the load of bus 1 (data atom 940), so a solve evaluates it
    /// and the layout holds it at `+inf`; the lower bound `35` is fixed, and the shape counts
    /// both sides.
    #[test]
    fn a_per_solve_side_is_flagged_and_has_no_fixed_bound() {
        let load = || ScalarExpr::param(EntityId(940));
        let layer = ScenarioLayer {
            data_atoms: vec![(
                EntityId(940),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(1),
                    block_id: None,
                },
            )],
            constraints: vec![ConstraintScenario {
                constraint_id: EntityId(13),
                active_when: Some(ResolvedCondition::Compare {
                    lhs: load(),
                    op: CompareOp::Gt,
                    rhs: ScalarExpr::literal(0.0),
                }),
                endpoint_shift: None,
            }],
            ..ScenarioLayer::default()
        };
        let mut gated = constraint(13, "gated", vec![LinearTerm::literal(1.0, thermal(0))]);
        gated.bound_lower = Some(ScalarExpr::literal(35.0));
        gated.bound_upper = Some(load());
        let constraints = vec![gated];
        let rgc = scenario_coefficients(&constraints, &layer);

        let rows =
            build_echo_rows_from_parts(&[vec![layout_entry(&rgc, 13)]], &[7], &rgc, &constraints);

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.term_index, Some(0));
        assert_eq!(r.bound_lower, Some(35.0));
        assert_eq!(r.bound_upper, None);
        assert!(!r.lower_per_solve);
        assert!(r.upper_per_solve);
        assert!(r.has_active_when);
        assert_eq!(r.derived_shape, "band");
        assert_eq!(r.value, None);
    }

    /// derived: plant 14's one unit group of [`PLANT_14_MW`] is its capacity at the one cell, a
    /// setup value, so the upper bound it sets stays fixed.
    #[test]
    fn a_capacity_read_is_a_row_with_its_setup_value_after_the_terms() {
        let capacity = EntityId(900);
        let layer = ScenarioLayer {
            availability_atoms: vec![(
                capacity,
                AvailabilityAtom {
                    hydro_id: EntityId(14),
                    block_id: None,
                },
            )],
            ..ScenarioLayer::default()
        };
        let mut c = constraint(
            21,
            "under_capacity",
            vec![
                LinearTerm::literal(1.0, thermal(0)),
                LinearTerm::literal(1.0, thermal(1)),
            ],
        );
        c.bound_upper = Some(ScalarExpr::param(capacity));
        let constraints = vec![c];
        let rgc = scenario_coefficients(&constraints, &layer);

        let rows =
            build_echo_rows_from_parts(&[vec![layout_entry(&rgc, 21)]], &[7], &rgc, &constraints);

        assert_eq!(
            rows.iter().map(|r| r.term_index).collect::<Vec<_>>(),
            [Some(0), Some(1), None]
        );
        let read = &rows[2];
        assert_eq!(read.variable_kind.as_deref(), Some("hydro_availability"));
        assert_eq!(
            read.variable.as_deref(),
            Some("hydro_availability(id=14, block=all)")
        );
        assert_eq!(read.coefficient, None);
        let setup = rgc.setup_value(capacity, 0, 0);
        assert_eq!(setup, Some(PLANT_14_MW));
        assert_eq!(read.value.map(f64::to_bits), setup.map(f64::to_bits));
        assert_eq!(read.bound_upper, Some(PLANT_14_MW));
        assert!(rows[..2].iter().all(|r| r.value.is_none()));
    }

    #[test]
    fn a_term_less_constraint_reading_a_capacity_has_no_placeholder_row() {
        let layer = ScenarioLayer {
            availability_atoms: vec![(
                EntityId(900),
                AvailabilityAtom {
                    hydro_id: EntityId(14),
                    block_id: None,
                },
            )],
            ..ScenarioLayer::default()
        };
        let mut c = constraint(22, "capacity_floor", vec![]);
        c.bound_lower = Some(ScalarExpr::literal(50.0) - ScalarExpr::param(EntityId(900)));
        let constraints = vec![c];
        let rgc = scenario_coefficients(&constraints, &layer);

        let rows =
            build_echo_rows_from_parts(&[vec![layout_entry(&rgc, 22)]], &[7], &rgc, &constraints);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].term_index, None);
        assert_eq!(rows[0].variable_kind.as_deref(), Some("hydro_availability"));
        assert_eq!(rows[0].value, Some(PLANT_14_MW));
    }

    /// Every row of constraints the scenario layer does not reach carries the fields it had
    /// before the scenario columns, no capacity value and every flag `false`.
    #[test]
    fn a_constraint_without_scenario_data_echoes_as_before() {
        let constraints = vec![
            constraint(
                54,
                "band",
                vec![
                    LinearTerm::literal(1.0, thermal(3)),
                    LinearTerm::literal(-2.0, thermal(4)),
                ],
            ),
            constraint(7, "const_only", vec![]),
        ];
        let entries = vec![vec![
            entry(0, 54, 0, false, Some(35.0), Some(100.0), PER_UNIT_5000),
            entry(0, 54, 1, false, Some(35.0), Some(100.0), PER_UNIT_5000),
            entry(1, 7, 0, true, Some(5.0), None, None),
        ]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[2],
            &coefficients(&constraints, &resolved, &[2], &entries),
            &constraints,
        );

        let term = GenericConstraintEchoRow {
            stage_id: 2,
            block_id: Some(0),
            constraint_id: 54,
            constraint_name: "band".to_string(),
            term_index: Some(0),
            variable_kind: Some("thermal_generation".to_string()),
            variable: Some("thermal_generation(id=3, block=all)".to_string()),
            coefficient: Some(1.0),
            bound_lower: Some(35.0),
            bound_upper: Some(100.0),
            derived_shape: "band".to_string(),
            slack_enabled: true,
            slack_penalty: Some(5000.0),
            slack_penalty_basis: Some("per_unit".to_string()),
            value: None,
            lower_per_solve: false,
            upper_per_solve: false,
            has_active_when: false,
        };
        let second_term = GenericConstraintEchoRow {
            term_index: Some(1),
            variable: Some("thermal_generation(id=4, block=all)".to_string()),
            coefficient: Some(-2.0),
            ..term.clone()
        };
        let placeholder = GenericConstraintEchoRow {
            block_id: None,
            constraint_id: 7,
            constraint_name: "const_only".to_string(),
            term_index: None,
            variable_kind: None,
            variable: None,
            coefficient: None,
            bound_lower: Some(5.0),
            bound_upper: None,
            derived_shape: "floor".to_string(),
            slack_enabled: false,
            slack_penalty: None,
            slack_penalty_basis: None,
            ..term.clone()
        };
        let expected = [
            placeholder,
            term.clone(),
            second_term.clone(),
            GenericConstraintEchoRow {
                block_id: Some(1),
                ..term
            },
            GenericConstraintEchoRow {
                block_id: Some(1),
                ..second_term
            },
        ];
        assert!(rows.iter().all(|r| r.value.is_none()
            && !r.lower_per_solve
            && !r.upper_per_solve
            && !r.has_active_when));
        assert_eq!(format!("{rows:?}"), format!("{expected:?}"));
    }

    /// An upper-only stage-level entry (the d13 shape) renders a `cap` row with
    /// `block_id = None`, one resolved term, and the entry's slack config.
    #[test]
    fn cap_shape_stage_level_thermal_row() {
        let constraints = vec![constraint(
            13,
            "d13",
            vec![LinearTerm::literal(1.0, thermal(0))],
        )];
        let entries = vec![vec![entry(0, 13, 0, true, None, Some(10.0), PER_UNIT_5000)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[7],
            &coefficients(&constraints, &resolved, &[7], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.stage_id, 7);
        assert_eq!(r.block_id, None);
        assert_eq!(r.constraint_id, 13);
        assert_eq!(r.constraint_name, "d13");
        assert_eq!(r.derived_shape, "cap");
        assert_eq!(r.bound_lower, None);
        assert_eq!(r.bound_upper, Some(10.0));
        assert_eq!(r.term_index, Some(0));
        assert_eq!(r.variable_kind.as_deref(), Some("thermal_generation"));
        assert_eq!(
            r.variable.as_deref(),
            Some("thermal_generation(id=0, block=all)")
        );
        assert_eq!(r.coefficient, Some(1.0));
        assert!(r.slack_enabled);
        assert_eq!(r.slack_penalty, Some(5000.0));
        assert_eq!(r.slack_penalty_basis.as_deref(), Some("per_unit"));
    }

    #[test]
    fn stage_volume_row_echoes_the_user_coefficient_not_tau() {
        let constraints = vec![constraint(
            21,
            "vol",
            vec![
                LinearTerm::literal(
                    2.0,
                    VariableRef::HydroOutflowVolume {
                        hydro_id: EntityId(0),
                    },
                ),
                LinearTerm::literal(
                    -1.0,
                    VariableRef::PumpingVolume {
                        station_id: EntityId(1),
                    },
                ),
            ],
        )];
        let entries = vec![vec![entry(0, 21, 0, true, None, Some(9.0), None)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].term_index, Some(0));
        assert_eq!(rows[0].block_id, None);
        assert_eq!(
            rows[0].variable_kind.as_deref(),
            Some("hydro_outflow_volume")
        );
        assert_eq!(
            rows[0].variable.as_deref(),
            Some("hydro_outflow_volume(id=0)")
        );
        assert_eq!(rows[0].coefficient, Some(2.0));
        assert_eq!(rows[1].term_index, Some(1));
        assert_eq!(rows[1].block_id, None);
        assert_eq!(rows[1].variable_kind.as_deref(), Some("pumping_volume"));
        assert_eq!(rows[1].variable.as_deref(), Some("pumping_volume(id=1)"));
        assert_eq!(rows[1].coefficient, Some(-1.0));
    }

    /// The echo shows the POST-NORMALIZATION flat form, not the authored
    /// `LHS op RHS` string: `thermal_generation(0) <= 73` echoes one term and
    /// the resolved `bound_upper` — there is no authored-expression field in
    /// [`GenericConstraintEchoRow`] to diverge from.
    #[test]
    fn constant_rhs_echoes_resolved_flat_form_not_authored_string() {
        let constraints = vec![constraint(
            1,
            "cap73",
            vec![LinearTerm::literal(1.0, thermal(0))],
        )];
        let entries = vec![vec![entry(0, 1, 0, true, None, Some(73.0), None)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.term_index, Some(0));
        assert_eq!(r.variable_kind.as_deref(), Some("thermal_generation"));
        assert_eq!(r.bound_upper, Some(73.0));
        assert_eq!(r.bound_lower, None);
        assert_eq!(r.derived_shape, "cap");
    }

    /// A folded endpoint (a parquet base composed with an inline affine
    /// remainder) echoes the already-composed `f64` verbatim — the echo reads
    /// the resolved bound, with no separate base/remainder breakdown.
    #[test]
    fn folded_endpoint_echoes_composed_value() {
        let constraints = vec![constraint(
            2,
            "folded",
            vec![LinearTerm::literal(1.0, thermal(0))],
        )];
        let entries = vec![vec![entry(0, 2, 0, true, None, Some(95.0), None)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_upper, Some(95.0));
        assert_eq!(rows[0].derived_shape, "cap");
    }

    /// A `hydro_useful_volume_final` constraint's `V_lo`-folded lower bound
    /// (already composed by the layout fold before the row entry is built)
    /// reaches `GenericConstraintEchoRow.bound_lower` verbatim, not only the
    /// internal `generic_constraint_rows` layout row it is read from.
    #[test]
    fn useful_volume_folded_lower_bound_reaches_the_echo_row() {
        let hydro_ref = VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(9),
            block_id: None,
        };
        let constraints = vec![constraint(
            9,
            "useful_volume_floor",
            vec![LinearTerm::literal(1.0, hydro_ref)],
        )];
        // 20.0 (raw) + 1.0 * V_lo(9) = 32.5, already folded by the time it reaches
        // this row entry.
        let entries = vec![vec![entry(0, 9, 0, true, Some(32.5), None, None)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, Some(32.5));
        assert_eq!(rows[0].derived_shape, "floor");
        assert_eq!(
            rows[0].variable_kind.as_deref(),
            Some("hydro_useful_volume_final")
        );
    }

    /// A two-endpoint per-block entry (the d54 shape) renders a `band` row with
    /// `block_id = Some(block_idx)` and both bounds carried through.
    #[test]
    fn band_shape_per_block_row() {
        let constraints = vec![constraint(
            54,
            "d54",
            vec![LinearTerm::literal(1.0, thermal(3))],
        )];
        let entries = vec![vec![entry(0, 54, 1, false, Some(35.0), Some(100.0), None)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[2],
            &coefficients(&constraints, &resolved, &[2], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.derived_shape, "band");
        assert_eq!(r.block_id, Some(1));
        assert_eq!(r.bound_lower, Some(35.0));
        assert_eq!(r.bound_upper, Some(100.0));
        assert!(!r.slack_enabled);
        assert_eq!(r.slack_penalty, None);
        assert_eq!(r.slack_penalty_basis, None);
    }

    /// A `param * scale` coefficient resolves to `resolved.get(id, stage, block) *
    /// scale`, matching `fill_generic_constraint_entries`, at each stage.
    #[test]
    fn parameter_coefficient_resolves_like_lp() {
        let scale = 2.5;
        let constraints = vec![constraint(
            1,
            "p",
            vec![LinearTerm {
                coefficient: ScalarExpr::param(EntityId(42)) * ScalarExpr::literal(scale),
                variable: thermal(0),
            }],
        )];
        let entries = vec![
            vec![entry(0, 1, 0, true, Some(0.0), None, None)],
            vec![entry(0, 1, 0, true, Some(0.0), None, None)],
        ];
        // Two stages, one block each (length-1 inner), so any block broadcasts.
        let resolved = ResolvedParameters {
            per_param: vec![vec![vec![2.0], vec![3.0]]],
            id_to_slot: vec![(42, 0)],
            ..Default::default()
        };

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0, 1],
            &coefficients(&constraints, &resolved, &[0, 1], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(
            rows[0].coefficient,
            Some(resolved.get(EntityId(42), 0, 0) * scale)
        );
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(
            rows[1].coefficient,
            Some(resolved.get(EntityId(42), 1, 0) * scale)
        );
    }

    /// A block-varying (`PerStageBlock`) coefficient echoes its own block's value:
    /// the block-0 and block-1 rows carry the distinct per-block coefficients, not
    /// a single stage-level one.
    #[test]
    fn parameter_coefficient_reads_the_rows_own_block() {
        let constraints = vec![constraint(
            1,
            "per_block",
            vec![LinearTerm {
                coefficient: ScalarExpr::param(EntityId(42)),
                variable: thermal(0),
            }],
        )];
        // One stage, two per-block rows (is_stage_level = false) at blocks 0 and 1.
        let entries = vec![vec![
            entry(0, 1, 0, false, Some(0.0), None, None),
            entry(0, 1, 1, false, Some(0.0), None, None),
        ]];
        // Slot 0, stage 0 carries two blocks: [10.0, 20.0].
        let resolved = ResolvedParameters {
            per_param: vec![vec![vec![10.0, 20.0]]],
            id_to_slot: vec![(42, 0)],
            ..Default::default()
        };

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 2);
        let block0 = rows.iter().find(|r| r.block_id == Some(0)).unwrap();
        let block1 = rows.iter().find(|r| r.block_id == Some(1)).unwrap();
        assert_eq!(block0.coefficient, Some(10.0));
        assert_eq!(block1.coefficient, Some(20.0));
        assert_ne!(
            block0.coefficient, block1.coefficient,
            "a block-varying coefficient must differ between the block-0 and block-1 rows"
        );
    }

    /// A parameter coefficient that resolves to `0.0` at one stage still echoes its row.
    #[test]
    fn term_resolving_to_zero_emits_its_row() {
        let constraints = vec![constraint(
            1,
            "zero_at_stage_1",
            vec![
                LinearTerm {
                    coefficient: ScalarExpr::param(EntityId(42)),
                    variable: thermal(0),
                },
                LinearTerm {
                    coefficient: ScalarExpr::literal(1.0),
                    variable: thermal(1),
                },
            ],
        )];
        let entries = vec![
            vec![entry(0, 1, 0, true, Some(0.0), None, None)],
            vec![entry(0, 1, 0, true, Some(0.0), None, None)],
        ];
        let resolved = ResolvedParameters {
            per_param: vec![vec![vec![1.0], vec![0.0]]],
            id_to_slot: vec![(42, 0)],
            ..Default::default()
        };

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0, 1],
            &coefficients(&constraints, &resolved, &[0, 1], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 4);
        let param_row = |stage_id: i32| {
            rows.iter()
                .find(|r| r.stage_id == stage_id && r.term_index == Some(0))
                .unwrap()
        };
        assert_eq!(param_row(1).coefficient, Some(0.0));
        assert_eq!(param_row(0).coefficient, Some(1.0));
    }

    /// Two variable names that address one LP column echo as two separate rows.
    #[test]
    fn aliased_references_stay_separate_rows() {
        let constraints = vec![constraint(
            3,
            "storage_alias",
            vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorage {
                        hydro_id: EntityId(3),
                    },
                ),
                LinearTerm::literal(
                    -1.0,
                    VariableRef::HydroStorageFinal {
                        hydro_id: EntityId(3),
                        block_id: None,
                    },
                ),
            ],
        )];
        let entries = vec![vec![entry(0, 3, 0, true, Some(0.0), None, None)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].term_index, Some(0));
        assert_eq!(rows[0].variable_kind.as_deref(), Some("hydro_storage"));
        assert_eq!(rows[0].coefficient, Some(1.0));
        assert_eq!(rows[1].term_index, Some(1));
        assert_eq!(
            rows[1].variable_kind.as_deref(),
            Some("hydro_storage_final")
        );
        assert_eq!(rows[1].coefficient, Some(-1.0));
    }

    /// Two `constraints` slices differing only in element order yield
    /// element-for-element identical rows (declaration-order invariance).
    #[test]
    fn declaration_order_invariance_two_orderings() {
        let c1 = constraint(1, "one", vec![LinearTerm::literal(1.0, thermal(0))]);
        let c2 = constraint(
            2,
            "two",
            vec![LinearTerm::literal(
                2.0,
                VariableRef::HydroStorage {
                    hydro_id: EntityId(5),
                },
            )],
        );
        let resolved = ResolvedParameters::default();

        let constraints_a = vec![c1.clone(), c2.clone()];
        let entries_a = vec![vec![
            entry(0, 1, 0, true, None, Some(10.0), None),
            entry(1, 2, 0, true, Some(5.0), None, None),
        ]];

        // Reverse the constraint order AND the entry order, with each entry's
        // `constraint_idx` re-pointed so it still names its own logical constraint.
        let constraints_b = vec![c2, c1];
        let entries_b = vec![vec![
            entry(0, 2, 0, true, Some(5.0), None, None),
            entry(1, 1, 0, true, None, Some(10.0), None),
        ]];

        let rows_a = build_echo_rows_from_parts(
            &entries_a,
            &[0],
            &coefficients(&constraints_a, &resolved, &[0], &entries_a),
            &constraints_a,
        );
        let rows_b = build_echo_rows_from_parts(
            &entries_b,
            &[0],
            &coefficients(&constraints_b, &resolved, &[0], &entries_b),
            &constraints_b,
        );

        assert_eq!(
            format!("{rows_a:?}"),
            format!("{rows_b:?}"),
            "reordering the constraints must not change the emitted rows"
        );
    }

    /// A constant-only (empty-`terms`) constraint yields exactly one placeholder
    /// row per `(stage, block)`, with every per-term column `None`.
    #[test]
    fn empty_terms_constraint_emits_one_placeholder_per_block() {
        let constraints = vec![constraint(7, "const_only", vec![])];
        let entries = vec![vec![
            entry(0, 7, 0, false, Some(5.0), None, None),
            entry(0, 7, 1, false, Some(5.0), None, None),
        ]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(
            &entries,
            &[0],
            &coefficients(&constraints, &resolved, &[0], &entries),
            &constraints,
        );

        assert_eq!(rows.len(), 2, "one placeholder row per (stage, block)");
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.block_id, Some(i as i32));
            assert_eq!(r.term_index, None);
            assert_eq!(r.variable_kind, None);
            assert_eq!(r.variable, None);
            assert_eq!(r.coefficient, None);
            assert_eq!(r.derived_shape, "floor");
        }
    }

    /// The four reachable null-patterns map to their shape labels.
    #[test]
    fn derived_shape_covers_each_null_pattern() {
        assert_eq!(derived_shape(Some(1.0), None), "floor");
        assert_eq!(derived_shape(None, Some(1.0)), "cap");
        assert_eq!(derived_shape(Some(35.0), Some(100.0)), "band");
        assert_eq!(derived_shape(Some(50.0), Some(50.0)), "equality");
    }

    /// The layout holds a per-solve side at the infinity of its side, which the shape counts as a
    /// present side and never as equal to the other.
    #[test]
    fn a_per_solve_side_counts_in_the_shape_as_the_infinity_of_its_side() {
        assert_eq!(derived_shape(Some(f64::NEG_INFINITY), None), "floor");
        assert_eq!(derived_shape(None, Some(f64::INFINITY)), "cap");
        assert_eq!(derived_shape(Some(35.0), Some(f64::INFINITY)), "band");
        assert_eq!(derived_shape(Some(f64::NEG_INFINITY), Some(100.0)), "band");
        assert_eq!(
            derived_shape(Some(f64::NEG_INFINITY), Some(f64::INFINITY)),
            "band"
        );
    }

    /// Rendered labels match the canonical `kind(args)` form, including `block=all`
    /// for a `None` block and a bus selector on a split plant.
    #[test]
    fn render_variable_labels_are_canonical() {
        let (kind, r) = render_variable(&VariableRef::ThermalGeneration {
            thermal_id: EntityId(0),
            block_id: None,
        });
        assert_eq!(kind, "thermal_generation");
        assert_eq!(r, "thermal_generation(id=0, block=all)");

        let (_, r) = render_variable(&VariableRef::HydroStorage {
            hydro_id: EntityId(66),
        });
        assert_eq!(r, "hydro_storage(id=66)");

        let (_, r) = render_variable(&VariableRef::HydroGeneration {
            hydro_id: EntityId(66),
            block_id: Some(2),
            bus_id: Some(EntityId(1)),
        });
        assert_eq!(r, "hydro_generation(id=66, bus=1, block=2)");

        let (kind, r) = render_variable(&VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId(66),
            block_id: None,
        });
        assert_eq!(kind, "hydro_useful_volume_initial");
        assert_eq!(r, "hydro_useful_volume_initial(id=66, block=all)");

        let (kind, r) = render_variable(&VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(66),
            block_id: Some(2),
        });
        assert_eq!(kind, "hydro_useful_volume_final");
        assert_eq!(r, "hydro_useful_volume_final(id=66, block=2)");

        let (kind, r) = render_variable(&VariableRef::HydroOutflowVolume {
            hydro_id: EntityId(66),
        });
        assert_eq!(kind, "hydro_outflow_volume");
        assert_eq!(r, "hydro_outflow_volume(id=66)");

        let (kind, r) = render_variable(&VariableRef::HydroDiversionVolume {
            hydro_id: EntityId(66),
        });
        assert_eq!(kind, "hydro_diversion_volume");
        assert_eq!(r, "hydro_diversion_volume(id=66)");

        let (kind, r) = render_variable(&VariableRef::PumpingVolume {
            station_id: EntityId(3),
        });
        assert_eq!(kind, "pumping_volume");
        assert_eq!(r, "pumping_volume(id=3)");

        let (kind, r) = render_variable(&VariableRef::HydroGroupGeneration {
            hydro_id: EntityId(14),
            group_id: EntityId(0),
            block_id: None,
        });
        assert_eq!(kind, "hydro_generation");
        assert_eq!(r, "hydro_generation(id=14, unit_group=0, block=all)");

        let (kind, r) = render_variable(&VariableRef::HydroAvailability {
            hydro_id: EntityId(14),
            block_id: Some(1),
        });
        assert_eq!(kind, "hydro_availability");
        assert_eq!(r, "hydro_availability(id=14, block=1)");
    }
}
