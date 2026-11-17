//! The per-stage inputs simulation needs to evaluate each hydro's tailrace.

use std::sync::Arc;

use novomodelo_core::{Hydro, is_exact_zero};

use crate::fpha_fitting::ProductionFunction;
use crate::hydro_models::ProductionModelSet;
use crate::inflow_method::InflowNonNegativityMethod;
use crate::lp::builder::ResolvedTailraceLateral;
use crate::setup::lp_build_inputs::resolve_tailrace_lateral;
use crate::simulation::types::SimulationHydroResult;

/// Each hydro's fitted production function per stage and its tailrace lateral
/// terms, resolved by the stage LP's own resolver.
#[derive(Debug, Clone, Default)]
pub struct SimulationTailrace {
    production_functions: Vec<Vec<Option<Arc<ProductionFunction>>>>,
    lateral: Vec<ResolvedTailraceLateral>,
}

impl SimulationTailrace {
    /// Copies each hydro's production-function handles stage-major (`[t][h]`)
    /// and resolves its lateral terms.
    pub(crate) fn build(production: &ProductionModelSet, hydros: &[Hydro]) -> Self {
        let production_functions = (0..production.n_stages())
            .map(|t| {
                (0..hydros.len())
                    .map(|h| production.fpha_production_function(h, t).cloned())
                    .collect()
            })
            .collect();
        Self {
            production_functions,
            lateral: resolve_tailrace_lateral(hydros),
        }
    }

    /// Stage `t`'s production function per hydro; empty out of range.
    pub(crate) fn stage(&self, t: usize) -> &[Option<Arc<ProductionFunction>>] {
        self.production_functions.get(t).map_or(&[], Vec::as_slice)
    }

    /// Each hydro's lateral terms, in system order.
    pub(crate) fn lateral(&self) -> &[ResolvedTailraceLateral] {
        &self.lateral
    }
}

/// Writes the tailrace flow, tailrace level and net head of every hydro with a
/// production function into its `n_blks` rows, from the same lateral terms the
/// stage LP's plane rows carry. Hydros without a function keep `None`.
pub(crate) fn fill_tailrace_quantities(
    hydros: &mut [SimulationHydroResult],
    n_blks: usize,
    stage_functions: &[Option<Arc<ProductionFunction>>],
    lateral: &[ResolvedTailraceLateral],
    inflow_method: InflowNonNegativityMethod,
) {
    if stage_functions.is_empty() {
        return;
    }
    debug_assert_eq!(hydros.len(), stage_functions.len() * n_blks);
    let truncates = inflow_method.truncates();
    for (h, function) in stage_functions.iter().enumerate() {
        let Some(function) = function else {
            continue;
        };
        let terms = &lateral[h];
        for b in 0..n_blks {
            let own = &hydros[h * n_blks + b];
            let q = own.turbined_m3s;
            let v = f64::midpoint(own.storage_initial_hm3, own.storage_final_hm3);
            let mut lateral_flow = terms.own_outflow_factor * own.spillage_m3s;
            for &(j, factor) in &terms.outflow {
                if is_exact_zero(factor) {
                    continue;
                }
                let release = &hydros[j.get() * n_blks + b];
                lateral_flow += factor * (release.turbined_m3s + release.spillage_m3s);
            }
            for &(k, factor) in &terms.incremental_inflow {
                if is_exact_zero(factor) {
                    continue;
                }
                let source = &hydros[k.get() * n_blks + b];
                let inflow = if truncates {
                    source.incremental_inflow_m3s + source.inflow_nonnegativity_slack_m3s
                } else {
                    source.incremental_inflow_m3s
                };
                lateral_flow += factor * inflow;
            }
            let tailrace_flow = function.tailrace_flow(q, lateral_flow);
            let net_head = function.net_head(v, q, lateral_flow);
            let row = &mut hydros[h * n_blks + b];
            row.tailrace_flow_m3s = Some(tailrace_flow);
            row.tailrace_level_m = Some(function.tailrace_level(tailrace_flow));
            row.net_head_m = Some(net_head);
        }
    }
}

#[cfg(test)]
mod tests {
    use novomodelo_core::entities::hydro::TailraceFlow;
    use novomodelo_core::{HydraulicLossesModel, TailraceModel};
    use novomodelo_io::extensions::HydroGeometryRow;

    use super::*;
    use crate::fpha_fitting::{ForebayTable, TailraceSource};
    use crate::hydro_models::ResolvedProductionModel;
    use crate::lp::indexer::HydroSys;
    use crate::test_support::minimal_hydros;

    const N_STAGES: usize = 2;

    fn production_function() -> Arc<ProductionFunction> {
        let rows: Vec<HydroGeometryRow> = [(0.0, 380.0), (1_000.0, 400.0)]
            .into_iter()
            .map(|(volume_hm3, height_m)| HydroGeometryRow {
                hydro_id: 0.into(),
                volume_hm3,
                height_m,
                area_km2: 0.0,
            })
            .collect();
        Arc::new(
            ProductionFunction::new(
                ForebayTable::new(&rows, "H0").expect("valid VHA curve"),
                TailraceSource::Entity(None),
                None,
                None,
                500.0,
                "H0".to_owned(),
            )
            .with_own_outflow_factor(1.2),
        )
    }

    #[test]
    fn build_keeps_stage_major_functions_and_lateral() {
        let mut hydros = minimal_hydros(1);
        hydros[0].tailrace_flow = TailraceFlow {
            own_outflow_factor: 1.2,
            ..TailraceFlow::default()
        };
        let function = production_function();
        let production = ProductionModelSet::new(
            vec![vec![
                ResolvedProductionModel::ConstantProductivity {
                    productivity: 1.0
                };
                N_STAGES
            ]],
            &hydros,
            N_STAGES,
        )
        .with_fpha_production_functions(vec![vec![Some(Arc::clone(&function)); N_STAGES]]);

        let lookup = SimulationTailrace::build(&production, &hydros);

        assert_eq!(lookup.stage(0).len(), 1);
        assert!(lookup.stage(0)[0].is_some());
        assert_eq!(lookup.lateral()[0].own_outflow_factor, 1.2);
        assert!(lookup.stage(99).is_empty());
    }

    #[test]
    fn default_lookup_is_empty() {
        let lookup = SimulationTailrace::default();

        assert!(lookup.stage(0).is_empty());
        assert!(lookup.lateral().is_empty());
    }

    fn closed_form_function() -> Arc<ProductionFunction> {
        let rows: Vec<HydroGeometryRow> = [(0.0, 100.0), (1_000.0, 200.0)]
            .into_iter()
            .map(|(volume_hm3, height_m)| HydroGeometryRow {
                hydro_id: 0.into(),
                volume_hm3,
                height_m,
                area_km2: 0.0,
            })
            .collect();
        Arc::new(
            ProductionFunction::new(
                ForebayTable::new(&rows, "H0").expect("valid VHA curve"),
                TailraceSource::Entity(Some(TailraceModel::Polynomial {
                    coefficients: vec![10.0, 0.001],
                })),
                Some(&HydraulicLossesModel::Constant { value_m: 1.0 }),
                None,
                500.0,
                "H0".to_owned(),
            )
            .with_own_outflow_factor(1.2),
        )
    }

    fn lateral(outflow_factor: f64) -> Vec<ResolvedTailraceLateral> {
        vec![
            ResolvedTailraceLateral {
                own_outflow_factor: 1.2,
                outflow: vec![(HydroSys::new(1), outflow_factor)],
                incremental_inflow: vec![(HydroSys::new(0), 0.3)],
            },
            ResolvedTailraceLateral::undeclared(),
        ]
    }

    fn hydro_row(
        turbined: f64,
        spillage: f64,
        storage: (f64, f64),
        inflow: f64,
        slack: f64,
    ) -> SimulationHydroResult {
        SimulationHydroResult {
            stage_id: 0,
            block_id: Some(0),
            hydro_id: 0,
            turbined_m3s: turbined,
            spillage_m3s: spillage,
            evaporation_m3s: None,
            diverted_inflow_m3s: None,
            diverted_outflow_m3s: None,
            incremental_inflow_m3s: inflow,
            inflow_m3s: inflow,
            storage_initial_hm3: storage.0,
            storage_final_hm3: storage.1,
            generation_mw: 0.0,
            equivalent_productivity_mw_per_m3s: 0.0,
            accumulated_productivity_mw_per_m3s: 0.0,
            incremental_inflow_energy_mw: 0.0,
            stored_energy_initial_mwh: 0.0,
            stored_energy_final_mwh: 0.0,
            spillage_cost: 0.0,
            water_value_per_hm3: 0.0,
            storage_binding_code: 0,
            operative_state_code: 0,
            turbined_slack_m3s: 0.0,
            outflow_slack_below_m3s: 0.0,
            outflow_slack_above_m3s: 0.0,
            generation_slack_mw: 0.0,
            fpha_violation_mw: 0.0,
            storage_violation_below_hm3: 0.0,
            filling_target_violation_hm3: 0.0,
            evaporation_violation_pos_m3s: 0.0,
            evaporation_violation_neg_m3s: 0.0,
            inflow_nonnegativity_slack_m3s: slack,
            water_withdrawal_violation_pos_m3s: 0.0,
            water_withdrawal_violation_neg_m3s: 0.0,
            integrated_equivalent_productivity_mw_per_m3s: 0.0,
            integrated_accumulated_productivity_mw_per_m3s: 0.0,
            stored_energy_initial_mw: 0.0,
            stored_energy_final_mw: 0.0,
            tailrace_flow_m3s: None,
            tailrace_level_m: None,
            net_head_m: None,
        }
    }

    fn fixture_rows(slack: f64) -> Vec<SimulationHydroResult> {
        vec![
            hydro_row(100.0, 20.0, (400.0, 600.0), 50.0, slack),
            hydro_row(80.0, 40.0, (0.0, 0.0), 0.0, 0.0),
        ]
    }

    fn fill(
        rows: &mut [SimulationHydroResult],
        n_blks: usize,
        lateral: &[ResolvedTailraceLateral],
        method: InflowNonNegativityMethod,
    ) {
        fill_tailrace_quantities(
            rows,
            n_blks,
            &[Some(closed_form_function()), None],
            lateral,
            method,
        );
    }

    #[test]
    fn closed_form_tailrace_quantities() {
        let mut rows = fixture_rows(0.0);

        fill(&mut rows, 1, &lateral(0.5), InflowNonNegativityMethod::None);

        let flow = 1.2 * 100.0 + 1.2 * 20.0 + 0.5 * 120.0 + 0.3 * 50.0;
        let level = 10.0 + 0.001 * flow;
        assert_eq!(rows[0].tailrace_flow_m3s, Some(219.0));
        assert_eq!(rows[0].tailrace_flow_m3s, Some(flow));
        assert_eq!(rows[0].tailrace_level_m, Some(level));
        assert_eq!(rows[0].net_head_m, Some(150.0 - level - 1.0));
        assert_eq!(rows[1].tailrace_flow_m3s, None);
        assert_eq!(rows[1].tailrace_level_m, None);
        assert_eq!(rows[1].net_head_m, None);
    }

    #[test]
    fn truncation_adds_the_inflow_slack_to_the_influencing_inflow() {
        for (method, expected) in [
            (InflowNonNegativityMethod::Truncation, 219.0 + 0.3 * 5.0),
            (
                InflowNonNegativityMethod::TruncationWithPenalty,
                219.0 + 0.3 * 5.0,
            ),
            (InflowNonNegativityMethod::Penalty, 219.0),
            (InflowNonNegativityMethod::None, 219.0),
        ] {
            let mut rows = fixture_rows(5.0);

            fill(&mut rows, 1, &lateral(0.5), method);

            assert_eq!(rows[0].tailrace_flow_m3s, Some(expected), "{method:?}");
        }
    }

    #[test]
    fn a_zero_factor_skips_the_influencing_plant() {
        let mut rows = fixture_rows(0.0);
        rows[1] = hydro_row(f64::NAN, f64::NAN, (0.0, 0.0), 0.0, 0.0);

        fill(&mut rows, 1, &lateral(0.0), InflowNonNegativityMethod::None);

        assert!(rows[0].tailrace_flow_m3s.is_some_and(f64::is_finite));
    }

    #[test]
    fn an_empty_stage_leaves_every_row_untouched() {
        let lookup = SimulationTailrace::default();
        let mut rows = fixture_rows(0.0);

        fill_tailrace_quantities(
            &mut rows,
            1,
            lookup.stage(0),
            lookup.lateral(),
            InflowNonNegativityMethod::None,
        );

        assert!(rows.iter().all(|row| {
            row.tailrace_flow_m3s.is_none()
                && row.tailrace_level_m.is_none()
                && row.net_head_m.is_none()
        }));
    }

    #[test]
    fn each_block_reads_its_own_rows_of_every_plant() {
        let mut rows = vec![
            hydro_row(100.0, 20.0, (400.0, 600.0), 50.0, 0.0),
            hydro_row(60.0, 10.0, (450.0, 550.0), 30.0, 0.0),
            hydro_row(80.0, 40.0, (0.0, 0.0), 0.0, 0.0),
            hydro_row(10.0, 10.0, (0.0, 0.0), 0.0, 0.0),
        ];

        fill(&mut rows, 2, &lateral(0.5), InflowNonNegativityMethod::None);

        assert_eq!(
            rows[0].tailrace_flow_m3s,
            Some(1.2 * 100.0 + 1.2 * 20.0 + 0.5 * 120.0 + 0.3 * 50.0)
        );
        assert_eq!(
            rows[1].tailrace_flow_m3s,
            Some(1.2 * 60.0 + 1.2 * 10.0 + 0.5 * 20.0 + 0.3 * 30.0)
        );
        assert_eq!(
            rows[1].net_head_m,
            Some(150.0 - rows[1].tailrace_level_m.expect("filled") - 1.0)
        );
    }
}
