//! The per-stage inputs simulation needs to evaluate each hydro's tailrace.

use std::sync::Arc;

use novomodelo_core::Hydro;

use crate::fpha_fitting::ProductionFunction;
use crate::hydro_models::ProductionModelSet;
use crate::lp::builder::ResolvedTailraceLateral;
use crate::setup::lp_build_inputs::resolve_tailrace_lateral;

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
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no reader until simulation extracts the tailrace quantities"
        )
    )]
    pub(crate) fn stage(&self, t: usize) -> &[Option<Arc<ProductionFunction>>] {
        self.production_functions.get(t).map_or(&[], Vec::as_slice)
    }

    /// Each hydro's lateral terms, in system order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no reader until simulation extracts the tailrace quantities"
        )
    )]
    pub(crate) fn lateral(&self) -> &[ResolvedTailraceLateral] {
        &self.lateral
    }
}

#[cfg(test)]
mod tests {
    use novomodelo_core::entities::hydro::TailraceFlow;
    use novomodelo_io::extensions::HydroGeometryRow;

    use super::*;
    use crate::fpha_fitting::{ForebayTable, TailraceSource};
    use crate::hydro_models::ResolvedProductionModel;
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
}
