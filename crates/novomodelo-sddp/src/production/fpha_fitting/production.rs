//! Production-function model for the FPHA fitting pipeline.
//!
//! Owns the head-conversion constant [`K`] and [`ProductionFunction`], the
//! evaluable bundle of forebay table + tailrace + hydraulic-loss + efficiency that
//! computes `phi(v, q, lateral_flow)`.

use novomodelo_core::{EfficiencyModel, HydraulicLossesModel, TailraceModel};

use super::geometry::{ForebayTable, evaluate_losses, evaluate_tailrace};
use super::tailrace::TailraceFamilies;

/// Source of the tailrace elevation `tailrace_level(outflow)` the production
/// function reads.
///
/// # Contract — the `Entity` arm is the inert fallback
///
/// A plant without a `tailrace_curves` table resolves to [`TailraceSource::Entity`]
/// and must fit bit-for-bit to the same planes as before the families path
/// existed: `net_head` evaluates the same
/// `evaluate_tailrace(model, own_outflow_factor·q + lateral_flow)`.
/// Routing a table-less plant through the `Families` arm would silently perturb
/// its fitted γ values and is forbidden.
#[derive(Debug, Clone)]
pub(crate) enum TailraceSource {
    /// The entity-level [`TailraceModel`]; `None` means constant zero tailrace.
    Entity(Option<TailraceModel>),
    /// Piecewise-quartic backwater-coupled families for a plant with a
    /// `tailrace_curves` table.
    Families {
        /// The plant's ordered, validated tailrace families.
        families: TailraceFamilies,
        /// Resolved downstream reference level (m) keying the backwater coupling
        /// (`None` ⇒ the evaluator falls back to the lowest-keyed family).
        downstream_level_m: Option<f64>,
    },
}

/// Gravity times water density over unit conversion `g·ρ / 1000 = 9.81e-3`,
/// converting `phi = K·eta·q·h_net` from hydraulic power (W) to MW.
pub(super) const K: f64 = 9.81 / 1000.0;

/// Complete hydro production function `phi(v, q, lateral_flow)` with analytical derivatives —
/// the forebay table, optional tailrace and loss models, and a constant turbine
/// efficiency bundled into one evaluable object yielding power in MW. All
/// validation is upstream; [`Self::new`] is infallible.
///
/// Both `net_head` and `evaluate` take `(v, q, lateral_flow)`: reservoir volume
/// \[hm³\], turbined flow \[m³/s\], tailrace lateral flow \[m³/s\] — the own
/// spill when no tailrace-flow declaration applies. The tailrace is read at
/// `own_outflow_factor·q + lateral_flow`; hydraulic losses stay on `q`.
#[derive(Debug, Clone)]
pub(crate) struct ProductionFunction {
    /// Forebay height interpolation table.
    forebay: ForebayTable,
    /// Tailrace elevation source.
    tailrace: TailraceSource,
    /// Hydraulic losses model. `None` means lossless penstock.
    hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine efficiency (dimensionless, in `(0, 1]`); `1.0` when the entity has
    /// no `EfficiencyModel`.
    efficiency: f64,
    /// Weight of the own turbined flow `q` in the tailrace flow (`1.0` by default).
    own_outflow_factor: f64,
    /// Maximum turbined flow \[m³/s\].
    pub(crate) max_turbined_m3s: f64,
    /// Installed-capacity ceiling \[MW\] that [`Self::evaluate_capped`] clips to
    /// `[0, max_generation_mw]`. `f64::INFINITY` disables the ceiling (the
    /// [`Self::new`] default); [`Self::with_max_generation_mw`] sets it.
    max_generation_mw: f64,
    /// Human-readable plant name for error messages.
    // Rationale: populated and read in integration-test diagnostics; production
    // code sets a name but never reads it back.
    #[allow(dead_code)]
    pub(crate) hydro_name: String,
}

impl ProductionFunction {
    /// Build a [`ProductionFunction`] from component models. `efficiency` supports
    /// only [`EfficiencyModel::Constant`]; `None` = 1.0.
    pub(crate) fn new(
        forebay: ForebayTable,
        tailrace: TailraceSource,
        hydraulic_losses: Option<&HydraulicLossesModel>,
        efficiency: Option<&EfficiencyModel>,
        max_turbined_m3s: f64,
        hydro_name: String,
    ) -> Self {
        let efficiency_value = match efficiency {
            Some(EfficiencyModel::Constant { value }) => *value,
            None => 1.0,
        };
        Self {
            forebay,
            tailrace,
            hydraulic_losses: hydraulic_losses.copied(),
            efficiency: efficiency_value,
            own_outflow_factor: 1.0,
            max_turbined_m3s,
            max_generation_mw: f64::INFINITY,
            hydro_name,
        }
    }

    /// Set the installed-capacity ceiling \[MW\].
    pub(crate) fn with_max_generation_mw(mut self, max_generation_mw: f64) -> Self {
        self.max_generation_mw = max_generation_mw;
        self
    }

    /// Set the weight of the own turbined flow in the tailrace flow.
    pub(crate) fn with_own_outflow_factor(mut self, factor: f64) -> Self {
        self.own_outflow_factor = factor;
        self
    }

    /// Flow that drives the tailrace \[m³/s\]: `own_outflow_factor·q + lateral_flow`.
    pub(crate) fn tailrace_flow(&self, q: f64, lateral_flow: f64) -> f64 {
        self.own_outflow_factor * q + lateral_flow
    }

    /// Tailrace elevation \[m\] at `tailrace_flow`.
    pub(crate) fn tailrace_level(&self, tailrace_flow: f64) -> f64 {
        // The `Entity` arm MUST call `evaluate_tailrace(model, tailrace_flow)` verbatim —
        // the inert-fallback contract on [`TailraceSource`].
        match &self.tailrace {
            TailraceSource::Entity(model) => model
                .as_ref()
                .map_or(0.0, |m| evaluate_tailrace(m, tailrace_flow)),
            TailraceSource::Families {
                families,
                downstream_level_m,
            } => families.evaluate(tailrace_flow, *downstream_level_m),
        }
    }

    /// Net head at the turbine \[m\]:
    /// `h_net = h_fore(v) − h_tail(own_outflow_factor·q + lateral_flow) −
    /// h_loss(gross_head, q)`, clamped to
    /// `max(0.0, h_net)` (negative net head is physically impossible). The
    /// [`HydraulicLossesModel::Factor`] variant simplifies to
    /// `(1 − k)·(h_fore − h_tail)`.
    pub(crate) fn net_head(&self, v: f64, q: f64, lateral_flow: f64) -> f64 {
        let h_fore = self.forebay.height(v);
        let h_tail = self.tailrace_level(self.tailrace_flow(q, lateral_flow));
        let gross_head = h_fore - h_tail;
        let h_loss = self
            .hydraulic_losses
            .as_ref()
            .map_or(0.0, |m| evaluate_losses(m, gross_head, q));
        let h_net = gross_head - h_loss;
        h_net.max(0.0)
    }

    /// Power output \[MW\], uncapped: `phi = K·eta·q·h_net`.
    ///
    /// The installed-capacity ceiling is NOT applied here (see
    /// [`Self::evaluate_capped`]): the lateral secant fits on this uncapped value so
    /// lateral-flow sensitivity is read from the raw head curve, not flattened where the
    /// ceiling binds.
    pub(crate) fn evaluate(&self, v: f64, q: f64, lateral_flow: f64) -> f64 {
        let h_net = self.net_head(v, q, lateral_flow);
        K * self.efficiency * q * h_net
    }

    /// Production clipped to the installed-capacity ceiling \[MW\]:
    /// `min(evaluate(v, q, lateral_flow), max_generation_mw)`.
    ///
    /// The cloud and α regression use this so the fitted envelope gains a flat
    /// `generation ≤ max_generation_mw` facet; the lateral secant uses the uncapped
    /// [`Self::evaluate`] instead.
    pub(crate) fn evaluate_capped(&self, v: f64, q: f64, lateral_flow: f64) -> f64 {
        self.evaluate(v, q, lateral_flow)
            .min(self.max_generation_mw)
    }
}
