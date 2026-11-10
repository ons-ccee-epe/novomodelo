//! The production solve sites by kind, each with the shape of the cut rows it loads.
//!
//! The per-shape dual oracles and the full-run cut-row recorder match on these kinds. The table
//! has no projection column: every kind that extracts a cut reads the successor pool's projection
//! for its resident rows and its own pool's for the extracted cut, so whether the two differ is a
//! property of the study.

use SolveSiteKind::{
    BackwardByNode, BackwardByScenario, BackwardReplicated, DynamicCutSelection, EnumeratedForward,
    FusedTerminal, LowerBound, SampledForward, Simulation,
};

use crate::solve::StageOpeningSolver;
use crate::training::stage_solve_prep::{BracketSource, NoiseTermSource};

/// One production solve site, in the mode its cut rows are configured for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolveSiteKind {
    /// The sampled forward pass's stage solve.
    SampledForward,
    /// The enumerated forward pass's node solve.
    EnumeratedForward,
    /// The terminal leaf's forward solve, which also extracts the leaf's cut for its parent.
    FusedTerminal,
    /// The simulation's stage solve.
    Simulation,
    /// The first-stage lower-bound evaluation.
    LowerBound,
    /// The backward pass scheduled by scenario.
    BackwardByScenario,
    /// The backward pass scheduled by node.
    BackwardByNode,
    /// The replicated backward solve of one node.
    BackwardReplicated,
    /// The lazy solve over the dynamic cut selection's resident set, at any site that has one.
    DynamicCutSelection,
}

/// Which cut rows a kind's loaded LP holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutRowShape {
    /// The frozen rows only.
    Frozen,
    /// The frozen rows, then the rows appended since the freeze.
    FrozenPlusDelta,
    /// Every row of the lower bound's cut-row map, deactivated rows included.
    LowerBoundMap,
    /// The dynamic cut selection's resident set.
    DynamicResidentSet,
}

impl SolveSiteKind {
    /// Every kind, in declaration order.
    pub const ALL: [SolveSiteKind; 9] = [
        SampledForward,
        EnumeratedForward,
        FusedTerminal,
        Simulation,
        LowerBound,
        BackwardByScenario,
        BackwardByNode,
        BackwardReplicated,
        DynamicCutSelection,
    ];

    /// The cut-row shape of this kind's loaded LP.
    #[must_use]
    pub fn cut_row_shape(self) -> CutRowShape {
        match self {
            SampledForward | EnumeratedForward | FusedTerminal | Simulation => CutRowShape::Frozen,
            BackwardByScenario | BackwardByNode | BackwardReplicated => {
                CutRowShape::FrozenPlusDelta
            }
            LowerBound => CutRowShape::LowerBoundMap,
            DynamicCutSelection => CutRowShape::DynamicResidentSet,
        }
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the coverage test only; the exhaustive match is the compile-time check"
    )
)]
pub(crate) fn kinds_reaching_opening_solver(
    solver: &StageOpeningSolver,
) -> &'static [SolveSiteKind] {
    match solver {
        StageOpeningSolver::Frozen => &[
            SampledForward,
            EnumeratedForward,
            FusedTerminal,
            Simulation,
            LowerBound,
            BackwardByScenario,
            BackwardByNode,
            BackwardReplicated,
        ],
        StageOpeningSolver::Lazy(_) => &[DynamicCutSelection],
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the coverage test only; the exhaustive match is the compile-time check"
    )
)]
pub(crate) fn kinds_reaching_bracket_source(
    source: &BracketSource<'_>,
) -> &'static [SolveSiteKind] {
    match source {
        BracketSource::Computed => &[
            SampledForward,
            EnumeratedForward,
            FusedTerminal,
            Simulation,
            LowerBound,
            BackwardReplicated,
            DynamicCutSelection,
        ],
        BracketSource::Slab(_) => &[BackwardByScenario, BackwardByNode],
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by the coverage test only; the exhaustive match is the compile-time check"
    )
)]
pub(crate) fn kinds_reaching_noise_term_source(
    source: &NoiseTermSource<'_>,
) -> &'static [SolveSiteKind] {
    match source {
        NoiseTermSource::OnTheFly => &[
            SampledForward,
            EnumeratedForward,
            FusedTerminal,
            Simulation,
            LowerBound,
            BackwardByScenario,
            BackwardByNode,
            BackwardReplicated,
            DynamicCutSelection,
        ],
        NoiseTermSource::Stage { .. } => &[BackwardByScenario, BackwardByNode, DynamicCutSelection],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dcs::DcsParams;

    fn assert_every_kind_is_reached(dispatch: &str, reached: &[&[SolveSiteKind]]) {
        for kind in SolveSiteKind::ALL {
            assert!(
                reached.iter().any(|kinds| kinds.contains(&kind)),
                "{kind:?} is reached by no variant of {dispatch}"
            );
        }
    }

    #[test]
    fn every_solve_site_kind_is_reached_by_each_dispatch_enum() {
        let frozen = kinds_reaching_opening_solver(&StageOpeningSolver::Frozen);
        let lazy = kinds_reaching_opening_solver(&StageOpeningSolver::Lazy(DcsParams {
            candidate_recency: None,
            seed_window: 1,
            max_added_per_round: 1,
            violation_tolerance: 1e-6,
            start_iteration: 1,
            max_inner_iterations: 1,
        }));
        assert_every_kind_is_reached("StageOpeningSolver", &[frozen, lazy]);
        assert_every_kind_is_reached(
            "BracketSource",
            &[
                kinds_reaching_bracket_source(&BracketSource::Computed),
                kinds_reaching_bracket_source(&BracketSource::Slab(None)),
            ],
        );
        assert_every_kind_is_reached(
            "NoiseTermSource",
            &[
                kinds_reaching_noise_term_source(&NoiseTermSource::OnTheFly),
                kinds_reaching_noise_term_source(&NoiseTermSource::Stage {
                    noise_term: &[],
                    cut_noise: &[],
                }),
            ],
        );

        for kind in SolveSiteKind::ALL {
            assert_eq!(
                lazy.contains(&kind),
                kind.cut_row_shape() == CutRowShape::DynamicResidentSet,
                "{kind:?}: reached by Lazy exactly when its cut rows are the resident set"
            );
        }
        assert!(
            frozen.contains(&FusedTerminal) && !lazy.contains(&FusedTerminal),
            "FusedTerminal is reached by Frozen only"
        );
    }
}
