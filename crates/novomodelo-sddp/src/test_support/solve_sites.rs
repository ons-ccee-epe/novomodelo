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

macro_rules! solve_site_kinds {
    ($($(#[$meta:meta])* $variant:ident,)+) => {
        /// One production solve site, in the mode its cut rows are configured for.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum SolveSiteKind {
            $($(#[$meta])* $variant,)+
        }

        impl SolveSiteKind {
            /// Every kind, in declaration order.
            pub const ALL: [SolveSiteKind; [$(SolveSiteKind::$variant),+].len()] =
                [$(SolveSiteKind::$variant),+];
        }
    };
}

solve_site_kinds! {
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
