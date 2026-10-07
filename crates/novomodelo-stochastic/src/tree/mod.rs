//! Scenario tree construction.
//!
//! Builds the opening scenario tree used during the first stage of iterative
//! optimization algorithms: a branching structure of initial hydro storage states
//! and inflow realisations sampled at the start of each iteration.

pub mod generate;
pub mod lhs;
pub mod opening_tree;
pub mod point_spec;
pub mod qmc_halton;
pub mod qmc_sobol;

pub use generate::{ClassDimensions, OpeningTreeGenerationInputs, generate_opening_tree};
pub use opening_tree::{OpeningTree, OpeningTreeView};
pub use point_spec::NoisePointSpec;
