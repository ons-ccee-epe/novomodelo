//! Correlation matrix construction and spectral factorisation.
//!
//! Applies spatial correlation between stochastic series during scenario
//! generation. Entity classes (inflow, load, non-controllable sources) each
//! carry an independent correlation group resolved to a dense matrix indexed by
//! internal entity IDs.

pub mod resolve;
pub mod spectral;

pub use resolve::{DecomposedCorrelation, EntityClass, GroupFactor};
pub use spectral::SpectralFactor;
