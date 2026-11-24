//! The current-format boundary input.

pub(crate) mod checks;
// Rationale: read by the schema-layer dispatch.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod file;
