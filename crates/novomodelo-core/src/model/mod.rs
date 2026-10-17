//! Resolved and scenario data model.
//!
//! Groups the temporal structure ([`temporal`]), the horizon-graph topology built
//! from it ([`horizon`]), per-stage resolved bounds and penalties ([`resolved`]),
//! scenario sources ([`scenario`]), the supporting parameter ([`parameters`])
//! and penalty-resolution ([`penalty`]) types, and scalar coefficient programs
//! ([`scalar_expr`]).

pub mod horizon;
pub mod parameters;
pub mod penalty;
pub mod post_study;
pub mod resolved;
pub mod scalar_expr;
pub mod scenario;
pub mod temporal;
