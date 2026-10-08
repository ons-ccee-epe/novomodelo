//! Constraint definitions and run-time condition records.
//!
//! Groups the user-defined generic linear constraints ([`generic_constraint`]),
//! the study's starting conditions ([`initial_conditions`]), and the per-step
//! event records emitted by iterative runners ([`training_event`]).

pub mod generic_constraint;
pub mod initial_conditions;
pub mod training_event;
