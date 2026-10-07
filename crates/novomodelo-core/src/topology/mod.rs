//! Topology representations for the hydro cascade structure.
//!
//! The topology sub-module defines a resolved, validated representation of
//! the hydro cascade chain, built during case loading and stored on the
//! [`crate::system`] struct.

pub mod cascade;

pub use cascade::CascadeTopology;
