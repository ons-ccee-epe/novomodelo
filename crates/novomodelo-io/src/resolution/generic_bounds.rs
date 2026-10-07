//! Resolution of raw generic constraint bound rows into an indexed lookup table.
//!
//! [`resolve_generic_constraint_bounds`] builds a [`ResolvedGenericConstraintBounds`]
//! supporting O(1) lookup by `(constraint_idx, stage_id)`. Rows with an unknown
//! `constraint_id` are silently skipped (already caught upstream).

use std::collections::HashMap;

use cobre_core::{GenericConstraint, ResolvedGenericConstraintBounds};

use crate::constraints::GenericConstraintBoundsRow;

/// Build the resolved generic constraint bound table from sorted parsed inputs.
///
/// `constraints` must be sorted by ID (slice position becomes the constraint index);
/// `raw_bounds` must be sorted by `(constraint_id, stage_id, block_id)`.
///
/// # Examples
///
/// ```
/// use cobre_core::GenericConstraint;
/// use cobre_core::generic_constraint::{ConstraintExpression, SlackConfig};
/// use cobre_core::EntityId;
/// use cobre_io::constraints::GenericConstraintBoundsRow;
/// use cobre_io::resolution::resolve_generic_constraint_bounds;
///
/// let constraint = GenericConstraint {
///     id: EntityId(0),
///     name: "c0".to_string(),
///     description: None,
///     expression: ConstraintExpression { terms: vec![] },
///     slack: SlackConfig { enabled: false, penalty: None },
///     bound_lower_affine: None,
///     bound_upper_affine: None,
/// };
///
/// let row = GenericConstraintBoundsRow {
///     constraint_id: 0,
///     stage_id: 1,
///     block_id: None,
///     bound_lower: Some(500.0),
///     bound_upper: None,
/// };
///
/// let table = resolve_generic_constraint_bounds(&[constraint], &[row]);
///
/// assert!(table.is_active(0, 1));
/// assert!(!table.is_active(0, 0));
///
/// let slice = table.bounds_for_stage(0, 1);
/// assert_eq!(slice.len(), 1);
/// assert_eq!(slice[0].block_id, None);
/// assert_eq!(slice[0].bound_lower, Some(500.0));
/// assert_eq!(slice[0].bound_upper, None);
/// ```
#[must_use]
pub fn resolve_generic_constraint_bounds(
    constraints: &[GenericConstraint],
    raw_bounds: &[GenericConstraintBoundsRow],
) -> ResolvedGenericConstraintBounds {
    let id_to_idx: HashMap<i32, usize> = constraints
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.id.0, idx))
        .collect();

    ResolvedGenericConstraintBounds::new(
        &id_to_idx,
        raw_bounds.iter().map(|r| {
            (
                r.constraint_id,
                r.stage_id,
                r.block_id,
                r.bound_lower,
                r.bound_upper,
            )
        }),
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use cobre_core::EntityId;
    use cobre_core::generic_constraint::ConstraintExpression;
    use cobre_core::model::resolved::GenericConstraintBoundEntry;

    fn make_constraint(id: i32) -> GenericConstraint {
        use cobre_core::generic_constraint::SlackConfig;
        GenericConstraint {
            id: EntityId(id),
            name: format!("c{id}"),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    /// `bound_upper` defaults to `None` — every call site below is byte-neutral
    /// unless it opts in explicitly.
    fn make_row(
        constraint_id: i32,
        stage_id: i32,
        block_id: Option<i32>,
        bound_lower: f64,
    ) -> GenericConstraintBoundsRow {
        GenericConstraintBoundsRow {
            constraint_id,
            stage_id,
            block_id,
            bound_lower: Some(bound_lower),
            bound_upper: None,
        }
    }

    /// Empty constraints and empty bounds produce an empty table.
    #[test]
    fn test_empty_constraints_empty_bounds() {
        let table = resolve_generic_constraint_bounds(&[], &[]);
        assert!(!table.is_active(0, 0));
        assert!(table.bounds_for_stage(0, 0).is_empty());
    }

    /// Two constraints with no bound rows: `is_active` always false.
    #[test]
    fn test_constraints_no_bounds() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let table = resolve_generic_constraint_bounds(&constraints, &[]);
        assert!(!table.is_active(0, 0));
        assert!(!table.is_active(1, 0));
    }

    /// Two constraints, bounds for constraint 0 at stages 0 and 1.
    /// Constraint 1 has no bounds.
    #[test]
    fn test_two_constraints_sparse_bounds() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let rows = vec![make_row(0, 0, None, 100.0), make_row(0, 1, None, 200.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);

        assert!(table.is_active(0, 0));
        assert!(table.is_active(0, 1));
        assert!(!table.is_active(1, 0));
        assert!(!table.is_active(1, 1));

        let s0 = table.bounds_for_stage(0, 0);
        assert_eq!(s0.len(), 1);
        assert!((s0[0].bound_lower.expect("lower present") - 100.0).abs() < f64::EPSILON);
        assert!(s0[0].block_id.is_none());

        let s1 = table.bounds_for_stage(0, 1);
        assert_eq!(s1.len(), 1);
        assert!((s1[0].bound_lower.expect("lower present") - 200.0).abs() < f64::EPSILON);
    }

    /// Block-specific bounds: multiple (`block_id`, bound) pairs for one (constraint, stage).
    #[test]
    fn test_block_specific_bounds() {
        let constraints = vec![make_constraint(0)];
        let rows = vec![
            make_row(0, 0, None, 50.0),
            make_row(0, 0, Some(0), 60.0),
            make_row(0, 0, Some(1), 70.0),
        ];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);

        assert!(table.is_active(0, 0));
        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 3);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(50.0),
                bound_upper: None,
            }
        );
        assert_eq!(
            slice[1],
            GenericConstraintBoundEntry {
                block_id: Some(0),
                bound_lower: Some(60.0),
                bound_upper: None,
            }
        );
        assert_eq!(
            slice[2],
            GenericConstraintBoundEntry {
                block_id: Some(1),
                bound_lower: Some(70.0),
                bound_upper: None,
            }
        );
    }

    /// `bound_upper`, when the row carries one, flows through resolution into the
    /// resolved entry alongside `bound` — the widened pipe's actual contract.
    #[test]
    fn test_bound_upper_flows_through_resolution() {
        let constraints = vec![make_constraint(0)];
        let mut row = make_row(0, 0, None, 50.0);
        row.bound_upper = Some(90.0);
        let table = resolve_generic_constraint_bounds(&constraints, &[row]);

        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(50.0),
                bound_upper: Some(90.0),
            }
        );
    }

    /// Rows referencing unknown constraint IDs are silently skipped.
    #[test]
    fn test_unknown_constraint_id_skipped() {
        let constraints = vec![make_constraint(0)];
        let rows = vec![
            make_row(0, 0, None, 100.0),
            make_row(99, 0, None, 9999.0), // Unknown ID.
        ];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);

        assert!(table.is_active(0, 0));
        // No entry for the unknown constraint (idx 0 is constraint_id=0, not 99).
        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert!((slice[0].bound_lower.expect("lower present") - 100.0).abs() < f64::EPSILON);
    }

    /// Acceptance criterion: constraint 0 at stage 0 `is_active` returns true.
    #[test]
    fn test_ac_is_active_true() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let rows = vec![make_row(0, 0, None, 100.0), make_row(0, 1, None, 150.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);
        assert!(table.is_active(0, 0));
    }

    /// Acceptance criterion: constraint 1 at stage 0 `is_active` returns false when no bounds.
    #[test]
    fn test_ac_is_active_false() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let rows = vec![make_row(0, 0, None, 100.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);
        assert!(!table.is_active(1, 0));
    }

    /// Acceptance criterion: `bounds_for_stage` returns (None, 100.0) for single bound row.
    #[test]
    fn test_ac_bounds_for_stage() {
        let constraints = vec![make_constraint(0)];
        let rows = vec![make_row(0, 0, None, 100.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);
        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(100.0),
                bound_upper: None,
            }
        );
    }
}
