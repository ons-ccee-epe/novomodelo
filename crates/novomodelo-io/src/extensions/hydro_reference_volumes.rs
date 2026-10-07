//! Per-`(plant, stage)` reference operating volume resolution.
//!
//! [`HydroReferenceVolumeFractions::get`] returns the resolved reference operating
//! volume in absolute storage units (hm³) for a given `(hydro_id, stage_pos)`
//! — `stage_pos` is the 0-based study-horizon position, not the domain
//! [`cobre_core::StageId`] — falling back to the case default for an unpopulated
//! pair.
//!
//! The caller resolves the declared input (absolute, percentile, or default) to
//! absolute hm³ — including the `[v_min, v_max]` band a percentile resolves
//! against — before construction, so this module stays free of any band assumption.

use std::collections::HashMap;

use cobre_core::{EntityId, StudyPos};

/// Resolver returning the reference operating volume (hm³) for a given
/// `(hydro_id, stage_pos)` pair, built via [`build_hydro_reference_volumes_resolved`].
#[derive(Debug, Clone)]
pub struct HydroReferenceVolumeFractions {
    resolved_hm3: HashMap<(EntityId, usize), f64>,
    default_value: f64,
}

impl HydroReferenceVolumeFractions {
    /// Resolved reference operating volume (hm³) for `(hydro, stage)`, or the
    /// scalar default for an unpopulated key.
    #[must_use]
    pub fn get(&self, hydro_id: EntityId, stage_pos: StudyPos) -> f64 {
        self.resolved_hm3
            .get(&(hydro_id, stage_pos.0))
            .copied()
            .unwrap_or(self.default_value)
    }
}

/// Build a resolver from already-resolved per-`(plant, stage)` reference volumes
/// (absolute hm³). `default_hm3` is a defensive fallback for any key the caller
/// did not populate; callers populate every `(plant, stage)` they will query.
///
/// Order-independent: the map contents do not depend on the input-pair order, so
/// declaration order does not affect any later `get`.
#[must_use]
pub fn build_hydro_reference_volumes_resolved(
    resolved_hm3: &[(EntityId, StudyPos, f64)],
    default_hm3: f64,
) -> HydroReferenceVolumeFractions {
    let mut map: HashMap<(EntityId, usize), f64> = HashMap::with_capacity(resolved_hm3.len());
    for &(hydro_id, stage_pos, value) in resolved_hm3 {
        map.insert((hydro_id, stage_pos.0), value);
    }
    HydroReferenceVolumeFractions {
        resolved_hm3: map,
        default_value: default_hm3,
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use cobre_core::{EntityId, StudyPos};

    use super::build_hydro_reference_volumes_resolved;

    #[test]
    fn resolved_constructor_returns_stored_value_directly() {
        let resolver = build_hydro_reference_volumes_resolved(
            &[
                (EntityId(42), StudyPos(0), 800.0),
                (EntityId(42), StudyPos(1), 150.0),
                (EntityId(7), StudyPos(0), 1234.5),
            ],
            999.0,
        );
        assert_eq!(resolver.get(EntityId(42), StudyPos(0)), 800.0);
        assert_eq!(resolver.get(EntityId(42), StudyPos(1)), 150.0);
        assert_eq!(resolver.get(EntityId(7), StudyPos(0)), 1234.5);
    }

    #[test]
    fn resolved_constructor_falls_back_to_default_for_unpopulated_key() {
        let resolver =
            build_hydro_reference_volumes_resolved(&[(EntityId(42), StudyPos(0), 800.0)], 999.0);
        assert_eq!(resolver.get(EntityId(42), StudyPos(5)), 999.0);
        assert_eq!(resolver.get(EntityId(99), StudyPos(0)), 999.0);
    }

    #[test]
    fn resolved_constructor_is_order_independent() {
        let forward = build_hydro_reference_volumes_resolved(
            &[
                (EntityId(1), StudyPos(0), 10.0),
                (EntityId(2), StudyPos(0), 20.0),
            ],
            0.0,
        );
        let reverse = build_hydro_reference_volumes_resolved(
            &[
                (EntityId(2), StudyPos(0), 20.0),
                (EntityId(1), StudyPos(0), 10.0),
            ],
            0.0,
        );
        assert_eq!(
            forward.get(EntityId(1), StudyPos(0)).to_bits(),
            reverse.get(EntityId(1), StudyPos(0)).to_bits()
        );
        assert_eq!(
            forward.get(EntityId(2), StudyPos(0)).to_bits(),
            reverse.get(EntityId(2), StudyPos(0)).to_bits()
        );
    }
}
