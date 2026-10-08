//! The single owner of the `stage_id` domain-id reading.
//!
//! A study's `stage_id` is a declared domain id, not a positional index: the
//! declared ids may be negative (pre-study), gapped, or numbered `1..T`. Every
//! input file that carries a `stage_id` resolves it against the ordered study
//! stages through one [`StageIdResolver`], so the same domain id maps to the same
//! study index everywhere and an id naming no declared stage is rejected, never
//! silently re-based. Contiguity and 0-basedness are never assumed.

use std::collections::HashMap;
use std::path::Path;

use crate::LoadError;

/// Bidirectional map between a study's declared `stage_id` domain ids and their
/// canonical study-stage indices, built over the ordered study-stage id slice
/// (the `s.id` sequence in canonical study order, pre-study negative ids
/// excluded by the caller).
#[derive(Debug, Clone)]
pub struct StageIdResolver {
    by_id: HashMap<i32, usize>,
    ids: Vec<i32>,
}

impl StageIdResolver {
    /// Build a resolver over `ids`, the study stages in canonical study order.
    ///
    /// Accepts any set of distinct ids (negative, gapped, or `1..T`) and maps
    /// each to its position in `ids`; no contiguity or 0-basedness is asserted.
    #[must_use]
    pub fn from_study_stage_ids(ids: &[i32]) -> Self {
        let by_id = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        Self {
            by_id,
            ids: ids.to_vec(),
        }
    }

    /// Domain id → study index; `None` iff `stage_id` names no declared study stage.
    #[must_use]
    pub fn resolve(&self, stage_id: i32) -> Option<usize> {
        self.by_id.get(&stage_id).copied()
    }

    /// Study index → domain id; `None` iff `index` is out of range.
    #[must_use]
    pub fn id_at(&self, index: usize) -> Option<i32> {
        self.ids.get(index).copied()
    }

    /// The declared study-stage ids in canonical study order.
    #[must_use]
    pub fn study_stage_ids(&self) -> &[i32] {
        &self.ids
    }

    /// The `stage_id → study-index` inverse map by borrow, so existing
    /// `&HashMap<i32, usize>` consumers read the resolver's own map without
    /// deriving a second copy.
    #[must_use]
    pub fn index_map(&self) -> &HashMap<i32, usize> {
        &self.by_id
    }

    /// The canonical [`LoadError`] for a `stage_id` that resolves to no declared
    /// study stage: names the file (`path`), the per-row `field` label, the
    /// offending value, and the declared study-stage-id set. Every input file
    /// carrying an out-of-range `stage_id` reports through this one constructor so
    /// the message shape is identical regardless of which file carried it.
    pub fn unresolved_stage_id_error(
        &self,
        path: impl AsRef<Path>,
        field: impl Into<String>,
        stage_id: i32,
    ) -> LoadError {
        LoadError::SchemaError {
            path: path.as_ref().to_path_buf(),
            field: field.into(),
            message: format!(
                "stage_id {stage_id} resolves to no declared study stage; \
                 declared study stage ids: {:?}",
                self.ids
            ),
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn stage_id_resolver_round_trips_and_rejects() {
        // Non-contiguous, negative, and non-0-based ids all map to their position.
        let resolver = StageIdResolver::from_study_stage_ids(&[-2, 0, 3, 7]);

        assert_eq!(resolver.resolve(-2), Some(0));
        assert_eq!(resolver.resolve(0), Some(1));
        assert_eq!(resolver.resolve(3), Some(2));
        assert_eq!(resolver.resolve(7), Some(3));

        // Ids inside the numeric span but not declared do not resolve.
        assert_eq!(resolver.resolve(1), None);
        assert_eq!(resolver.resolve(2), None);

        // id_at round-trips every declared id.
        for (i, &id) in resolver.study_stage_ids().iter().enumerate() {
            assert_eq!(resolver.id_at(i), Some(id));
            assert_eq!(resolver.resolve(id), Some(i));
        }
        assert_eq!(resolver.id_at(4), None);

        // A `1..T` deck imposes no 0-basedness.
        let one_based = StageIdResolver::from_study_stage_ids(&[1, 2, 3]);
        assert_eq!(one_based.resolve(1), Some(0));
        assert_eq!(one_based.resolve(0), None);
    }

    #[test]
    fn stage_id_resolver_index_map_matches_resolve() {
        let resolver = StageIdResolver::from_study_stage_ids(&[-2, 0, 3, 7]);
        let map = resolver.index_map();

        for &id in resolver.study_stage_ids() {
            assert_eq!(Some(map[&id]), resolver.resolve(id));
        }
        assert_eq!(map.len(), resolver.study_stage_ids().len());
    }

    #[test]
    fn unresolved_stage_id_error_names_file_value_and_declared_set() {
        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
        let err = resolver.unresolved_stage_id_error(
            "scenarios/noise_openings.parquet",
            "noise_openings[3].stage_id",
            5,
        );
        match &err {
            LoadError::SchemaError {
                path,
                field,
                message,
            } => {
                assert!(
                    path.to_string_lossy()
                        .contains("scenarios/noise_openings.parquet"),
                    "path names the file: {path:?}"
                );
                assert_eq!(field, "noise_openings[3].stage_id");
                assert!(message.contains('5'), "message names the value: {message}");
                assert!(
                    message.contains("[0, 1]"),
                    "message names the declared set: {message}"
                );
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }
}
