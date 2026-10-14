//! Commissioning-window and filling-lifecycle predicates shared by every
//! equipment family.

use crate::entities::FillingConfig;

/// Whether an entity is operationally commissioned at `stage_id`:
/// `entry <= stage_id < exit`, with the exit bound **half-open** (a stage equal to
/// `exit` is decommissioned) and `None` entry/exit meaning no lower/upper bound.
///
/// Single owner of the commissioning predicate for every equipment family.
#[inline]
#[must_use]
pub fn commissioning_active(entry: Option<i32>, exit: Option<i32>, stage_id: i32) -> bool {
    entry.is_none_or(|e| e <= stage_id) && exit.is_none_or(|e| stage_id < e)
}

/// Lifecycle phase of a commissioned reservoir at one stage. Derived solely by
/// [`filling_phase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Before `start_stage_id` (only when `start_stage_id > 0`), and every
    /// commissioning-dormant stage of a non-filling hydro: the dam is not built;
    /// the river flows past its site.
    PreFilling,
    /// `start_stage_id <= stage_id < entry`: impounding water toward the dead
    /// volume, not yet a generating plant.
    Filling,
    /// `stage_id >= entry`, and every commissioned stage of a hydro with no
    /// `FillingConfig`: a normal operating plant.
    Operating,
}

/// The lifecycle [`Phase`] of a hydro at `stage_id`.
///
/// Keyed on the study **stage id** (`stage.id`), never the stage index: the two
/// diverge under multi-resolution / decomposition stages, where keying on the index
/// assigns the wrong phase (mirrors [`commissioning_active`]).
///
/// A non-filling hydro (`filling.is_none()`) is [`Phase::PreFilling`] at every
/// commissioning-dormant stage (`!commissioning_active`) and [`Phase::Operating`]
/// otherwise: a dormant site routes its inflow downstream via the short-circuit;
/// classifying it `Operating` with zeroed bounds traps the water. `entry/exit =
/// None` ⇒ always commissioned ⇒ `Operating` at every stage, bit-identical to a
/// normal hydro (parity-neutral).
///
/// Single owner of the phase derivation; callers recompute per stage — never
/// cache a per-stage `Phase` mask. Total function, no panic.
#[inline]
#[must_use]
pub fn filling_phase(
    filling: Option<&FillingConfig>,
    entry: Option<i32>,
    exit: Option<i32>,
    stage_id: i32,
) -> Phase {
    let Some(config) = filling else {
        return if commissioning_active(entry, exit, stage_id) {
            Phase::Operating
        } else {
            Phase::PreFilling
        };
    };
    if config.start_stage_id > 0 && stage_id < config.start_stage_id {
        return Phase::PreFilling;
    }
    match entry {
        Some(e) if stage_id < e => Phase::Filling,
        _ => Phase::Operating,
    }
}

/// Whether hydro is operationally active (generating) at `stage_id`:
/// `true` iff [`filling_phase`] is [`Phase::Operating`].
///
/// The single source of truth for the per-`(hydro, stage)` active decision. A
/// dormant/`PreFilling`/`Filling` hydro returns `false`. Total function, no panic.
#[inline]
#[must_use]
pub fn hydro_operating_active(
    filling: Option<&FillingConfig>,
    entry: Option<i32>,
    exit: Option<i32>,
    stage_id: i32,
) -> bool {
    matches!(
        filling_phase(filling, entry, exit, stage_id),
        Phase::Operating
    )
}

#[cfg(test)]
mod tests {
    use super::{FillingConfig, Phase, filling_phase, hydro_operating_active};

    /// Builds a `FillingConfig` with the given `start_stage_id`; the impound cap
    /// is irrelevant to phase derivation, so any non-negative value is fine.
    fn config(start_stage_id: i32) -> FillingConfig {
        FillingConfig {
            start_stage_id,
            filling_min_rate_m3s: 0.0,
        }
    }

    /// `filling.is_none()` with `entry/exit = None` ⇒ `Operating` at every stage —
    /// the parity-neutrality contract: an un-commissioned-window hydro is
    /// bit-identical to a normal operating hydro. The forbidden alternative — letting
    /// the new dormant branch push a window-free hydro into `PreFilling` — would
    /// change every existing D-case's physics.
    #[test]
    fn none_filling_none_window_is_operating_at_every_stage() {
        for stage_id in [-1, 0, 1, 4, 100] {
            assert_eq!(
                filling_phase(None, None, None, stage_id),
                Phase::Operating,
                "none filling, no window, stage_id={stage_id}"
            );
        }
    }

    /// `filling.is_none()` with a commissioning window: `PreFilling` before `entry`,
    /// `Operating` from `entry` (no intervening `Filling` — a non-filling hydro has
    /// no impounding stage), then `PreFilling` again at/after `exit`. The forbidden
    /// alternative — `Operating` before `entry` — would leave the un-built dam's
    /// inflow trapped on its own balance row.
    #[test]
    fn none_filling_with_window_is_prefilling_outside_entry_exit() {
        let entry = Some(2);
        let exit = Some(5);
        assert_eq!(filling_phase(None, entry, exit, 0), Phase::PreFilling);
        assert_eq!(filling_phase(None, entry, exit, 1), Phase::PreFilling);
        assert_eq!(filling_phase(None, entry, exit, 2), Phase::Operating);
        assert_eq!(filling_phase(None, entry, exit, 4), Phase::Operating);
        assert_eq!(filling_phase(None, entry, exit, 5), Phase::PreFilling);
        assert_eq!(filling_phase(None, entry, exit, 9), Phase::PreFilling);
    }

    /// `filling.is_none()` with `entry` beyond the horizon ⇒ `PreFilling` at every
    /// stage (always-dormant), the "plant never commissions in this study" case.
    #[test]
    fn none_filling_entry_beyond_horizon_is_prefilling_everywhere() {
        let entry = Some(1000);
        for stage_id in [0, 1, 4, 100] {
            assert_eq!(
                filling_phase(None, entry, None, stage_id),
                Phase::PreFilling,
                "always-dormant non-filling at stage_id={stage_id}"
            );
        }
    }

    /// With `start_stage_id = 2` and `entry = 4`, the three phases at their exact
    /// transition ids: `stage_id == entry - 1` is the last Filling stage and
    /// `stage_id == entry` is the first Operating stage (the half-open `< entry`
    /// boundary — a non-strict `<= entry` would keep the reservoir Filling one
    /// stage too long).
    #[test]
    fn three_phases_at_exact_boundaries() {
        let f = config(2);
        let entry = Some(4);
        assert_eq!(filling_phase(Some(&f), entry, None, 0), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), entry, None, 1), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), entry, None, 2), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, None, 3), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, None, 4), Phase::Operating);
        assert_eq!(filling_phase(Some(&f), entry, None, 5), Phase::Operating);
    }

    /// `start_stage_id == 0` ⇒ no `PreFilling`: Filling runs from stage 0 (how a
    /// study that starts mid-filling is expressed). The forbidden alternative —
    /// treating `stage_id < start_stage_id` without the `start_stage_id > 0`
    /// guard — is moot here (no stage is below 0), but stage 0 must be Filling,
    /// not `PreFilling`.
    #[test]
    fn start_zero_is_filling_at_stage_zero() {
        let f = config(0);
        let entry = Some(4);
        assert_eq!(filling_phase(Some(&f), entry, None, 0), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, None, 3), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, None, 4), Phase::Operating);
    }

    /// A `FillingConfig` with `entry = None` is never Filling/Operating-by-entry:
    /// before `start_stage_id` it is `PreFilling`, at/after it falls through to
    /// `Operating` (the `_` match arm), since there is no entry boundary to cross.
    #[test]
    fn filling_with_none_entry_falls_through_to_operating() {
        let f = config(2);
        assert_eq!(filling_phase(Some(&f), None, None, 1), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), None, None, 2), Phase::Operating);
        assert_eq!(filling_phase(Some(&f), None, None, 5), Phase::Operating);
    }

    /// `hydro_operating_active` truth table — the per-(hydro, stage) operating-active
    /// predicate: `true` iff `Operating`, `false` for `PreFilling`/`Filling`. Pins each
    /// branch (non-filling no-window, non-filling dormant before/at/after window,
    /// filling lifecycle).
    #[test]
    fn hydro_operating_active_truth_table() {
        // Non-filling, no window: active everywhere.
        assert!(hydro_operating_active(None, None, None, 0));
        assert!(hydro_operating_active(None, None, None, 100));
        // Non-filling, window [2, 5): dormant before entry, active inside, dormant
        // from exit.
        assert!(!hydro_operating_active(None, Some(2), Some(5), 1));
        assert!(hydro_operating_active(None, Some(2), Some(5), 2));
        assert!(hydro_operating_active(None, Some(2), Some(5), 4));
        assert!(!hydro_operating_active(None, Some(2), Some(5), 5));
        // Filling lifecycle (start 2, entry 4): inactive in PreFilling and Filling,
        // active from entry.
        let f = config(2);
        assert!(!hydro_operating_active(Some(&f), Some(4), None, 0));
        assert!(!hydro_operating_active(Some(&f), Some(4), None, 3));
        assert!(hydro_operating_active(Some(&f), Some(4), None, 4));
    }
}
