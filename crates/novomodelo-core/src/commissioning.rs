//! Commissioning-window and filling-lifecycle predicates shared by every
//! equipment family.

use crate::entities::{FillingConfig, Hydro};

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

/// The hard lower bound of `hydro`'s storage column at `stage_id`: `0.0` where
/// the floor is relaxed, otherwise `min_storage_hm3`.
///
/// CONTRACT: `min_storage` is a HARD lower bound for every hydro EXCEPT (a) a
/// filling one (`hydro.filling.is_some()`), whose floor relaxes to `0` in ALL
/// phases and is re-imposed as a SOFT floor by the stage LP, and (b) a
/// non-filling hydro while [`Phase::PreFilling`] (commissioning-dormant), whose
/// frozen-identity row pins its storage to the inert initial storage — a hard
/// floor above that value would reject the pin and make the LP infeasible.
/// FORBIDDEN — globalizing the relax to all `Operating` hydros (makes dead
/// volume soft system-wide); keeping it hard through a dormant non-filling stage
/// (rejects the initial-storage pin). The dormant relax disappears at
/// `Operating`, restoring the hard floor.
///
/// The caller chooses `min_storage_hm3`: a stage LP passes that stage's resolved
/// bound, the initial-state projection passes the entity's own value. Single
/// owner of which hydros have a hard floor. Total function, no panic; a NaN
/// `min_storage_hm3` passes through unchanged.
#[inline]
#[must_use]
pub fn hard_storage_floor(hydro: &Hydro, stage_id: i32, min_storage_hm3: f64) -> f64 {
    let relaxed = hydro.filling.is_some()
        || matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage_id
            ),
            Phase::PreFilling
        );
    if relaxed { 0.0 } else { min_storage_hm3 }
}

/// The lowest incoming storage a stage can see, as classified by
/// [`incoming_storage_floor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IncomingStorageFloor {
    /// First study stage: the incoming storage is the canonicalized initial
    /// storage.
    InitialStorage,
    /// The predecessor stage is [`Phase::PreFilling`]: the incoming storage is the
    /// pinned seed, which is the canonicalized initial storage held unchanged
    /// since the first study stage.
    PinnedSeed,
    /// The incoming storage is the predecessor stage's outgoing storage, bounded
    /// below by that stage's [`hard_storage_floor`] (hm³).
    PredecessorFloor(f64),
}

/// The [`IncomingStorageFloor`] of `hydro` at a stage whose predecessor study
/// stage is `predecessor`, given as that stage's id and its resolved
/// `min_storage_hm3`; `None` means the first study stage.
///
/// `predecessor` carries the PREDECESSOR's resolved minimum, not the current
/// stage's: the incoming storage was produced under the predecessor's bounds.
/// Phases are monotone in stage id, so a hydro that is `PreFilling` at the
/// predecessor and active now has been `PreFilling` since the first study stage.
/// Total function, no panic.
#[inline]
#[must_use]
pub fn incoming_storage_floor(
    hydro: &Hydro,
    predecessor: Option<(i32, f64)>,
) -> IncomingStorageFloor {
    let Some((predecessor_id, predecessor_min_hm3)) = predecessor else {
        return IncomingStorageFloor::InitialStorage;
    };
    let predecessor_phase = filling_phase(
        hydro.filling.as_ref(),
        hydro.entry_stage_id,
        hydro.exit_stage_id,
        predecessor_id,
    );
    if matches!(predecessor_phase, Phase::PreFilling) {
        IncomingStorageFloor::PinnedSeed
    } else {
        IncomingStorageFloor::PredecessorFloor(hard_storage_floor(
            hydro,
            predecessor_id,
            predecessor_min_hm3,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FillingConfig, Hydro, IncomingStorageFloor, Phase, filling_phase, hard_storage_floor,
        hydro_operating_active, incoming_storage_floor,
    };
    use crate::test_support::{HydroSpec, make_hydro};

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
    /// the dormant branch push a window-free hydro into `PreFilling` — would
    /// change every D-case's physics.
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

    fn hydro(filling: Option<FillingConfig>, entry: Option<i32>, exit: Option<i32>) -> Hydro {
        let mut hydro = make_hydro(HydroSpec::default());
        hydro.filling = filling;
        hydro.entry_stage_id = entry;
        hydro.exit_stage_id = exit;
        hydro
    }

    /// A filling hydro's floor relaxes in ALL phases (it is re-imposed as a soft
    /// floor elsewhere). The forbidden alternative — relaxing only while
    /// `PreFilling` — would hard-bound the dead volume while the reservoir fills.
    #[test]
    fn hard_storage_floor_is_zero_for_a_filling_hydro_in_every_phase() {
        let filling_hydro = hydro(Some(config(2)), Some(4), None);
        for (stage_id, phase) in [
            (0, Phase::PreFilling),
            (2, Phase::Filling),
            (4, Phase::Operating),
        ] {
            assert_eq!(
                filling_phase(
                    filling_hydro.filling.as_ref(),
                    filling_hydro.entry_stage_id,
                    filling_hydro.exit_stage_id,
                    stage_id
                ),
                phase,
                "fixture phase at stage_id={stage_id}"
            );
            assert_eq!(
                hard_storage_floor(&filling_hydro, stage_id, 30.0),
                0.0,
                "filling hydro at stage_id={stage_id}"
            );
        }
    }

    /// A non-filling hydro's floor relaxes while commissioning-dormant — before
    /// `entry_stage_id` and from `exit_stage_id` on (the half-open exit).
    #[test]
    fn hard_storage_floor_is_zero_while_a_non_filling_hydro_is_dormant() {
        let windowed = hydro(None, Some(2), Some(5));
        for stage_id in [1, 5] {
            assert_eq!(
                hard_storage_floor(&windowed, stage_id, 30.0),
                0.0,
                "dormant non-filling hydro at stage_id={stage_id}"
            );
        }
    }

    /// The relax disappears at `Operating`: the floor is the caller's minimum,
    /// unchanged. Globalizing the relax to every operating hydro would make dead
    /// volume soft system-wide.
    #[test]
    fn hard_storage_floor_is_the_given_minimum_for_an_operating_hydro() {
        let always_on = hydro(None, None, None);
        assert_eq!(hard_storage_floor(&always_on, 0, 30.0), 30.0);
        let windowed = hydro(None, Some(2), Some(5));
        for stage_id in [2, 4] {
            assert_eq!(
                hard_storage_floor(&windowed, stage_id, 30.0),
                30.0,
                "operating non-filling hydro at stage_id={stage_id}"
            );
        }
    }

    #[test]
    fn incoming_storage_floor_is_the_initial_storage_at_the_first_stage() {
        for h in [
            hydro(None, None, None),
            hydro(Some(config(0)), Some(4), None),
        ] {
            assert_eq!(
                incoming_storage_floor(&h, None),
                IncomingStorageFloor::InitialStorage
            );
        }
    }

    /// The hydro is dormant at the predecessor and active now: its incoming
    /// storage is the seed the frozen-identity row pinned, whatever the
    /// predecessor's resolved minimum.
    #[test]
    fn incoming_storage_floor_is_the_pinned_seed_after_a_dormant_stage() {
        let non_filling = hydro(None, Some(2), None);
        assert_eq!(
            incoming_storage_floor(&non_filling, Some((1, 30.0))),
            IncomingStorageFloor::PinnedSeed
        );
        let filling_hydro = hydro(Some(config(2)), Some(4), None);
        assert_eq!(
            incoming_storage_floor(&filling_hydro, Some((1, 30.0))),
            IncomingStorageFloor::PinnedSeed
        );
    }

    /// The predecessor's own resolved minimum feeds the hard floor, never the
    /// current stage's.
    #[test]
    fn incoming_storage_floor_is_the_predecessor_hard_floor() {
        let non_filling = hydro(None, Some(2), Some(5));
        assert_eq!(
            incoming_storage_floor(&non_filling, Some((3, 30.0))),
            IncomingStorageFloor::PredecessorFloor(30.0)
        );
        let filling_hydro = hydro(Some(config(0)), Some(4), None);
        assert_eq!(
            incoming_storage_floor(&filling_hydro, Some((2, 30.0))),
            IncomingStorageFloor::PredecessorFloor(0.0)
        );
    }
}
