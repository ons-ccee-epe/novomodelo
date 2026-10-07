//! Typed stage keys distinguishing the domain stage identifier, the 0-based
//! study-horizon position, and the calendar month a stage falls in.
//!
//! [Stage] carries `id` (the domain identifier from
//! `stages.json`, not guaranteed contiguous or 0-based), `index` (a position
//! over the FULL canonical stage vector, pre-study stages included), and
//! `season_id` (a season-cycle index whose meaning — calendar month, week, or
//! an arbitrary custom bucket — depends on `SeasonCycleType` and is *not*
//! always the calendar month). A bare `usize`/`i32` distinguishes neither
//! "look up by domain id" from "look up by study position" nor "season-cycle
//! index" from "calendar month": a mismatch keys by the wrong convention and
//! still compiles (a downstream table keyed by domain id relies on this
//! distinction; `season_id` is not the calendar month on `Weekly`/`Custom`
//! cycles). [`StageId`], [`StudyPos`], and [`CalendarMonth`] make each
//! convention a distinct type so a mismatched call site is a compile error,
//! not a silent wrong-stage lookup.

use crate::temporal::Stage;
use chrono::Datelike;

/// A study's domain stage identifier (`Stage::id`), independent of storage
/// position. Not guaranteed 0-based or contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct StageId(pub i32);

/// A 0-based position within the study horizon (the stages with
/// `Stage::id >= 0`, enumerated in that filtered order). Equal to
/// `Stage::index` only when the study declares no pre-study stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct StudyPos(pub usize);

/// A 0-based calendar month (0 = January … 11 = December), derived from a
/// stage's `start_date` rather than from [`Stage::season_id`]. Constructible
/// only via [`month_of`], which sources it from `chrono::Datelike::month0`
/// (always `0..=11`), so every value is in range for indexing a 12-element
/// month-keyed array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct CalendarMonth(usize);

impl CalendarMonth {
    /// The 0-based month index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Derives the calendar month `stage` falls in from its `start_date` — the
/// anchor for a stage spanning a month boundary (`end_date` is exclusive and
/// may land in the following month).
#[must_use]
pub fn month_of(stage: &Stage) -> CalendarMonth {
    CalendarMonth(stage.start_date.month0() as usize)
}

#[cfg(test)]
mod tests {
    use crate::temporal::{
        BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use chrono::NaiveDate;

    use super::*;

    #[test]
    fn stage_id_is_zero_cost() {
        assert_eq!(std::mem::size_of::<StageId>(), std::mem::size_of::<i32>());
    }

    #[test]
    fn study_pos_is_zero_cost() {
        assert_eq!(
            std::mem::size_of::<StudyPos>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn stage_id_orders_by_wrapped_value() {
        assert!(StageId(-1) < StageId(0));
        assert!(StageId(0) < StageId(60));
        assert_eq!(StageId(60), StageId(60));
    }

    #[test]
    fn study_pos_orders_by_wrapped_value() {
        assert!(StudyPos(0) < StudyPos(1));
        assert!(StudyPos(1) < StudyPos(2));
        assert_eq!(StudyPos(2), StudyPos(2));
    }

    fn make_stage(id: i32, start_date: NaiveDate, season_id: Option<usize>) -> Stage {
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date,
            end_date: start_date,
            season_id,
            blocks: Vec::new(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// On a Monthly-cycle study `season_id == month0`, so `month_of` must
    /// reproduce it exactly.
    #[test]
    fn month_of_monthly_stage_equals_season_id() {
        let june = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap_or_default();
        let stage = make_stage(0, june, Some(5));
        assert_eq!(month_of(&stage).index(), 5);
    }

    /// On a Custom cycle, `season_id` need not be the calendar month; `month_of`
    /// derives the month from `start_date` regardless of what `season_id` holds.
    #[test]
    fn month_of_custom_stage_derives_from_date_not_season_id() {
        let june = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap_or_default();
        let stage = make_stage(0, june, Some(3));
        assert_eq!(month_of(&stage).index(), 5);
        assert_ne!(
            month_of(&stage).index(),
            stage.season_id.unwrap_or(usize::MAX)
        );
    }

    /// A Weekly-cycle `season_id` (`>= 12`) does not participate in month
    /// derivation at all — `month_of` reads only `start_date`.
    #[test]
    fn month_of_weekly_stage_derives_valid_month_despite_season_id_ge_12() {
        let december = NaiveDate::from_ymd_opt(2024, 12, 1).unwrap_or_default();
        let stage = make_stage(0, december, Some(48));
        assert_eq!(month_of(&stage).index(), 11);
    }
}
