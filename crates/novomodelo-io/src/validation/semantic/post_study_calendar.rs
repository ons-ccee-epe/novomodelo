//! Layer 5a — post-study calendar well-formedness, shared by every post-study calendar key.

use chrono::{NaiveDate, TimeDelta};
use novomodelo_core::PostStudyStage;
use novomodelo_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};

use super::super::{ValidationContext, rules};

/// Build a post-study calendar as `Stage`s (each `end_date` = `start_date`
/// advanced by `duration_hours` rounded to whole days — the whole-day alignment
/// [`StageCalendar`](novomodelo_stochastic::season_cast::StageCalendar) coverage
/// needs, mirroring the pre-study duration→days round in `travel_time`),
/// enforcing invariant (a). `key` is the calendar key the user wrote and names
/// the field path of every diagnostic. Returns the stages and whether they are
/// well-formed; a non-well-formed calendar suppresses the coverage pass so its
/// errors are not noise on a broken calendar.
pub(super) fn build_post_study_calendar(
    stages: &[PostStudyStage],
    key: &str,
    study_end: Option<NaiveDate>,
    ctx: &mut ValidationContext,
) -> (Vec<Stage>, bool) {
    let mut calendar_stages: Vec<Stage> = Vec::with_capacity(stages.len());
    let mut well_formed = true;

    for (i, stage) in stages.iter().enumerate() {
        match post_study_end_date(stage.start_date, stage.duration_hours) {
            Some(end_date) if end_date > stage.start_date => {
                calendar_stages.push(make_calendar_stage(
                    i,
                    stage.start_date,
                    end_date,
                    stage.duration_hours,
                ));
            }
            _ => {
                ctx.emit(
                    &rules::POST_STUDY_BOUNDARY_INCONSISTENT,
                    "post_study_stages.json",
                    Some(format!("{key}[{i}]")),
                    format!(
                        "post-study stage starting {} has duration_hours {} that rounds to a \
                         non-positive whole-day span; declare a duration of at least a day.",
                        stage.start_date, stage.duration_hours
                    ),
                );
                well_formed = false;
            }
        }
    }

    match (calendar_stages.first(), study_end) {
        (Some(first), Some(end)) if first.start_date != end => {
            ctx.emit(
                &rules::POST_STUDY_BOUNDARY_INCONSISTENT,
                "post_study_stages.json",
                Some(format!("{key}[0].start_date")),
                format!(
                    "first post-study stage starts {} but the study horizon ends {}; the \
                     post-study calendar must begin exactly at the study horizon end.",
                    first.start_date, end
                ),
            );
            well_formed = false;
        }
        (Some(_), None) => {
            ctx.emit(
                &rules::POST_STUDY_BOUNDARY_INCONSISTENT,
                "post_study_stages.json",
                Some(key),
                "post_study_stages.json is declared but the study has no study stages to anchor \
                 the post-study calendar against."
                    .to_string(),
            );
            well_formed = false;
        }
        _ => {}
    }

    for pair in calendar_stages.windows(2) {
        if pair[0].end_date != pair[1].start_date {
            ctx.emit(
                &rules::POST_STUDY_BOUNDARY_INCONSISTENT,
                "post_study_stages.json",
                Some(format!("{key} start_date={}", pair[1].start_date)),
                format!(
                    "post-study stages are not date-contiguous: the stage starting {} ends {} \
                     but the next stage starts {}; each stage must end exactly where the next \
                     begins.",
                    pair[0].start_date, pair[0].end_date, pair[1].start_date
                ),
            );
            well_formed = false;
        }
    }

    (calendar_stages, well_formed)
}

/// Post-study stage end date: `start_date` advanced by `duration_hours` rounded
/// to whole days. `None` only on calendar overflow.
#[allow(clippy::cast_possible_truncation)] // a validated finite positive duration yields a small non-negative day count
fn post_study_end_date(start_date: NaiveDate, duration_hours: f64) -> Option<NaiveDate> {
    let days = (duration_hours / 24.0).round() as i64;
    start_date.checked_add_signed(TimeDelta::days(days))
}

/// Synthesize a `Stage` carrying only the fields
/// [`StageCalendar`](novomodelo_stochastic::season_cast::StageCalendar) reads
/// (`start_date`, `end_date`, `blocks`); the rest are inert placeholders.
fn make_calendar_stage(
    index: usize,
    start_date: NaiveDate,
    end_date: NaiveDate,
    duration_hours: f64,
) -> Stage {
    Stage {
        index,
        id: i32::try_from(index).unwrap_or(i32::MAX),
        start_date,
        end_date,
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "post_study".to_string(),
            duration_hours,
        }],
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use chrono::{NaiveDate, TimeDelta};
    use novomodelo_core::PostStudyStage;

    use super::build_post_study_calendar;
    use crate::validation::ValidationContext;

    #[test]
    fn field_paths_name_the_given_key() {
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stages = [
            PostStudyStage {
                start_date: start,
                duration_hours: 168.0,
            },
            PostStudyStage {
                start_date: start + TimeDelta::days(14),
                duration_hours: 168.0,
            },
        ];
        let mut ctx = ValidationContext::new();

        let (_, well_formed) =
            build_post_study_calendar(&stages, "water_bucket_stages", Some(start), &mut ctx);

        assert!(!well_formed);
        let errors = ctx.errors();
        assert_eq!(errors.len(), 1, "got: {errors:?}");
        let entity = errors[0].entity.as_deref().unwrap();
        assert!(
            entity.starts_with("water_bucket_stages"),
            "field path must name the given key, got: {entity}"
        );
    }
}
