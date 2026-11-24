//! Layer 5a — format rules for `post_study_stages.json`.

use novomodelo_core::{Hydro, WaterBucketStages, deepest_uncovered_travel_time_arc};

use super::super::{ValidationContext, rules, schema::ParsedData};
use super::post_study_calendar::build_post_study_calendar;
use crate::legacy::post_study_stages::PostStudyFormat;

const WATER_CALENDAR_KEY: &str = "water_bucket_stages";
const THERMAL_CALENDAR_KEY: &str = "thermal_stages";

/// Rule `semantic.5a.59`: a deck with a travel-time arc and no file. Rule `semantic.5a.63`: a
/// file declaring only one of `thermal_stages` and `thermal_bounds`. Rules `semantic.5a.61` and
/// `semantic.5a.62`: a file that omits the water calendar, or declares one too short, on a deck
/// with a travel-time arc. Rules `semantic.5a.64` and `semantic.5a.65`: a file declaring
/// `water_bucket_stages` on a deck with no travel-time arc, or `thermal_stages` on a deck with
/// no anticipated thermal.
pub(super) fn check_post_study_format(data: &ParsedData, ctx: &mut ValidationContext) {
    match data.post_study_format {
        Some(PostStudyFormat::Absent)
            if data.hydros.iter().any(Hydro::declares_travel_time_arc) =>
        {
            ctx.emit(
                &rules::POST_STUDY_FILE_MISSING,
                "post_study_stages.json",
                None::<String>,
                "the case declares a travel-time arc without `water_bucket_stages` and runs on the \
                 legacy post-study format, which is deprecated: declare `thermal_stages` and \
                 `water_bucket_stages` in post_study_stages.json, or run \
                 `python3 scripts/migrate/post_study_stages.py <case>`",
            );
        }
        Some(PostStudyFormat::Present {
            thermal_stages,
            thermal_bounds,
            ..
        }) if thermal_stages != thermal_bounds => {
            let (present, missing) = if thermal_stages {
                ("thermal_stages", "thermal_bounds")
            } else {
                ("thermal_bounds", "thermal_stages")
            };
            ctx.emit(
                &rules::POST_STUDY_THERMAL_PAIR_INCOMPLETE,
                "post_study_stages.json",
                Some(missing),
                format!(
                    "post_study_stages.json declares `{present}` without `{missing}`; \
                     `thermal_stages` and `thermal_bounds` must be declared together (an empty \
                     list counts as declared)."
                ),
            );
        }
        _ => {}
    }

    if let Some(PostStudyFormat::Present {
        thermal_stages,
        water_bucket_stages,
        ..
    }) = data.post_study_format
    {
        check_water_calendar(data, water_bucket_stages, ctx);
        if thermal_stages
            && !data
                .thermals
                .iter()
                .any(|thermal| thermal.anticipated_config.is_some())
        {
            ctx.emit(
                &rules::POST_STUDY_THERMAL_CALENDAR_UNUSED,
                "post_study_stages.json",
                Some(THERMAL_CALENDAR_KEY),
                format!(
                    "post_study_stages.json declares `{THERMAL_CALENDAR_KEY}` but no thermal is \
                     anticipated; the key has no effect without an anticipated thermal."
                ),
            );
        }
    }
}

fn check_water_calendar(data: &ParsedData, declared: bool, ctx: &mut ValidationContext) {
    if !declared {
        if let Some(hydro) = data
            .hydros
            .iter()
            .filter(|hydro| hydro.declares_travel_time_arc())
            .min_by_key(|hydro| hydro.id)
        {
            ctx.emit(
                &rules::POST_STUDY_WATER_CALENDAR_MISSING,
                "post_study_stages.json",
                Some(WATER_CALENDAR_KEY),
                format!(
                    "post_study_stages.json omits `{WATER_CALENDAR_KEY}` but hydro {} declares a \
                     travel-time arc; declare a water calendar covering its travel_time_hours.",
                    hydro.id
                ),
            );
        }
        return;
    }

    if !data.hydros.iter().any(Hydro::declares_travel_time_arc) {
        ctx.emit(
            &rules::POST_STUDY_WATER_CALENDAR_UNUSED,
            "post_study_stages.json",
            Some(WATER_CALENDAR_KEY),
            format!(
                "post_study_stages.json declares `{WATER_CALENDAR_KEY}` but no hydro declares a \
                 travel-time arc; the key has no effect without a travel-time arc."
            ),
        );
    }

    let Some(WaterBucketStages::Declared(water_stages)) = data
        .post_study_stages
        .as_ref()
        .map(|stages| &stages.water_bucket_stages)
    else {
        return;
    };

    let study_end = data
        .stages
        .stages
        .iter()
        .rfind(|stage| stage.id >= 0)
        .map(|stage| stage.end_date);
    build_post_study_calendar(water_stages, WATER_CALENDAR_KEY, study_end, ctx);

    if let Some(arc) = deepest_uncovered_travel_time_arc(&data.hydros, water_stages) {
        ctx.emit(
            &rules::POST_STUDY_WATER_CALENDAR_SHORT,
            "post_study_stages.json",
            Some(WATER_CALENDAR_KEY),
            format!(
                "`{WATER_CALENDAR_KEY}` covers {} h but hydro {} declares a travel-time arc of \
                 {} h; extend the water calendar to at least {} h.",
                arc.declared_hours, arc.hydro_id, arc.travel_time_hours, arc.travel_time_hours
            ),
        );
    }
}
