//! Layer 5a — format rules for `post_study_stages.json`.

use novomodelo_core::{Hydro, WaterBucketStages, deepest_uncovered_travel_time_arc};

use super::super::{
    ValidationContext, rules,
    schema::{BoundaryFormat, ParsedData},
};
use super::post_study_calendar::build_post_study_calendar;
use crate::legacy::post_study_stages::PostStudyFormat;

const WATER_CALENDAR_KEY: &str = "water_bucket_stages";
const THERMAL_CALENDAR_KEY: &str = "thermal_stages";

#[derive(Clone, Copy)]
struct CalendarWording {
    file: &'static str,
    water_key: &'static str,
    thermal_key: &'static str,
}

const LEGACY_WORDING: CalendarWording = CalendarWording {
    file: "post_study_stages.json",
    water_key: WATER_CALENDAR_KEY,
    thermal_key: THERMAL_CALENDAR_KEY,
};

const CURRENT_WORDING: CalendarWording = CalendarWording {
    file: "stages.json",
    water_key: "post_study_water_stages",
    thermal_key: "post_study_thermal_stages",
};

/// Rule `semantic.5a.59`: a deck with a travel-time arc and no file. Rule `semantic.5a.63`: a
/// file declaring only one of `thermal_stages` and `thermal_bounds`. Rules `semantic.5a.61` and
/// `semantic.5a.62`: a file, or `stages.json` in the current boundary format, that omits the
/// water calendar, or declares one too short, on a deck with a travel-time arc. Rules
/// `semantic.5a.64` and `semantic.5a.65`: a water calendar on a deck with no travel-time arc, or
/// a thermal calendar on a deck with no anticipated thermal. Rule `semantic.5a.71`: a
/// `post_study_*` list in `stages.json` beside the deprecated `initial_conditions.json` file
/// set, which ignores it.
pub(super) fn check_post_study_format(data: &ParsedData, ctx: &mut ValidationContext) {
    match data.boundary_format {
        Some(BoundaryFormat::Legacy(PostStudyFormat::Absent))
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
        Some(BoundaryFormat::Legacy(PostStudyFormat::Present {
            thermal_stages,
            thermal_bounds,
            ..
        })) if thermal_stages != thermal_bounds => {
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

    if matches!(data.boundary_format, Some(BoundaryFormat::Legacy(_)))
        && (data.stages.post_study.thermal.is_some() || data.stages.post_study.water.is_some())
    {
        ctx.emit(
            &rules::POST_STUDY_CALENDARS_WITH_LEGACY_FILES,
            CURRENT_WORDING.file,
            None::<String>,
            "stages.json declares a post_study_* list, but the case reads its boundary data from \
             the deprecated initial_conditions.json file set, which ignores it; move the \
             boundary data to boundary_conditions.json or remove the list.",
        );
    }

    let (wording, declared_thermal, declared_water) = match data.boundary_format {
        Some(BoundaryFormat::Legacy(PostStudyFormat::Present {
            thermal_stages,
            water_bucket_stages,
            ..
        })) => (LEGACY_WORDING, thermal_stages, water_bucket_stages),
        Some(BoundaryFormat::Current) => (
            CURRENT_WORDING,
            data.stages.post_study.thermal.is_some(),
            data.stages.post_study.water.is_some(),
        ),
        _ => return,
    };

    check_water_calendar(data, wording, declared_water, ctx);
    if declared_thermal
        && !data
            .thermals
            .iter()
            .any(|thermal| thermal.anticipated_config.is_some())
    {
        let CalendarWording {
            file, thermal_key, ..
        } = wording;
        ctx.emit(
            &rules::POST_STUDY_THERMAL_CALENDAR_UNUSED,
            file,
            Some(thermal_key),
            format!(
                "{file} declares `{thermal_key}` but no thermal is anticipated; the key has no \
                 effect without an anticipated thermal."
            ),
        );
    }
}

fn check_water_calendar(
    data: &ParsedData,
    wording: CalendarWording,
    declared: bool,
    ctx: &mut ValidationContext,
) {
    let CalendarWording {
        file, water_key, ..
    } = wording;
    if !declared {
        if let Some(hydro) = data
            .hydros
            .iter()
            .filter(|hydro| hydro.declares_travel_time_arc())
            .min_by_key(|hydro| hydro.id)
        {
            ctx.emit(
                &rules::POST_STUDY_WATER_CALENDAR_MISSING,
                file,
                Some(water_key),
                format!(
                    "{file} omits `{water_key}` but hydro {} declares a travel-time arc; declare a \
                     water calendar covering its travel_time_hours.",
                    hydro.id
                ),
            );
        }
        return;
    }

    if !data.hydros.iter().any(Hydro::declares_travel_time_arc) {
        ctx.emit(
            &rules::POST_STUDY_WATER_CALENDAR_UNUSED,
            file,
            Some(water_key),
            format!(
                "{file} declares `{water_key}` but no hydro declares a travel-time arc; the key \
                 has no effect without a travel-time arc."
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
    build_post_study_calendar(water_stages, file, water_key, study_end, ctx);

    if let Some(arc) = deepest_uncovered_travel_time_arc(&data.hydros, water_stages) {
        ctx.emit(
            &rules::POST_STUDY_WATER_CALENDAR_SHORT,
            file,
            Some(water_key),
            format!(
                "`{water_key}` covers {} h but hydro {} declares a travel-time arc of {} h; \
                 extend the water calendar to at least {} h.",
                arc.declared_hours, arc.hydro_id, arc.travel_time_hours, arc.travel_time_hours
            ),
        );
    }
}
