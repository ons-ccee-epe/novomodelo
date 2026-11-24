# Boundary-conditions input

> **Status:** Proposal (not yet implemented). Replaces `initial_conditions.json`
> and `post_study_stages.json` with one plant-keyed `boundary_conditions.json`,
> and moves the post-horizon calendars into `stages.json`. The change is to the
> input format only: the core types, the LP, the state space and the outputs are
> unchanged.

---

## 1. Problem

A study's boundary data is spread over two files. Each file is grouped by kind
of record, not by plant, and the two use different keys:

| File                      | Content                                                            | Key                                     |
| ------------------------- | ------------------------------------------------------------------ | --------------------------------------- |
| `initial_conditions.json` | `storage`, `filling_storage`, `recent_observations`, `past_defluences` | `hydro_id`                          |
|                           | `past_anticipated_commitments`                                     | `thermal_id` + date window              |
| `post_study_stages.json`  | `thermal_stages[]`, `water_bucket_stages[]`                        | position in the list, `{start_date, duration_hours}` |
|                           | `thermal_bounds[]` (cost/min/max per thermal and post-study stage) | `post_study_stage_index`                |

`post_study_stages.json` mixes stage definitions, which `stages.json` already
owns, with plant data. It also breaks three conventions the rest of the input
follows:

1. **Keyed by list position, not by id.** Every other stage reference resolves
   a declared `id`. `thermal_bounds` joins on `post_study_stage_index`, a
   position in the start-date-sorted list, so inserting an earlier post-study
   stage silently re-points every bound row.
2. **A second date spelling.** `{start_date, duration_hours}` instead of the
   `{start_date, end_date}` that study and pre-study stages use.
3. **One plant's data in three places.** An anticipated thermal's identity and
   lead are in `thermals.json`; its fixed commitments in
   `initial_conditions.json`; its post-horizon cost and limits in
   `post_study_stages.json`.

## 2. Principle

- **Calendars go in `stages.json`.** The two post-horizon calendars become
  id-bearing stage lists next to `pre_study_stages[]`.
- **Values at the edges of the horizon go in `boundary_conditions.json`.** That
  covers the starting state, past history, fixed commitments on either side of
  the horizon, and the cost and limits of post-horizon deliveries. The file is
  grouped by plant, with `hydros` and `thermals` as the top-level keys.
- **Per-stage operating limits stay where they are.** `thermal_bounds.parquet`
  and `hydro_bounds.parquet` keep covering study stages only; post-horizon
  limits are not mixed into them.
- **Nothing is keyed by list position.**

## 3. Target format

### 3.1 `stages.json`: two post-study calendars

```json
{
  "pre_study_stages": [
    { "id": -1, "start_date": "2025-12-01", "end_date": "2026-01-01" }
  ],
  "stages": [ /* study stages */ ],
  "post_study_thermal_stages": [
    { "id": 120, "start_date": "2026-11-01", "end_date": "2026-11-07" },
    { "id": 121, "start_date": "2026-11-07", "end_date": "2026-11-14" }
  ],
  "post_study_water_stages": [
    { "id": 200, "start_date": "2026-11-01", "end_date": "2026-12-01" }
  ]
}
```

- **Two calendars, kept separate.** Thermal deliveries and water-bucket slots
  are dated by different calendars, as they are today.
- **Same shape as `pre_study_stages[]`:** `{ id, start_date, end_date }`. There
  are no blocks, openings, risk measure or seasons, because these stages are
  never solved. Hours are `end_date − start_date`; `duration_hours` is dropped.
- **Ids are unique across all four stage sets.** That is pre-study, study and
  both post-study lists. The existing collision check extends to them. The
  water stages carry ids only to keep one shape and one uniqueness rule;
  nothing references them.
- **Each calendar is validated on its own:**
  - its first `start_date` equals the study horizon end;
  - each `end_date` equals the next stage's `start_date`;
  - ids are distinct, and the list is ordered by `start_date`.
- **The water calendar must still cover the deepest travel time.** This is the
  existing check in `setup/mod.rs` (`UncoveredArc`).

### 3.2 `boundary_conditions.json`

```json
{
  "$schema": ".../boundary_conditions.schema.json",
  "hydros": [
    {
      "hydro_id": 0,
      "storage_hm3": 80.0,
      "recent_inflows": [
        { "start_date": "2025-12-15", "end_date": "2026-01-01", "value_m3s": 12.0 }
      ],
      "past_defluences": [
        { "start_date": "2025-11-01", "end_date": "2026-01-01", "value_m3s": 20.0 }
      ]
    }
  ],
  "thermals": [
    {
      "thermal_id": 86,
      "commitments": [
        { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 0.0 },
        { "start_date": "2026-11-01", "end_date": "2026-11-07", "value_mw": 120.0 }
      ],
      "post_study_bounds": [
        { "stage_id": 121, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }
  ]
}
```

**Hydros**

- **`storage_hm3` replaces the `storage` / `filling_storage` split.** Whether a
  plant is filling is already declared by `hydros.json`'s `filling` field, so
  the validator picks the rule set from there: the normal storage rules, or the
  filling-seed rules in `validation/semantic/hydro.rs`.
- **`recent_inflows`** is today's `recent_observations` and **`past_defluences`**
  is unchanged. Both keep their `[start_date, end_date)` window shape and their
  existing validation (`windowed_history::validate_windowed_records`).

**Thermals**

- **`commitments` is today's `past_anticipated_commitments`, unchanged in
  meaning.** These are dated `[start_date, end_date)` windows with `value_mw`,
  covering two cases with the same shape:
  - the leading in-study delivery stages decided before the study (the left
    edge);
  - the post-horizon deliveries decided before the study (the right edge).

  The "past" prefix is dropped because the windows can fall after the horizon.
- **`post_study_bounds` is today's `thermal_bounds`, re-keyed.** It is
  `stage_id` (a `post_study_thermal_stages` id) instead of
  `post_study_stage_index`. All three fields stay required, and every reached
  stage that needs a row (§4) must declare one, exactly as today.

**Both lists**

- Plant ids are unique within each list. Lists and windows are sorted after
  reading, so declaration order does not affect results.
- A plant with no boundary data may be left out. The existing presence rules
  still apply, so an anticipated thermal must still declare its commitments.

## 4. How commitments and post-study bounds relate

The two thermal lists are **disjoint by design**: each `(plant, post-study
stage)` cell is covered by at most one of them. The validator's
`classify_deliveries` (`validation/semantic/thermal.rs`) puts every post-study
stage of an anticipated plant into one class, from the plant's lead over the
study calendar followed by the thermal post-study calendar:

| Class                    | Delivery decided              | Required input                                   |
| ------------------------ | ----------------------------- | ------------------------------------------------ |
| `fixed_post_study`       | before the study              | a `commitments` window covering it exactly; no bound |
| `carried`                | by an in-study decision       | a `post_study_bounds` row; no window             |
| `beyond_reach`           | after the horizon             | neither                                          |
| `commissioning_inactive` | plant out of service          | neither; a window there must be 0 MW             |

- **A commitment is a fixed value, not a decision.** It never enters an LP. It
  is used only by validation, a constant folded into the boundary cuts, and the
  outputs, and it is booked as sunk cost.
- **A bound row prices and limits a delivery the study decides.**
- **The bridge's DECOMP converter produces exactly this split.** It writes the
  weeks already committed past the horizon as commitment windows, and only the
  stages after them as bound rows (`classify_gnl_windows`, `convert_gnl`).

The rules carry over unchanged in meaning and are keyed by stage id:

- **Rule 1:** each `carried` stage has a bound row.
- **V2:** commitment windows cover each `fixed_post_study` stage exactly.
- **V3:** no window covers a `carried` or `beyond_reach` stage.
- **V5:** a window on a `commissioning_inactive` stage has a value of 0.
- **The horizon-straddle guard:** no window crosses the horizon end.
- **Committed-value limits:**
  - an in-study value is checked against that stage's resolved generation
    limits, `thermal_bounds.parquet` overrides included;
  - a post-horizon value is checked against the plant's fixed
    `[min_generation_mw, max_generation_mw]`, since that cell has no bound row.

Grouping both lists under the plant lets one error message state the whole
picture. For example: "thermal 86: stage 121 is decided in-study and needs a
`post_study_bounds` row".

## 5. What `pre_study_stages` feed

Nothing in `boundary_conditions.json` depends on pre-study stages, so this
proposal leaves them as they are. For reference, they are read only by the
inflow model's lag machinery:

- **Lag statistics in the inflow model (`PrecomputedPar::build`).** Inflow-model
  rows at negative `stage_id`, and the season of each pre-study stage, supply
  the statistics of lags that reach back before the study.
- **Estimation (`synthesize_prestudy_stages`).** It creates missing lag-season
  stages for a partial-year study.
- **The policy season manifest (`build_season_manifest`).**
- **Dates of inflow-lag slots in policy export (`lag_reference_anchor`).**
- **`Stage::index`.** It runs over the full stage list, pre-study stages
  included.

Everything else uses dates or the study start, not pre-study stages:

- anticipated "decided before the study" comes from the lead measured from the
  study start;
- commitment windows, `past_defluences`, recent inflows and the inflow history
  are all matched by date.

## 6. The legacy reader

The old files remain accepted for a deprecation period.

| Files present                                                    | Reader                                                         |
| ---------------------------------------------------------------- | -------------------------------------------------------------- |
| `boundary_conditions.json`                                       | new                                                            |
| `initial_conditions.json` (± `post_study_stages.json`)           | legacy, with one deprecation warning                           |
| `boundary_conditions.json` with `initial_conditions.json` or `post_study_stages.json` | error                                     |
| neither `boundary_conditions.json` nor `initial_conditions.json` | error (the existing missing-file error)                        |
| `post_study_stages.json` + any `post_study_*` list in `stages.json` | error (mixed)                                               |

- **Both readers build the same core types:** `InitialConditions` and
  `PostStudyStages`, with `post_study_stage_index` worked out from `stage_id`
  for the new format. Nothing downstream can tell which format a deck used.
- **The legacy reader keeps its current behaviour.** That includes the
  `PostStudyFormat` handling (`Legacy` / `Current` / `Mixed`) and the
  `WaterBucketStages::FromThermalStages` fallback.
- **Removing the legacy reader is a separate, later decision.**

## 7. Affected surfaces

| Crate / repo            | Change                                                                                                                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `novomodelo-io`         | New `boundary_conditions.rs` parser. `initial_conditions.rs` and `post_study_stages.rs` move into a legacy module. `stages.rs` reads both post-study calendars. `pipeline.rs` picks the reader. `validation/semantic/{thermal,hydro,scenarios,inflow_seeding}.rs` change their file and field names, and `thermal.rs` switches to stage ids. `schema.rs` exports the new schema and keeps the legacy ones. |
| `novomodelo-core`       | No structural change. Doc comments that name the old files are updated.                                                                                                                                                                                     |
| `novomodelo-sddp`, `novomodelo-stochastic` | No behaviour change. New-format fixtures are added beside the legacy ones (`tests/fixtures/**`, `tests/unswept_fixtures/**`, `test_support`).                                                                                            |
| `novomodelo-cli`        | `templates.rs` / `build.rs` write the new format.                                                                                                                                                                                                            |
| `novomodelo-python`     | Reads through `novomodelo-io`, so output parity holds without extra work. Any file-name mentions are updated.                                                                                                                                               |
| `schemas/`              | Add `boundary_conditions.schema.json` and update `stages.schema.json`. The legacy schemas stay while the legacy reader exists.                                                                                                                              |
| `novomodelo-bridge`     | The DECOMP (`converters/anticipated.py`, `travel_time.py`, `pipeline.py`) and NEWAVE (`converters/inflow_windows.py`, `pipeline.py`) emitters write the new format. The switch ships after the first `novomodelo` release that contains the new reader.                     |
| Decks / docs            | Migrate the example decks with a converter, alongside `scripts/migrate/post_study_stages.py`. Update `anticipated-thermals-and-water-travel-time.md` (it names `thermal_stages` / `water_bucket_stages`) and the docs site. The CHANGELOG entry is a deprecation plus a new format. |

## 8. Migration

The converter is mechanical:

1. **Calendars.**
   - Move `thermal_stages[]` (or the deprecated `stages[]`) to
     `post_study_thermal_stages[]`, and `water_bucket_stages[]` to
     `post_study_water_stages[]`.
   - Give each stage an ascending id past every study id, distinct across both
     lists.
   - Convert `duration_hours` to `end_date`.
2. **Hydros.** Fold `storage`, `filling_storage`, `recent_observations` and
   `past_defluences` into one entry per `hydro_id`.
3. **Thermals.** Fold `past_anticipated_commitments` into each plant's
   `commitments`. Turn each `thermal_bounds` row into a `post_study_bounds` row,
   replacing `post_study_stage_index` with the id step 1 assigned.

**Decks without a water calendar.** A legacy deck that omits
`water_bucket_stages` resolves to `WaterDeliveryBasis::Legacy`: the thermal
calendar, padded by the bucket topology. A declared water calendar is used
exactly, with no padding, so copying the thermal calendar into
`post_study_water_stages` is **not** guaranteed to give the same result for a
deck with travel-time arcs. For such a deck the converter must write out the
padded calendar the bucket topology would have built, or leave the deck on the
legacy reader. A deck without travel-time arcs is unaffected.

## 9. Acceptance

1. **Migration changes no result.** Every migrated deck's outputs are
   bit-identical to its legacy copy. A moved hash means something must be
   investigated; it is never re-baselined.
2. **The two readers agree.** Each legacy fixture gets a new-format twin, and
   the pair must produce identical core types and identical solved outputs.
3. **Declaration order doesn't matter.** Order-invariance tests cover the new
   file's lists: plants, windows and bound rows.
4. **Schemas match.** The CI `schemas` job is clean.

## 10. Non-goals

- **No change to the state space, the anticipated ring, the water-bucket ring,
  boundary pricing or outputs.**
- **No default for missing `post_study_bounds` fields** (falling back to the
  plant's own cost and limits). It would be a separate change to what decks must
  declare.
- **No post-horizon rows in `thermal_bounds.parquet` / `hydro_bounds.parquet`.**
- **No change to `pre_study_stages`.**

## 11. Resolved format decisions

These are format rules, fixed for the reader and the writers.

- **Field names.** The names in §3 are final: `post_study_thermal_stages`,
  `post_study_water_stages`, `post_study_bounds`, `commitments`,
  `recent_inflows` and `storage_hm3`.
- **Post-study ids.** Thermal post-study stage ids equal
  `max(study id) + 1 + j`, with `j` the stage's position in `start_date`
  order. Water post-study stage ids are unique and greater than every study id.
- **Declaration order.** A post-study list declared out of `start_date` order is
  accepted and sorted on reading. A duplicate `start_date` within a list is an
  error.
- **Water calendar.** A travel-time arc without `post_study_water_stages` is an
  error in the new format. The legacy `FromThermalStages` fallback stays only in
  the legacy reader. The delivery axis the validator checks (`build_extended_delivery_axis`)
  is built from the declared calendar.
- **Optional and required values.** `hydros`, `thermals`, `storage_hm3` and the
  window lists are optional. The three `post_study_bounds` values
  (`cost_per_mwh`, `min_mw`, `max_mw`) are required on every row.
- **Unknown plant ids.** A plant id that `hydros.json` or `thermals.json` does
  not declare is an error, not ignored.
- **Durations.** Post-study stage durations must be whole days. A deck with a
  post-study stage of another length stays on the legacy reader.
  `parse_post_study_file` is the legacy entry point for such decks.

Open item:

- **The bridge's clamp for already-committed weeks after the horizon.** It
  clamps those windows into the last study stage's `tg` capacity, but prices
  the following stages with the post-study month's own `tg` record. If DECOMP
  declares a different record for that month, the clamp uses the wrong month.
  novomodelo only checks such windows against the plant's fixed limits, so it
  doesn't catch this. Worth confirming on a real deck while the emitter is
  being changed.

## Alternatives considered

An earlier proposal for this input surface took two other routes. This design
does not take either.

- **One post-horizon calendar for both thermal delivery and water travel time.**
  The proposal put a single `post_study_stages[]` list in `stages.json`. The two
  calendars answer different questions. The thermal one dates the deliveries an
  anticipated plant can still make after the horizon, and its length follows the
  plant leads. The water one dates the buckets that hold water in transit, and it
  must cover the deepest travel-time arc. A shared list either pads thermal
  stages that carry no delivery or truncates the water axis, so the calendars
  stay separate.
- **Post-horizon delivery bounds on `thermals.json`.** The proposal put a
  per-plant `anticipated_config.post_horizon_delivery[]` list, with fields
  defaulting to the plant's own cost and limits. Cost and limits of a
  post-horizon delivery are boundary data, so they belong with the other
  per-plant boundary records in `boundary_conditions.json` as
  `post_study_bounds`. Keeping them there leaves `anticipated_config` as the
  lead declaration only, and keeps the defaulting question (§10) out of the
  plant definition.
