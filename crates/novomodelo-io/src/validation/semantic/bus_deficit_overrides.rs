//! Layer 5a — bus deficit override rules that need the bus's base curve.

use std::collections::HashMap;

use novomodelo_core::Bus;

use super::super::{ValidationContext, rules, schema::ParsedData};
use crate::constraints::BusDeficitOverrideRow;
use crate::deficit_curve::{CurveLocation, check_deficit_curve, indices_cover_curve};

const FILE: &str = "constraints/penalty_overrides_bus_deficit.parquet";

/// Rule 69: rejects an override key `(bus_id, stage_id, block_id)` whose distinct
/// `segment_index` values are not exactly `0..n`, `n` being the segment count of
/// the bus's base curve. A repeated index is rule 36's, and a key of an undeclared
/// bus is skipped — referential validation owns it.
pub(super) fn check_override_key_completeness(data: &ParsedData, ctx: &mut ValidationContext) {
    for (bus, first, rows) in override_keys(data) {
        let n = bus.deficit_segments.len();
        let distinct = distinct_indices(rows);
        if indices_cover_curve(&distinct, n) {
            continue;
        }

        let listed = distinct
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        ctx.emit(
            &rules::SEMANTIC_BUS_DEFICIT_OVERRIDE_INCOMPLETE,
            FILE,
            Some(key_entity(first)),
            format!(
                "{} lists segment_index [{listed}] but bus {} has {n} deficit segment(s) \
                 (valid range: 0..{n}, each exactly once)",
                key_site(first),
                first.bus_id.0
            ),
        );
    }
}

/// Rule 70: pairs a complete, duplicate-free override key's depths with the
/// bus's base costs and rejects the curve when it breaks a deficit-curve rule.
/// Keys refused by rules 36 and 69 are skipped, as are keys of an undeclared bus.
pub(super) fn check_override_curves(data: &ParsedData, ctx: &mut ValidationContext) {
    for (bus, first, rows) in override_keys(data) {
        let n = bus.deficit_segments.len();
        if rows.len() != n || !indices_cover_curve(&distinct_indices(rows), n) {
            continue;
        }

        let curve = rows
            .iter()
            .zip(&bus.deficit_segments)
            .map(|(row, segment)| (row.depth_mw, segment.cost_per_mwh));
        let Err(defect) = check_deficit_curve(curve) else {
            continue;
        };

        let segment = match defect.location {
            CurveLocation::Segment(j, _) => format!("segment_index={j}: "),
            CurveLocation::Curve => String::new(),
        };
        ctx.emit(
            &rules::SEMANTIC_BUS_DEFICIT_OVERRIDE_CURVE_INVALID,
            FILE,
            Some(key_entity(first)),
            format!("{}: {segment}{}", key_site(first), defect.message),
        );
    }
}

/// Each override key of a declared bus as `(bus, first row, all rows)`. Rows are
/// sorted by key then `segment_index`, so a key's rows are contiguous.
fn override_keys(
    data: &ParsedData,
) -> impl Iterator<Item = (&Bus, &BusDeficitOverrideRow, &[BusDeficitOverrideRow])> {
    let buses: HashMap<i32, &Bus> = data.buses.iter().map(|bus| (bus.id.0, bus)).collect();
    data.penalty_overrides_bus_deficit
        .chunk_by(|a, b| (a.bus_id, a.stage_id, a.block_id) == (b.bus_id, b.stage_id, b.block_id))
        .filter_map(move |rows| {
            let first = rows.first()?;
            Some((*buses.get(&first.bus_id.0)?, first, rows))
        })
}

fn distinct_indices(rows: &[BusDeficitOverrideRow]) -> Vec<i32> {
    let mut distinct: Vec<i32> = rows.iter().map(|row| row.segment_index).collect();
    distinct.dedup();
    distinct
}

fn key_entity(first: &BusDeficitOverrideRow) -> String {
    format!("bus_id={}, stage_id={}", first.bus_id.0, first.stage_id)
}

fn key_site(first: &BusDeficitOverrideRow) -> String {
    let block = first
        .block_id
        .map_or_else(|| "stage-wide".to_string(), |b| format!("block_id={b}"));
    format!(
        "Bus {}: penalty_overrides_bus_deficit curve at stage_id={}, {block}",
        first.bus_id.0, first.stage_id
    )
}

#[cfg(test)]
mod tests {
    use novomodelo_core::{Bus, EntityId};

    use super::super::super::{ErrorKind, ValidationContext};
    use super::super::validate_semantic_hydro_thermal;
    use super::{FILE, check_override_curves, check_override_key_completeness};
    use crate::ValidationEntry;
    use crate::constraints::BusDeficitOverrideRow;
    use crate::test_support::{base_parsed_data, make_bus_with_segments, make_stages};
    use crate::validation::schema::ParsedData;

    fn key_rows(
        bus_id: i32,
        stage_id: i32,
        block_id: Option<i32>,
        indices: &[i32],
    ) -> Vec<BusDeficitOverrideRow> {
        indices
            .iter()
            .map(|&segment_index| BusDeficitOverrideRow {
                bus_id: EntityId::from(bus_id),
                stage_id,
                block_id,
                segment_index,
                depth_mw: None,
            })
            .collect()
    }

    fn depth_rows(
        bus_id: i32,
        stage_id: i32,
        block_id: Option<i32>,
        entries: &[(i32, Option<f64>)],
    ) -> Vec<BusDeficitOverrideRow> {
        entries
            .iter()
            .map(|&(segment_index, depth_mw)| BusDeficitOverrideRow {
                bus_id: EntityId::from(bus_id),
                stage_id,
                block_id,
                segment_index,
                depth_mw,
            })
            .collect()
    }

    fn data(buses: Vec<Bus>, rows: Vec<BusDeficitOverrideRow>) -> ParsedData {
        ParsedData {
            penalty_overrides_bus_deficit: rows,
            ..base_parsed_data(make_stages(vec![0, 1]), buses)
        }
    }

    fn completeness_findings(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        check_override_key_completeness(data, &mut ctx);
        ctx.errors().into_iter().cloned().collect()
    }

    fn curve_findings(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        check_override_curves(data, &mut ctx);
        ctx.errors().into_iter().cloned().collect()
    }

    #[test]
    fn missing_segment_index_is_refused() {
        let findings = completeness_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            key_rows(1, 0, None, &[0]),
        ));

        assert_eq!(findings.len(), 1, "{findings:?}");
        let finding = &findings[0];
        assert_eq!(finding.kind, ErrorKind::BusinessRuleViolation);
        assert_eq!(finding.file, std::path::Path::new(FILE));
        assert_eq!(finding.entity.as_deref(), Some("bus_id=1, stage_id=0"));
        assert_eq!(
            finding.message,
            "Bus 1: penalty_overrides_bus_deficit curve at stage_id=0, stage-wide lists \
             segment_index [0] but bus 1 has 2 deficit segment(s) (valid range: 0..2, each \
             exactly once)"
        );
    }

    #[test]
    fn extra_or_gapped_segment_index_is_refused() {
        let buses = vec![make_bus_with_segments(1, 2)];

        let extra =
            completeness_findings(&data(buses.clone(), key_rows(1, 0, Some(1), &[0, 1, 2])));
        assert_eq!(extra.len(), 1, "{extra:?}");
        assert!(
            extra[0]
                .message
                .contains("stage_id=0, block_id=1 lists segment_index [0, 1, 2]"),
            "{}",
            extra[0].message
        );

        let gapped = completeness_findings(&data(buses, key_rows(1, 0, Some(1), &[0, 2])));
        assert_eq!(gapped.len(), 1, "{gapped:?}");
        assert!(
            gapped[0].message.contains("lists segment_index [0, 2]"),
            "{}",
            gapped[0].message
        );
    }

    #[test]
    fn repeated_index_is_not_reported_as_incomplete() {
        let findings = completeness_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            key_rows(1, 0, None, &[0, 1, 1]),
        ));

        assert!(findings.is_empty(), "{findings:?}");
    }

    #[test]
    fn unknown_bus_key_is_skipped() {
        let findings = completeness_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            key_rows(9, 0, None, &[0]),
        ));

        assert!(findings.is_empty(), "{findings:?}");
    }

    #[test]
    fn each_bus_uses_its_own_segment_count() {
        let buses = vec![make_bus_with_segments(1, 2), make_bus_with_segments(2, 3)];
        let rows = |bus_2_indices: &[i32]| {
            let mut rows = key_rows(1, 0, None, &[0, 1]);
            rows.extend(key_rows(2, 0, None, bus_2_indices));
            rows
        };

        let complete = completeness_findings(&data(buses.clone(), rows(&[0, 1, 2])));
        assert!(complete.is_empty(), "{complete:?}");

        let short = completeness_findings(&data(buses, rows(&[0, 1])));
        assert_eq!(short.len(), 1, "{short:?}");
        assert!(
            short[0].message.contains("bus 2 has 3 deficit segment(s)"),
            "{}",
            short[0].message
        );
    }

    #[test]
    fn completeness_rule_runs_in_layer_5a() {
        let data = data(
            vec![make_bus_with_segments(1, 2)],
            key_rows(1, 0, None, &[0]),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);

        let incomplete: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.message
                    .contains("lists segment_index [0] but bus 1 has 2")
            })
            .collect();
        assert_eq!(incomplete.len(), 1, "{incomplete:?}");
    }

    #[test]
    fn zero_depth_override_is_accepted() {
        let findings = curve_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            depth_rows(1, 0, None, &[(0, Some(0.0)), (1, None)]),
        ));

        assert!(findings.is_empty(), "{findings:?}");
    }

    #[test]
    fn negative_depth_override_is_refused() {
        let findings = curve_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            depth_rows(1, 0, None, &[(0, Some(-1.0)), (1, None)]),
        ));

        assert_eq!(findings.len(), 1, "{findings:?}");
        let finding = &findings[0];
        assert_eq!(finding.kind, ErrorKind::InvalidValue);
        assert_eq!(finding.file, std::path::Path::new(FILE));
        assert_eq!(finding.entity.as_deref(), Some("bus_id=1, stage_id=0"));
        assert_eq!(
            finding.message,
            "Bus 1: penalty_overrides_bus_deficit curve at stage_id=0, stage-wide: \
             segment_index=0: depth_mw must be >= 0.0, got -1"
        );
    }

    #[test]
    fn non_finite_depth_override_is_refused() {
        for depth in [f64::NAN, f64::INFINITY] {
            let findings = curve_findings(&data(
                vec![make_bus_with_segments(1, 2)],
                depth_rows(1, 0, None, &[(0, Some(depth)), (1, None)]),
            ));

            assert_eq!(findings.len(), 1, "{depth}: {findings:?}");
            assert!(
                findings[0].message.contains("depth_mw must be finite"),
                "{}",
                findings[0].message
            );
        }
    }

    #[test]
    fn misplaced_uncapped_segment_override_is_refused() {
        let early_null = curve_findings(&data(
            vec![make_bus_with_segments(1, 3)],
            depth_rows(1, 0, None, &[(0, Some(100.0)), (1, None), (2, None)]),
        ));
        assert_eq!(early_null.len(), 1, "{early_null:?}");
        assert!(
            early_null[0]
                .message
                .contains("segment_index=1: only the last deficit segment may have depth_mw: null"),
            "{}",
            early_null[0].message
        );

        let capped_last = curve_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            depth_rows(1, 0, None, &[(0, Some(100.0)), (1, Some(200.0))]),
        ));
        assert_eq!(capped_last.len(), 1, "{capped_last:?}");
        assert!(
            capped_last[0]
                .message
                .contains("segment_index=1: the last deficit segment must have depth_mw: null"),
            "{}",
            capped_last[0].message
        );
    }

    #[test]
    fn block_key_defect_names_the_block() {
        let findings = curve_findings(&data(
            vec![make_bus_with_segments(1, 2)],
            depth_rows(1, 0, Some(1), &[(0, Some(-5.0)), (1, None)]),
        ));

        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0]
                .message
                .contains("stage_id=0, block_id=1: segment_index=0"),
            "{}",
            findings[0].message
        );
    }

    #[test]
    fn refused_keys_are_not_curve_checked() {
        let buses = vec![make_bus_with_segments(1, 2)];

        let incomplete = curve_findings(&data(
            buses.clone(),
            depth_rows(1, 0, None, &[(0, Some(-1.0))]),
        ));
        assert!(incomplete.is_empty(), "{incomplete:?}");

        let duplicate = curve_findings(&data(
            buses,
            depth_rows(1, 0, None, &[(0, Some(-1.0)), (1, None), (1, None)]),
        ));
        assert!(duplicate.is_empty(), "{duplicate:?}");
    }
}
