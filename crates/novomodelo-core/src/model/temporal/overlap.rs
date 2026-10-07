//! Interval-overlap primitive on an hour-resolution stage clock.
//!
//! Pure hour arithmetic: no calendar cycle, no dates, no block mode.

/// Intersect `[window_start_hours, window_start_hours + window_width_hours)`
/// against consecutive periods whose durations are `stage_lengths_hours`
/// (period 0 starts at hour `0.0`; each subsequent period starts where the
/// previous one ends).
///
/// Returns one overlap-hours entry per period, from period 0 through the
/// deepest overlapped period — a contiguous `0..=L` range, not a count: a
/// period between two overlapped periods that the window happens to miss
/// keeps its `0.0` entry rather than being dropped. An empty return means the
/// window overlaps no period at all.
///
/// `window_width_hours` and every `stage_lengths_hours` entry must be `>
/// 0.0`, and all inputs must be finite; these are caller preconditions
/// checked with `debug_assert!` in debug builds only.
#[must_use]
pub fn window_period_overlaps(
    window_start_hours: f64,
    window_width_hours: f64,
    stage_lengths_hours: &[f64],
) -> Vec<f64> {
    let mut overlaps: Vec<f64> = Vec::with_capacity(stage_lengths_hours.len());
    overlaps.extend(period_overlaps_hours(
        window_start_hours,
        window_width_hours,
        stage_lengths_hours,
    ));

    match overlaps.iter().rposition(|&overlap| overlap > 0.0) {
        Some(last_nonzero) => {
            overlaps.truncate(last_nonzero + 1);
            overlaps
        }
        None => Vec::new(),
    }
}

/// Whether the window reaches at least one period — the same `overlap > 0.0`
/// sweep as [`window_period_overlaps`], equal to
/// `!window_period_overlaps(...).is_empty()`, short-circuiting on the first
/// match without allocating.
#[must_use]
pub fn window_reaches_any_period(
    window_start_hours: f64,
    window_width_hours: f64,
    stage_lengths_hours: &[f64],
) -> bool {
    period_overlaps_hours(window_start_hours, window_width_hours, stage_lengths_hours)
        .any(|overlap| overlap > 0.0)
}

/// Index of the deepest period the window reaches, or `0` if none — the same
/// sweep as [`window_period_overlaps`], equal to
/// `window_period_overlaps(...).len().saturating_sub(1)`, without allocating.
#[must_use]
pub fn window_period_reach_depth(
    window_start_hours: f64,
    window_width_hours: f64,
    stage_lengths_hours: &[f64],
) -> usize {
    period_overlaps_hours(window_start_hours, window_width_hours, stage_lengths_hours)
        .enumerate()
        .fold(
            0,
            |depth, (index, overlap)| {
                if overlap > 0.0 { index } else { depth }
            },
        )
}

/// Shared sweep behind [`window_period_overlaps`], [`window_reaches_any_period`],
/// and [`window_period_reach_depth`]: one overlap-hours value per period, from
/// period 0 through the last period starting before the window's end.
fn period_overlaps_hours(
    window_start_hours: f64,
    window_width_hours: f64,
    stage_lengths_hours: &[f64],
) -> impl Iterator<Item = f64> + '_ {
    debug_assert!(
        window_start_hours.is_finite(),
        "window_start_hours must be finite"
    );
    debug_assert!(
        window_width_hours.is_finite() && window_width_hours > 0.0,
        "window_width_hours must be finite and > 0.0"
    );
    debug_assert!(
        stage_lengths_hours
            .iter()
            .all(|&length| length.is_finite() && length > 0.0),
        "every stage_lengths_hours entry must be finite and > 0.0"
    );

    let window_end_hours = window_start_hours + window_width_hours;
    let mut period_start = 0.0_f64;
    stage_lengths_hours.iter().map_while(move |&length| {
        if period_start >= window_end_hours {
            return None;
        }
        let period_end = period_start + length;
        let overlap_start = window_start_hours.max(period_start);
        let overlap_end = window_end_hours.min(period_end);
        let overlap = (overlap_end - overlap_start).max(0.0);
        period_start = period_end;
        Some(overlap)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: got {actual:?}, expected {expected:?}"
        );
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() < TOL,
                "value mismatch: got {actual:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn test_uniform_stages_shallow_window() {
        let overlaps = window_period_overlaps(0.0, 360.0, &[168.0, 168.0, 168.0]);
        assert_close(&overlaps, &[168.0, 168.0, 24.0]);
    }

    #[test]
    fn test_non_uniform_stages_depth_three() {
        let overlaps = window_period_overlaps(360.0, 720.0, &[720.0, 168.0, 168.0, 168.0]);
        assert_eq!(overlaps.len(), 4);
        assert!(overlaps[3] > 0.0, "index 3 must be nonzero: {overlaps:?}");
    }

    #[test]
    fn test_skipped_intermediate_stage_stays_contiguous() {
        let overlaps = window_period_overlaps(336.0, 200.0, &[168.0, 168.0, 168.0, 168.0]);
        assert_eq!(overlaps.len(), 4);
        assert_eq!(
            overlaps[1], 0.0,
            "index 1 must be exactly 0.0: {overlaps:?}"
        );
        assert_close(&overlaps, &[0.0, 0.0, 168.0, 32.0]);
    }

    #[test]
    fn test_window_past_all_stages_returns_empty() {
        let overlaps = window_period_overlaps(1000.0, 10.0, &[168.0, 168.0]);
        assert!(overlaps.is_empty(), "expected empty, got {overlaps:?}");
    }

    #[test]
    fn test_exact_boundary_window_returns_depth_zero() {
        let overlaps = window_period_overlaps(0.0, 720.0, &[720.0, 168.0, 168.0]);
        assert_close(&overlaps, &[720.0]);
    }

    #[test]
    fn test_predicate_and_depth_match_window_period_overlaps_across_golden_inputs() {
        let cases: [(f64, f64, &[f64]); 5] = [
            (0.0, 360.0, &[168.0, 168.0, 168.0]),
            (360.0, 720.0, &[720.0, 168.0, 168.0, 168.0]),
            (336.0, 200.0, &[168.0, 168.0, 168.0, 168.0]),
            (1000.0, 10.0, &[168.0, 168.0]),
            (0.0, 720.0, &[720.0, 168.0, 168.0]),
        ];
        for &(start, width, lengths) in &cases {
            let overlaps = window_period_overlaps(start, width, lengths);
            assert_eq!(
                window_reaches_any_period(start, width, lengths),
                !overlaps.is_empty(),
                "predicate mismatch for ({start}, {width}, {lengths:?})"
            );
            assert_eq!(
                window_period_reach_depth(start, width, lengths),
                overlaps.len().saturating_sub(1),
                "depth mismatch for ({start}, {width}, {lengths:?})"
            );
        }
    }
}
