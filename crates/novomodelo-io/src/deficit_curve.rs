//! Single owner of the rules a deficit-segment curve must satisfy.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentField {
    DepthMw,
    Cost,
}

impl SegmentField {
    fn as_str(self) -> &'static str {
        match self {
            Self::DepthMw => "depth_mw",
            Self::Cost => "cost",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CurveLocation {
    Curve,
    Segment(usize, SegmentField),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CurveDefect {
    pub(crate) location: CurveLocation,
    pub(crate) message: String,
}

impl CurveDefect {
    fn segment(index: usize, field: SegmentField, message: String) -> Self {
        Self {
            location: CurveLocation::Segment(index, field),
            message,
        }
    }

    /// `prefix` for `Curve`; `{prefix}[{j}].{field}` for `Segment(j, field)`.
    pub(crate) fn field_path(&self, prefix: &str) -> String {
        match self.location {
            CurveLocation::Curve => prefix.to_string(),
            CurveLocation::Segment(index, field) => {
                format!("{prefix}[{index}].{}", field.as_str())
            }
        }
    }
}

/// Returns the first defect of a `(depth_mw, cost)` curve in segment order.
///
/// The cost, last-segment and monotonicity checks run before the depth checks:
/// a curve failing both keeps the message the cost rules have always reported.
pub(crate) fn check_deficit_curve<I>(segments: I) -> Result<(), CurveDefect>
where
    I: IntoIterator<Item = (Option<f64>, f64)>,
    I::IntoIter: ExactSizeIterator + Clone,
{
    let segments = segments.into_iter();
    let len = segments.len();
    if len == 0 {
        return Err(CurveDefect {
            location: CurveLocation::Curve,
            message: "deficit_segments must not be empty".to_string(),
        });
    }
    let last = len - 1;
    let costs = || segments.clone().map(|(_, cost)| cost);

    if let Some((j, cost)) = costs().enumerate().find(|&(_, cost)| cost <= 0.0) {
        return Err(CurveDefect::segment(
            j,
            SegmentField::Cost,
            format!("penalty value must be > 0.0, got {cost}"),
        ));
    }

    if segments
        .clone()
        .last()
        .is_some_and(|(depth, _)| depth.is_some())
    {
        return Err(CurveDefect::segment(
            last,
            SegmentField::DepthMw,
            "the last deficit segment must have depth_mw: null (uncapped final segment)"
                .to_string(),
        ));
    }

    let steps = costs().zip(costs().skip(1)).enumerate();
    for (i, (prev_cost, curr_cost)) in steps {
        if curr_cost <= prev_cost {
            let j = i + 1;
            return Err(CurveDefect::segment(
                j,
                SegmentField::Cost,
                format!(
                    "deficit segment costs must be monotonically increasing: \
                     segment[{j}].cost ({curr_cost}) must be > segment[{i}].cost ({prev_cost})"
                ),
            ));
        }
    }

    for (j, (depth, _)) in segments.take(last).enumerate() {
        let message = match depth {
            None => "only the last deficit segment may have depth_mw: null".to_string(),
            Some(d) if !d.is_finite() => format!("depth_mw must be finite, got {d}"),
            Some(d) if d < 0.0 => format!("depth_mw must be >= 0.0, got {d}"),
            Some(_) => continue,
        };
        return Err(CurveDefect::segment(j, SegmentField::DepthMw, message));
    }

    Ok(())
}

/// `true` iff `distinct_sorted_indices` is exactly `0, 1, ..., n_segments - 1`.
pub(crate) fn indices_cover_curve(distinct_sorted_indices: &[i32], n_segments: usize) -> bool {
    distinct_sorted_indices.len() == n_segments
        && distinct_sorted_indices
            .iter()
            .enumerate()
            .all(|(position, &index)| usize::try_from(index) == Ok(position))
}

#[cfg(test)]
mod tests {
    use super::{
        CurveDefect, CurveLocation, SegmentField, check_deficit_curve, indices_cover_curve,
    };

    fn refused(location: CurveLocation, message: &str) -> Result<(), CurveDefect> {
        Err(CurveDefect {
            location,
            message: message.to_string(),
        })
    }

    #[test]
    fn empty_curve_is_refused() {
        assert_eq!(
            check_deficit_curve(Vec::<(Option<f64>, f64)>::new()),
            refused(CurveLocation::Curve, "deficit_segments must not be empty"),
        );
    }

    #[test]
    fn legacy_checks_keep_their_precedence() {
        assert_eq!(
            check_deficit_curve([(Some(-5.0), 0.0), (None, 5000.0)]),
            refused(
                CurveLocation::Segment(0, SegmentField::Cost),
                "penalty value must be > 0.0, got 0",
            ),
        );
        assert_eq!(
            check_deficit_curve([(Some(-5.0), 500.0), (Some(1.0), 5000.0)]),
            refused(
                CurveLocation::Segment(1, SegmentField::DepthMw),
                "the last deficit segment must have depth_mw: null (uncapped final segment)",
            ),
        );
        assert_eq!(
            check_deficit_curve([(Some(-5.0), 5000.0), (None, 500.0)]),
            refused(
                CurveLocation::Segment(1, SegmentField::Cost),
                "deficit segment costs must be monotonically increasing: \
                 segment[1].cost (500) must be > segment[0].cost (5000)",
            ),
        );
    }

    #[test]
    fn non_last_null_depth_is_refused() {
        assert_eq!(
            check_deficit_curve([(Some(10.0), 100.0), (None, 200.0), (None, 300.0)]),
            refused(
                CurveLocation::Segment(1, SegmentField::DepthMw),
                "only the last deficit segment may have depth_mw: null",
            ),
        );
    }

    #[test]
    fn non_finite_depth_is_refused() {
        for (depth, shown) in [(f64::NAN, "NaN"), (f64::INFINITY, "inf")] {
            assert_eq!(
                check_deficit_curve([(Some(depth), 500.0), (None, 5000.0)]),
                refused(
                    CurveLocation::Segment(0, SegmentField::DepthMw),
                    &format!("depth_mw must be finite, got {shown}"),
                ),
            );
        }
    }

    #[test]
    fn negative_depth_is_refused() {
        assert_eq!(
            check_deficit_curve([(Some(-5.0), 500.0), (None, 5000.0)]),
            refused(
                CurveLocation::Segment(0, SegmentField::DepthMw),
                "depth_mw must be >= 0.0, got -5",
            ),
        );
    }

    #[test]
    fn zero_depth_is_accepted() {
        assert_eq!(
            check_deficit_curve([(Some(0.0), 500.0), (None, 5000.0)]),
            Ok(()),
        );
    }

    #[test]
    fn indices_cover_curve_accepts_only_zero_to_n() {
        assert!(indices_cover_curve(&[0, 1], 2));
        for (indices, n) in [
            (&[0][..], 2),
            (&[0, 1, 2], 2),
            (&[0, 2], 2),
            (&[1, 2], 2),
            (&[-1, 0], 2),
            (&[], 1),
        ] {
            assert!(!indices_cover_curve(indices, n), "{indices:?} vs {n}");
        }
    }

    #[test]
    fn field_path_renders_curve_and_segment_locations() {
        let curve = CurveDefect {
            location: CurveLocation::Curve,
            message: String::new(),
        };
        let segment = CurveDefect {
            location: CurveLocation::Segment(2, SegmentField::Cost),
            message: String::new(),
        };
        assert_eq!(
            curve.field_path("bus.deficit_segments"),
            "bus.deficit_segments"
        );
        assert_eq!(
            segment.field_path("bus.deficit_segments"),
            "bus.deficit_segments[2].cost",
        );
    }
}
