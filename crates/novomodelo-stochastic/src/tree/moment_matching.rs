//! Per-coordinate moment matching of a generated stage's openings.
//!
//! The target variance of an inflow coordinate comes from the correlation
//! factor the stage applies. Matching runs per stage; every node of a stage
//! shares that stage's openings, so per stage equals per node. Noise-group
//! copies inherit the matched values of the stage they copy.

/// Moment-matching mode for the openings of a generated tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MomentMatching {
    /// Keep the generated openings as drawn.
    #[default]
    None,
    /// Shift and scale each inflow coordinate of each generated stage over its
    /// openings to mean 0 and the variance its correlation factor implies.
    PerCoordinate,
}

/// Standard-deviation ratio at or below which a coordinate counts as constant.
const DEGENERATE_SD_RATIO: f64 = 1e-12;

/// Pre-correction statistics of one matched stage.
///
/// The three ratios cover only the `rated` coordinates, those with a positive
/// target standard deviation; `untouched` counts coordinates left as drawn
/// because their drawn values are constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StageMomentStats {
    /// Largest `|mean| / sigma`.
    pub(crate) max_abs_mean_ratio: f64,
    /// Smallest `sample sd / sigma`.
    pub(crate) min_sd_ratio: f64,
    /// Largest `sample sd / sigma`.
    pub(crate) max_sd_ratio: f64,
    /// Coordinates that entered the ratios.
    pub(crate) rated: usize,
    /// Coordinates left as drawn.
    pub(crate) untouched: usize,
}

impl StageMomentStats {
    /// Identity of folding stats over several stages: no coordinate rated.
    pub(crate) const UNRATED: Self = Self {
        max_abs_mean_ratio: 0.0,
        min_sd_ratio: f64::INFINITY,
        max_sd_ratio: 0.0,
        rated: 0,
        untouched: 0,
    };
}

/// Shifts and scales coordinates `0..target_variance.len()` of one stage's
/// openings to population mean 0 and variance `target_variance[c]`, in place.
///
/// `stage_slice` is opening-major: coordinate `c` of opening `k` is
/// `stage_slice[k * dim + c]`. Coordinates at or above
/// `target_variance.len()` are never read or written. A stage with fewer than
/// two openings, and a coordinate whose drawn values are constant, are left
/// as drawn. The summation order is fixed (serial, ascending opening) for
/// bitwise reproducibility.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn match_stage_moments(
    stage_slice: &mut [f64],
    n_openings: usize,
    dim: usize,
    target_variance: &[f64],
) -> StageMomentStats {
    debug_assert_eq!(stage_slice.len(), n_openings * dim);
    debug_assert!(target_variance.len() <= dim);
    if n_openings < 2 {
        return StageMomentStats::UNRATED;
    }

    let count = n_openings as f64;
    let mut stats = StageMomentStats::UNRATED;
    for (c, &variance) in target_variance.iter().enumerate() {
        let sigma = variance.sqrt();

        let mut sum = 0.0_f64;
        for &x in stage_slice[c..].iter().step_by(dim) {
            sum += x;
        }
        let mean = sum / count;

        let mut sum_sq = 0.0_f64;
        for &x in stage_slice[c..].iter().step_by(dim) {
            let d = x - mean;
            sum_sq += d * d;
        }
        let sd = (sum_sq / count).sqrt();

        if sigma > 0.0 {
            stats.rated += 1;
            stats.max_abs_mean_ratio = stats.max_abs_mean_ratio.max(mean.abs() / sigma);
            stats.min_sd_ratio = stats.min_sd_ratio.min(sd / sigma);
            stats.max_sd_ratio = stats.max_sd_ratio.max(sd / sigma);
        }

        if sd <= DEGENERATE_SD_RATIO * sigma {
            stats.untouched += 1;
        } else {
            let scale = sigma / sd;
            for x in stage_slice[c..].iter_mut().step_by(dim) {
                *x = (*x - mean) * scale;
            }
        }
    }
    stats
}

/// Folds the per-stage statistics and copy mismatches of one tree generation
/// and reports them once through [`MomentMatchingTally::emit`].
pub(crate) struct MomentMatchingTally {
    untouched_stages: usize,
    untouched_coordinates: usize,
    mismatched_copies: usize,
    rated_stages: usize,
    max_abs_mean_ratio: f64,
    min_sd_ratio: f64,
    max_sd_ratio: f64,
}

impl MomentMatchingTally {
    pub(crate) fn new() -> Self {
        Self {
            untouched_stages: 0,
            untouched_coordinates: 0,
            mismatched_copies: 0,
            rated_stages: 0,
            max_abs_mean_ratio: StageMomentStats::UNRATED.max_abs_mean_ratio,
            min_sd_ratio: StageMomentStats::UNRATED.min_sd_ratio,
            max_sd_ratio: StageMomentStats::UNRATED.max_sd_ratio,
        }
    }

    pub(crate) fn record_stage(&mut self, stats: StageMomentStats) {
        if stats.untouched > 0 {
            self.untouched_stages += 1;
            self.untouched_coordinates += stats.untouched;
        }
        if stats.rated > 0 {
            self.rated_stages += 1;
            self.max_abs_mean_ratio = self.max_abs_mean_ratio.max(stats.max_abs_mean_ratio);
            self.min_sd_ratio = self.min_sd_ratio.min(stats.min_sd_ratio);
            self.max_sd_ratio = self.max_sd_ratio.max(stats.max_sd_ratio);
        }
    }

    pub(crate) fn record_group_k_mismatch(&mut self) {
        self.mismatched_copies += 1;
    }

    fn debug_summary(&self) -> Option<(usize, f64, f64, f64)> {
        (self.rated_stages > 0).then_some((
            self.rated_stages,
            self.max_abs_mean_ratio,
            self.min_sd_ratio,
            self.max_sd_ratio,
        ))
    }

    pub(crate) fn emit(&self) {
        if self.untouched_coordinates > 0 {
            let stages = self.untouched_stages;
            let coordinates = self.untouched_coordinates;
            tracing::warn!(
                stages,
                coordinates,
                "moment matching left {coordinates} inflow coordinate(s) in {stages} stage(s) as drawn: every opening is equal there"
            );
        }
        if self.mismatched_copies > 0 {
            let stages = self.mismatched_copies;
            tracing::warn!(
                stages,
                "{stages} stage(s) copy a noise-group stage with a different opening count; their copied openings are not moment-matched"
            );
        }
        if let Some((stages, max_abs_mean_ratio, min_sd_ratio, max_sd_ratio)) = self.debug_summary()
        {
            tracing::debug!(
                stages,
                max_abs_mean_ratio,
                min_sd_ratio,
                max_sd_ratio,
                "moment matching corrected the inflow openings of {stages} stage(s)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;

    use super::{MomentMatchingTally, StageMomentStats, match_stage_moments};
    use crate::rng_from_seed;

    const SCALES: [f64; 4] = [0.5, 2.0, 3.5, 1.0];
    const OFFSETS: [f64; 4] = [4.0, -1.5, 0.25, 9.0];

    fn drawn(seed: u64, n_openings: usize, dim: usize) -> Vec<f64> {
        let mut rng = rng_from_seed(seed);
        let mut slice = Vec::with_capacity(n_openings * dim);
        for _ in 0..n_openings {
            for c in 0..dim {
                slice.push(rng.random::<f64>() * SCALES[c % 4] + OFFSETS[c % 4]);
            }
        }
        slice
    }

    fn column(slice: &[f64], dim: usize, c: usize) -> Vec<f64> {
        slice[c..].iter().step_by(dim).copied().collect()
    }

    fn population_mean(values: &[f64]) -> f64 {
        let mut sum = 0.0;
        for &x in values {
            sum += x;
        }
        sum / f64::from(u32::try_from(values.len()).unwrap())
    }

    fn population_variance(values: &[f64]) -> f64 {
        let mean = population_mean(values);
        let mut ss = 0.0;
        for &x in values {
            let d = x - mean;
            ss += d * d;
        }
        ss / f64::from(u32::try_from(values.len()).unwrap())
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    fn argsort(values: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        order
    }

    #[test]
    fn matched_coordinates_have_zero_mean_and_the_target_variance() {
        let (k, dim) = (10, 4);
        let target: [f64; 3] = [1.0, 1.3, 0.7];
        let mut slice = drawn(11, k, dim);
        let untouched_before = bits(&column(&slice, dim, 3));

        let stats = match_stage_moments(&mut slice, k, dim, &target);

        for (c, &variance) in target.iter().enumerate() {
            let values = column(&slice, dim, c);
            let mean = population_mean(&values);
            let var = population_variance(&values);
            assert!(
                mean.abs() <= 1e-13 * variance.sqrt(),
                "coordinate {c}: mean {mean}"
            );
            assert!(
                (var - variance).abs() <= 1e-13 * variance,
                "coordinate {c}: variance {var} vs target {variance}"
            );
        }
        assert_eq!(bits(&column(&slice, dim, 3)), untouched_before);
        assert_eq!(stats.rated, 3);
        assert_eq!(stats.untouched, 0);
    }

    #[test]
    fn a_constant_coordinate_is_left_as_drawn_and_counted() {
        let (k, dim) = (5, 2);
        let mut slice = drawn(12, k, dim);
        for opening in slice.chunks_exact_mut(dim) {
            opening[0] = 3.25;
        }
        let constant_before = bits(&column(&slice, dim, 0));

        let stats = match_stage_moments(&mut slice, k, dim, &[1.0, 1.0]);

        assert_eq!(bits(&column(&slice, dim, 0)), constant_before);
        let matched = column(&slice, dim, 1);
        assert!(population_mean(&matched).abs() <= 1e-13);
        assert!((population_variance(&matched) - 1.0).abs() <= 1e-13);
        assert_eq!(stats.untouched, 1);
        assert_eq!(stats.rated, 2);
    }

    #[test]
    fn statistics_are_taken_before_correction() {
        let mut slice = [1.0, 3.0];

        let stats = match_stage_moments(&mut slice, 2, 1, &[1.0]);

        assert_eq!(stats.max_abs_mean_ratio.to_bits(), 2.0_f64.to_bits());
        assert_eq!(stats.min_sd_ratio.to_bits(), 1.0_f64.to_bits());
        assert_eq!(stats.max_sd_ratio.to_bits(), 1.0_f64.to_bits());
        assert_eq!(bits(&slice), bits(&[-1.0, 1.0]));
    }

    #[test]
    fn matching_keeps_each_coordinates_opening_order() {
        let (k, dim) = (7, 3);
        let mut slice = drawn(13, k, dim);
        let before: Vec<Vec<usize>> = (0..dim).map(|c| argsort(&column(&slice, dim, c))).collect();

        match_stage_moments(&mut slice, k, dim, &[2.0, 0.5, 1.0]);

        for (c, order_before) in before.iter().enumerate() {
            assert_eq!(&argsort(&column(&slice, dim, c)), order_before);
        }
    }

    #[test]
    fn a_single_opening_stage_is_left_as_drawn() {
        let dim = 3;
        let mut slice = drawn(14, 1, dim);
        let before = bits(&slice);

        let stats = match_stage_moments(&mut slice, 1, dim, &[1.0, 1.0, 1.0]);

        assert_eq!(bits(&slice), before);
        assert_eq!(stats, StageMomentStats::UNRATED);
        assert_eq!(stats.rated, 0);
        assert_eq!(stats.untouched, 0);
        assert_eq!(stats.min_sd_ratio.to_bits(), f64::INFINITY.to_bits());
    }

    #[test]
    fn the_tally_folds_ratios_from_rated_stages_only() {
        let mut tally = MomentMatchingTally::new();

        tally.record_stage(StageMomentStats::UNRATED);
        assert!(tally.debug_summary().is_none());

        tally.record_stage(StageMomentStats {
            max_abs_mean_ratio: 0.5,
            min_sd_ratio: 0.8,
            max_sd_ratio: 1.2,
            rated: 2,
            untouched: 1,
        });

        let (rated_stages, max_abs_mean_ratio, min_sd_ratio, max_sd_ratio) =
            tally.debug_summary().unwrap();
        assert_eq!(rated_stages, 1);
        assert_eq!(max_abs_mean_ratio.to_bits(), 0.5_f64.to_bits());
        assert_eq!(min_sd_ratio.to_bits(), 0.8_f64.to_bits());
        assert_eq!(max_sd_ratio.to_bits(), 1.2_f64.to_bits());
        assert_eq!(tally.untouched_stages, 1);
        assert_eq!(tally.untouched_coordinates, 1);
    }
}
