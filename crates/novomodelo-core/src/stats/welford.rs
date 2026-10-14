//! Online accumulator for running mean and variance using Welford's algorithm,
//! which avoids the catastrophic cancellation of the two-pass naive formula.

/// Running mean and sample standard deviation via Welford's single-pass online update.
#[derive(Debug)]
pub struct WelfordAccumulator {
    count: u64,
    mean: f64,
    /// Sum of squared deviations from the running mean.
    m2: f64,
}

impl WelfordAccumulator {
    /// Create a new accumulator with no observations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Incorporate a new observation into the running statistics.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        #[allow(clippy::cast_precision_loss)] // count stays in u32 range; f64 is exact there
        let count_f64 = self.count as f64;
        self.mean += delta / count_f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Running mean of all observed values, or `0.0` if no observations.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample standard deviation with Bessel's correction
    /// (`sqrt(m2 / (n - 1))`), or `0.0` if fewer than 2 observations.
    #[must_use]
    pub fn sample_std_dev(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = self.count as f64;
            (self.m2 / (count_f64 - 1.0)).sqrt()
        }
    }

    /// Half-width of the 95% confidence interval using the sample standard
    /// deviation (`1.96 * sample_std / sqrt(n)`).
    ///
    /// Returns `0.0` when fewer than 2 observations are available.
    #[must_use]
    pub fn sample_ci_95_half_width(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = self.count as f64;
            1.96 * self.sample_std_dev() / count_f64.sqrt()
        }
    }
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::WelfordAccumulator;

    #[test]
    fn welford_known_dataset_mean_and_sample_std_dev() {
        let values = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = WelfordAccumulator::new();
        for &v in &values {
            acc.update(v);
        }
        assert!(
            (acc.mean() - 5.0).abs() < 1e-10,
            "mean: expected 5.0, got {}",
            acc.mean()
        );
        assert!(
            (acc.sample_std_dev() - (32.0_f64 / 7.0).sqrt()).abs() < 1e-10,
            "sample_std_dev: expected 2.138089935299395, got {}",
            acc.sample_std_dev()
        );
    }

    #[test]
    fn welford_single_value_no_variance() {
        let mut acc = WelfordAccumulator::new();
        acc.update(42.0);
        assert!(
            (acc.mean() - 42.0).abs() < 1e-10,
            "mean: expected 42.0, got {}",
            acc.mean()
        );
        assert_eq!(
            acc.sample_std_dev(),
            0.0,
            "sample_std_dev must be 0.0 with one observation"
        );
        assert_eq!(
            acc.sample_ci_95_half_width(),
            0.0,
            "ci_95_half_width must be 0.0 with one observation"
        );
    }

    #[test]
    fn welford_zero_updates() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.mean(), 0.0, "mean must be 0.0 with no observations");
        assert_eq!(
            acc.sample_std_dev(),
            0.0,
            "sample_std_dev must be 0.0 with no observations"
        );
    }
}
