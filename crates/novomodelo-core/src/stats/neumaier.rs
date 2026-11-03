//! Neumaier (1974) compensated summation.

/// Running sum that keeps the rounding error of each addition in a separate compensation term.
///
/// The result depends on the order of [`add`](Self::add) calls, so callers fix that order to keep
/// sums reproducible.
///
/// # Examples
///
/// ```
/// use novomodelo_core::NeumaierAccumulator;
///
/// let mut acc = NeumaierAccumulator::new();
/// for term in [0.1; 10] {
///     acc.add(term);
/// }
/// assert_eq!(acc.value(), 1.0);
/// ```
#[derive(Debug)]
pub struct NeumaierAccumulator {
    sum: f64,
    compensation: f64,
}

impl NeumaierAccumulator {
    /// Create an accumulator whose value is `0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Add `term` to the running sum.
    pub fn add(&mut self, term: f64) {
        let t = self.sum + term;
        if self.sum.abs() >= term.abs() {
            self.compensation += (self.sum - t) + term;
        } else {
            self.compensation += (term - t) + self.sum;
        }
        self.sum = t;
    }

    /// Compensated total of the terms added so far.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

impl Default for NeumaierAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::NeumaierAccumulator;

    fn branch_form_sum(terms: &[f64]) -> f64 {
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;
        for &w in terms {
            let t = sum + w;
            if sum.abs() >= w.abs() {
                compensation += (sum - t) + w;
            } else {
                compensation += (w - t) + sum;
            }
            sum = t;
        }
        sum + compensation
    }

    fn first_index_exceeding(u: f64, terms: &[f64]) -> Option<usize> {
        let mut acc = NeumaierAccumulator::new();
        for (i, &term) in terms.iter().enumerate() {
            acc.add(term);
            if u < acc.value() {
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn add_recovers_terms_lost_to_cancellation() {
        let mut acc = NeumaierAccumulator::new();
        for term in [1.0, 1e100, 1.0, -1e100] {
            acc.add(term);
        }
        assert_eq!(acc.value(), 2.0);
    }

    #[test]
    fn weighted_terms_sum_to_their_compensated_total() {
        let (weight, cost) = (0.5_f64, 0.2_f64);
        let mut acc = NeumaierAccumulator::new();
        for _ in 0..10 {
            acc.add(weight * cost);
        }
        assert_eq!(acc.value(), 1.0);
    }

    #[test]
    fn value_after_each_add_drives_an_early_exit() {
        let terms = [0.1; 10];
        assert_eq!(
            first_index_exceeding(0.999_999_999_999_999_9, &terms),
            Some(9)
        );
    }

    #[test]
    fn add_matches_the_branch_form_bit_for_bit() {
        let terms = [1.0, 1e100, 1.0, -1e100, 0.1, -0.3, 1e-17, 3.5, -2.25, 1e16];
        let mut acc = NeumaierAccumulator::new();
        for (i, &term) in terms.iter().enumerate() {
            acc.add(term);
            assert_eq!(
                acc.value().to_bits(),
                branch_form_sum(&terms[..=i]).to_bits(),
                "prefix of length {}",
                i + 1
            );
        }
    }

    #[test]
    fn new_and_default_start_at_zero() {
        assert_eq!(NeumaierAccumulator::new().value(), 0.0);
        assert_eq!(NeumaierAccumulator::default().value(), 0.0);
    }
}
