//! Brute-force reference of the Level 1 ownership rule of de Matos, Philpott
//! and Finardi (2015), section 4.2, for tests to compare faster code against.
//!
//! A pool is a slice of [`RefCut`] in slot order. Its live set is every
//! [`RefRole::Generated`] and [`RefRole::ScoredPrefix`] cut plus every
//! [`RefRole::StatelessPrefix`] cut whose flag is active. A state's owner is
//! the lowest live slot whose value there equals the live maximum, compared
//! exactly, with no tolerance. A scored cut is selected when it owns a state,
//! and the active set after a selection run is the selected cuts plus the
//! active stateless prefix cuts.
//!
//! `states` is the list of stored states and no state belongs to a slot, so a
//! window is the same call with fewer states. The functions take plain data
//! and a plain loop instead of `CutPool` and `gemm_block`, so the reference
//! shares no storage and no arithmetic with the code compared against it.

use std::collections::BTreeSet;

/// A slot's role, fixed when the slot is written. `Vacant` is a slot never
/// written. A prefix cut is one the run started with, scored when its source
/// scored it. A scored role carries no activity flag: a scored cut's activity
/// is an output of the rule, never an input.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RefRole {
    Vacant,
    StatelessPrefix { active: bool },
    ScoredPrefix,
    Generated,
}

#[derive(Debug, Clone)]
pub(crate) struct RefCut {
    pub(crate) intercept: f64,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) role: RefRole,
}

/// The cut's value at `state`: the dot product accumulated from `0.0` left to
/// right, then the intercept added once.
///
/// # Panics
///
/// Panics when the coefficient count differs from the state's dimension.
pub(crate) fn value(cut: &RefCut, state: &[f64]) -> f64 {
    assert_eq!(
        cut.coefficients.len(),
        state.len(),
        "coefficient count differs from the state dimension"
    );
    let mut dot = 0.0;
    for (coefficient, x) in cut.coefficients.iter().zip(state) {
        dot += coefficient * x;
    }
    dot + cut.intercept
}

/// Each state's owner: the lowest live slot whose value there equals the live
/// maximum, or `None` when no slot is live.
pub(crate) fn owners(cuts: &[RefCut], states: &[Vec<f64>]) -> Vec<Option<usize>> {
    states
        .iter()
        .map(|state| {
            let mut incumbent: Option<(usize, f64)> = None;
            for (slot, cut) in cuts.iter().enumerate() {
                let live = matches!(
                    cut.role,
                    RefRole::Generated
                        | RefRole::ScoredPrefix
                        | RefRole::StatelessPrefix { active: true }
                );
                if !live {
                    continue;
                }
                let candidate = value(cut, state);
                if incumbent.is_none_or(|(_, best)| candidate > best) {
                    incumbent = Some((slot, candidate));
                }
            }
            incumbent.map(|(slot, _)| slot)
        })
        .collect()
}

/// The scored slots that own at least one state.
pub(crate) fn selected(cuts: &[RefCut], states: &[Vec<f64>]) -> BTreeSet<usize> {
    owners(cuts, states)
        .into_iter()
        .flatten()
        .filter(|&slot| matches!(cuts[slot].role, RefRole::Generated | RefRole::ScoredPrefix))
        .collect()
}

/// The active set after a selection run: the active stateless prefix slots
/// plus [`selected`].
pub(crate) fn active_after(cuts: &[RefCut], states: &[Vec<f64>]) -> BTreeSet<usize> {
    let mut active = selected(cuts, states);
    active.extend(
        cuts.iter()
            .enumerate()
            .filter(|(_, cut)| matches!(cut.role, RefRole::StatelessPrefix { active: true }))
            .map(|(slot, _)| slot),
    );
    active
}

pub(crate) mod tests {
    use std::collections::BTreeSet;

    use super::{RefCut, RefRole, active_after, owners, selected, value};

    pub(crate) type Fixture = (Vec<RefCut>, Vec<Vec<f64>>);

    fn line(intercept: f64, slope: f64, role: RefRole) -> RefCut {
        RefCut {
            intercept,
            coefficients: vec![slope],
            role,
        }
    }

    fn generated_lines(lines: &[(f64, f64)]) -> Vec<RefCut> {
        lines
            .iter()
            .map(|&(intercept, slope)| line(intercept, slope, RefRole::Generated))
            .collect()
    }

    fn points(xs: &[f64]) -> Vec<Vec<f64>> {
        xs.iter().map(|&x| vec![x]).collect()
    }

    fn figure_four_cuts() -> Vec<RefCut> {
        generated_lines(&[(10.0, -4.0), (6.5, -1.5), (5.0, -1.5), (4.0, -0.5)])
    }

    fn flat_prefix_then_flat_and_ramp(prefix_intercept: f64, prefix_role: RefRole) -> Fixture {
        (
            vec![
                line(prefix_intercept, 0.0, prefix_role),
                line(5.0, 0.0, RefRole::Generated),
                line(1.0, 1.0, RefRole::Generated),
            ],
            points(&[0.0, 7.0]),
        )
    }

    fn ties() -> Fixture {
        (
            generated_lines(&[(4.0, -1.0), (1.0, 2.0), (3.0, 0.0), (2.0, 0.0)]),
            points(&[1.0, 3.0]),
        )
    }

    fn figure_four() -> Fixture {
        (figure_four_cuts(), points(&[1.0, 3.0, 4.0]))
    }

    fn figure_four_with_a_new_state() -> Fixture {
        (figure_four_cuts(), points(&[1.0, 3.0, 4.0, 2.0]))
    }

    fn figure_four_windowed() -> Fixture {
        (figure_four_cuts(), points(&[3.0, 4.0, 2.0]))
    }

    fn active_stateless_prefix_tie() -> Fixture {
        flat_prefix_then_flat_and_ramp(5.0, RefRole::StatelessPrefix { active: true })
    }

    fn inactive_stateless_prefix() -> Fixture {
        flat_prefix_then_flat_and_ramp(5.0, RefRole::StatelessPrefix { active: false })
    }

    fn scored_prefix_below_the_envelope() -> Fixture {
        flat_prefix_then_flat_and_ramp(1.0, RefRole::ScoredPrefix)
    }

    fn scored_prefix_alone_at_the_top() -> Fixture {
        (
            vec![
                line(5.0, 0.0, RefRole::ScoredPrefix),
                line(1.0, 1.0, RefRole::Generated),
            ],
            points(&[0.0, 7.0]),
        )
    }

    fn scored_prefix_tie() -> Fixture {
        flat_prefix_then_flat_and_ramp(5.0, RefRole::ScoredPrefix)
    }

    fn vacant_slot() -> Fixture {
        (
            vec![
                line(-1.0, 0.0, RefRole::Generated),
                line(0.0, 0.0, RefRole::Vacant),
                line(-2.0, 0.0, RefRole::Generated),
            ],
            points(&[0.0]),
        )
    }

    fn two_dimensions() -> Fixture {
        let plane = |coefficients: [f64; 2]| RefCut {
            intercept: 0.0,
            coefficients: coefficients.to_vec(),
            role: RefRole::Generated,
        };
        (
            vec![plane([1.0, 0.0]), plane([0.0, 1.0]), plane([1.0, 1.0])],
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, -1.0]],
        )
    }

    pub(crate) fn fixtures() -> Vec<(&'static str, Fixture)> {
        vec![
            ("ties", ties()),
            ("figure_four", figure_four()),
            (
                "figure_four_with_a_new_state",
                figure_four_with_a_new_state(),
            ),
            ("figure_four_windowed", figure_four_windowed()),
            ("active_stateless_prefix_tie", active_stateless_prefix_tie()),
            ("inactive_stateless_prefix", inactive_stateless_prefix()),
            (
                "scored_prefix_below_the_envelope",
                scored_prefix_below_the_envelope(),
            ),
            (
                "scored_prefix_alone_at_the_top",
                scored_prefix_alone_at_the_top(),
            ),
            ("scored_prefix_tie", scored_prefix_tie()),
            ("vacant_slot", vacant_slot()),
            ("two_dimensions", two_dimensions()),
        ]
    }

    fn slots(expected: &[usize]) -> BTreeSet<usize> {
        expected.iter().copied().collect()
    }

    fn assert_rule(
        fixture: &Fixture,
        expected_owners: &[usize],
        expected_selected: &[usize],
        expected_active_after: &[usize],
    ) {
        let (cuts, states) = fixture;
        let expected_owners: Vec<Option<usize>> =
            expected_owners.iter().copied().map(Some).collect();
        assert_eq!(owners(cuts, states), expected_owners, "owners");
        assert_eq!(selected(cuts, states), slots(expected_selected), "selected");
        assert_eq!(
            active_after(cuts, states),
            slots(expected_active_after),
            "active after"
        );
    }

    fn envelope(cuts: &[RefCut], over: impl Iterator<Item = usize>, state: &[f64]) -> f64 {
        over.map(|slot| value(&cuts[slot], state))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn ties_go_to_the_lowest_slot() {
        assert_rule(&ties(), &[0, 1], &[0, 1], &[0, 1]);
    }

    #[test]
    fn paper_figure_four_selects_the_two_steep_cuts() {
        assert_rule(&figure_four(), &[0, 3, 3], &[0, 3], &[0, 3]);
    }

    #[test]
    fn a_new_state_reactivates_its_owner() {
        assert_rule(
            &figure_four_with_a_new_state(),
            &[0, 3, 3, 1],
            &[0, 1, 3],
            &[0, 1, 3],
        );
    }

    #[test]
    fn a_window_drops_a_state_and_keeps_every_cut_live() {
        assert_rule(&figure_four_windowed(), &[3, 3, 1], &[1, 3], &[1, 3]);
    }

    #[test]
    fn a_stateless_prefix_owner_blocks_a_tied_generated_cut() {
        assert_rule(&active_stateless_prefix_tie(), &[0, 2], &[2], &[0, 2]);
    }

    #[test]
    fn an_inactive_stateless_prefix_cut_is_inert() {
        assert_rule(&inactive_stateless_prefix(), &[1, 2], &[1, 2], &[1, 2]);
    }

    #[test]
    fn a_scored_prefix_cut_that_owns_nothing_is_deactivated() {
        assert_rule(
            &scored_prefix_below_the_envelope(),
            &[1, 2],
            &[1, 2],
            &[1, 2],
        );
    }

    #[test]
    fn a_scored_prefix_cut_that_owns_a_state_is_active() {
        assert_rule(&scored_prefix_alone_at_the_top(), &[0, 1], &[0, 1], &[0, 1]);
    }

    #[test]
    fn a_generated_cut_tied_with_a_lower_scored_prefix_owner_is_not_selected() {
        assert_rule(&scored_prefix_tie(), &[0, 2], &[0, 2], &[0, 2]);
    }

    #[test]
    fn a_vacant_slot_is_not_a_cut() {
        assert_rule(&vacant_slot(), &[0], &[0], &[0]);
    }

    #[test]
    fn ownership_reads_every_state_dimension() {
        assert_rule(&two_dimensions(), &[0, 1, 0], &[0, 1], &[0, 1]);
    }

    #[test]
    fn no_live_cut_leaves_a_state_unowned() {
        let cuts = vec![
            line(1.0, 0.0, RefRole::Vacant),
            line(2.0, 0.0, RefRole::StatelessPrefix { active: false }),
        ];
        let states = points(&[0.0]);
        assert_eq!(owners(&cuts, &states), vec![None]);
        assert!(selected(&cuts, &states).is_empty());
        assert!(active_after(&cuts, &states).is_empty());
    }

    #[test]
    fn paper_figure_four_loses_the_middle_cut_at_an_unvisited_state() {
        let (cuts, states) = figure_four();
        let unvisited = [2.0];
        assert_eq!(
            envelope(&cuts, selected(&cuts, &states).into_iter(), &unvisited),
            3.0
        );
        assert_eq!(envelope(&cuts, 0..cuts.len(), &unvisited), 3.5);
    }

    #[test]
    fn selected_set_reproduces_the_live_envelope_at_every_state() {
        for (_, (cuts, states)) in &fixtures() {
            let active = active_after(cuts, states);
            for state in states {
                let live = (0..cuts.len()).filter(|&slot| {
                    !matches!(
                        cuts[slot].role,
                        RefRole::Vacant | RefRole::StatelessPrefix { active: false }
                    )
                });
                assert_eq!(
                    envelope(cuts, active.iter().copied(), state).to_bits(),
                    envelope(cuts, live, state).to_bits()
                );
            }
        }
    }
}
