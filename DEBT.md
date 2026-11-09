# DEBT register

Ids are zero-padded: `DEBT-001`, `DEBT-002`, and so on. Allocate the next id
by scanning this register for the highest id present and taking `max+1`. An id
is unique within this register and is never reused, because a `DEBT(` breadcrumb
in source cites it permanently.

An entry's state is the section it sits in. Paying an entry moves it verbatim
from `## Open` to `## Paid` with its `paid:` line appended. An entry is never
deleted and never renumbered.

## Entry Format

```
### DEBT-<nnn>
what:      one-line description of the shortcut/smell and where (file:symbol)
why-now:   the deliberate-prudent reason it was taken (or "inadvertent — found")
quadrant:  deliberate|inadvertent x prudent|reckless
interest:  what it slows / risks, and whether it compounds
payoff:    the concrete remediation
trigger:   the event that forces payoff
owner:     who closes this entry
blast:     files & consumers affected; tests that must pass
origin:    the plan and ticket that minted this entry
paid:      the ticket that paid it off (written only on retirement)
```

## Open

### DEBT-001
what:      crates/novomodelo-sddp/src/lp/builder/layout/tests.rs:useful_volume_fold_unresolvable_hydro_id_fires_debug_assert is a #[should_panic] test of a debug_assert! with no debug-only gate, so it fails under `cargo test --release`
why-now:   inadvertent — found while gating the sibling debug_assert! tests in lp/builder/entries.rs
quadrant:  inadvertent x prudent
interest:  a release-profile test run reports a false failure, which hides real release-only regressions; it compounds with every new ungated should_panic test of a debug_assert!
payoff:    gate the test on debug assertions (#[cfg(debug_assertions)], or #[cfg_attr(not(debug_assertions), ignore = "...")]), as the entries.rs tests are gated
trigger:   adding a release-profile test job to CI, or the next edit to lp/builder/layout/tests.rs
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/src/lp/builder/layout/tests.rs only; `cargo test -p novomodelo-sddp --lib` must pass in debug and in release
origin:    found on branch feat/generic-constraint-stage-volume while adding the stage-volume builder guard; not introduced by that work

## Paid
