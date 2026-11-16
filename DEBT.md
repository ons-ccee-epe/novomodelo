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

### DEBT-002
what:      novomodelo-sddp integration tests hold test-local generic_constraint_bounds.parquet writers beside the shared tests/common/parquet_fixtures.rs:write_generic_constraint_bounds — tests/deterministic.rs (the inline writers and the module-local write_generic_constraint_bounds), tests/cut_basis.rs:write_generic_constraint_bounds_parquet, tests/anticipated_scenarios.rs:write_constraint_bounds_parquet — and tests/template_integration/generic_constraints.rs restates the GenericBoundsRow tuple alias
why-now:   deliberate — the stage-volume work adds the shared writer for its own decks and leaves the pre-existing copies, to keep the change to the stated capability
quadrant:  deliberate x prudent
interest:  a change to the bounds parquet schema (a new column, a type change) must be repeated in every copy; a missed copy writes a file the loader refuses or reads differently, and each new test that copies a local writer adds one more
payoff:    replace each test-local writer with a call to common::parquet_fixtures::write_generic_constraint_bounds and delete the local copies, in a structure-only change
trigger:   the next change to the generic_constraint_bounds.parquet schema, or the next edit to one of the listed writers
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/tests/{deterministic.rs,cut_basis.rs,anticipated_scenarios.rs,template_integration/generic_constraints.rs}; the deterministic, cut_basis, anticipated_scenarios and template_integration test binaries must pass with --features test-support, slow-tests included
origin:    recorded on branch feat/generic-constraint-stage-volume when the shared bounds writer was planned; the local copies predate that work

### DEBT-003
what:      crates/novomodelo-sddp/src/simulation/extraction.rs:compute_cost_result builds SimulationCostResult with storage_violation_cost and filling_target_cost fixed at 0.0, and compute_hydro_violation_costs sums no storage-floor (filled_min_storage_floor_col) or filling-target (filling_target_col) slack, although both slacks are priced in the LP and reach total_cost
why-now:   deliberate — the stage-volume bounds work left it as found, to keep the change to its stated capability
quadrant:  deliberate x prudent
interest:  the category columns of simulation/costs do not show a priced storage-floor or filling-target violation: the price appears in total_cost and in no category column, so a reader of the breakdown sees 0.0 where a violation was paid for; it does not compound with new work, but every consumer of the columns inherits the gap
payoff:    compute both from their slack columns with the range_sum(range) * cost_scale_factor pattern behind an is_empty() guard (an empty range_sum is -0.0, not 0.0), decide whether they join hydro_violation_cost, and update the output dictionary in crates/novomodelo-io/src/output/dictionary.rs and the hydro_violation_cost_decomposition test in crates/novomodelo-sddp/src/simulation/extraction/tests.rs
trigger:   the next edit to compute_cost_result or compute_hydro_violation_costs, or the first report of a storage-floor or filling-target violation missing from the cost columns
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/src/simulation/extraction.rs, its test module crates/novomodelo-sddp/src/simulation/extraction/tests.rs and crates/novomodelo-io/src/output/dictionary.rs; the payoff changes the values of the simulation cost columns and needs its own CHANGELOG.md entry; the novomodelo-sddp lib tests, the deterministic and conformance test binaries, and the CLI and Python output tests must pass
origin:    recorded on branch feat/hydro-bounds-stage-volume when the outflow-volume cost columns were added to the same breakdown; the always-zero columns predate that work

### DEBT-004
what:      crates/novomodelo-io/src/penalties.rs:validate_hydro and crates/novomodelo-io/src/system/hydros.rs:convert_penalty_overrides leave the existing hydro penalty fields unvalidated (validate_hydro checks turbined_cost >= 0 and the required hydro fields > 0 but not the optional directional fields; convert_penalty_overrides copies every existing per-hydro override unchecked) while an explicitly declared outflow-volume price is validated at every level
why-now:   deliberate — the stage-volume bounds work validated only the prices it added and left the existing fields as found, to keep the change to its stated capability; tightening could refuse decks that load today
quadrant:  deliberate x prudent
interest:  a negative per-hydro penalty gives its slack a negative objective coefficient, which makes the LP unbounded, and a zero penalty turns the bound it prices off; the stage level in crates/novomodelo-io/src/constraints/penalty_overrides.rs already refuses a non-finite or non-positive present value, so the same value is accepted or refused depending on the level that carries it; an entity outflow_violation_*_cost of zero or below also reaches the outflow-volume slack through the derived price (crates/novomodelo-core/src/model/penalty.rs:resolve_hydro_penalties), so the native outflow-volume row can be priced at or below zero; it compounds with each penalty field added without a check
payoff:    validate every existing hydro penalty field finite and > 0 (>= 0 for turbined_cost) at the global-optional and entity levels, with refusal-parity rows in crates/novomodelo-cli/tests/refusal_parity.rs and crates/novomodelo-python/tests/test_refusal_parity.py
trigger:   the next field added to HydroPenalties, or the next edit to validate_hydro or convert_penalty_overrides
owner:     novomodelo-io maintainers
blast:     crates/novomodelo-io/src/penalties.rs, crates/novomodelo-io/src/system/hydros.rs and the refusal-parity tables of the CLI and Python suites; the payoff refuses decks that load today and needs its own CHANGELOG.md entry; the novomodelo-io lib tests, the CLI refusal_parity test binary and test_refusal_parity.py must pass
origin:    recorded on branch feat/hydro-bounds-stage-volume when the explicitly declared outflow-volume prices were validated at every level; the unvalidated existing fields predate that work

### DEBT-005
what:      crates/novomodelo-io/src/validation/semantic/stage_volume_bounds.rs:check_stage_volume_bounds refuses a min above a max across rows for the hm³ columns only (semantic.5a.59), so the m³/s override columns of constraints/hydro_bounds.parquet and constraints/pumping_bounds.parquet have no cross-row min-above-max refusal
why-now:   deliberate — the stage-volume bounds work left it as found, to keep the change to its stated capability
quadrant:  deliberate x prudent
interest:  an inverted override pair reaches the LP as a column whose lower bound exceeds its upper and fails at solve time instead of at load; the neighbouring rules do not cover it: semantic.5a.2 to semantic.5a.5 in crates/novomodelo-io/src/validation/semantic/hydro.rs check the entity base values in hydros.json, and semantic.5a.36 in crates/novomodelo-io/src/validation/semantic/block_bounds.rs refuses two rows setting the same column, not a min on one row above a max on another; it compounds with each min/max override column pair that no cross-row rule covers
payoff:    extend the semantic.5a.59 resolved cross-row check to the m³/s override columns under a new rule id, with refusal-parity rows in crates/novomodelo-cli/tests/refusal_parity.rs and crates/novomodelo-python/tests/test_refusal_parity.py
trigger:   the next edit to check_stage_volume_bounds, or the first report of an inverted m³/s override pair failing at solve time
owner:     novomodelo-io maintainers
blast:     crates/novomodelo-io/src/validation/semantic/stage_volume_bounds.rs, crates/novomodelo-io/src/validation/rules.rs and the refusal-parity tables of the CLI and Python suites; the novomodelo-io lib tests, the CLI refusal_parity test binary and test_refusal_parity.py must pass
origin:    recorded on branch feat/hydro-bounds-stage-volume when semantic.5a.59 was added for the hm³ columns; the missing check for the m³/s columns predates that work

### DEBT-006
what:      crates/novomodelo-io/src/constraints/bounds.rs:parse_hydro_bounds, parse_pumping_bounds and the other constraints/*.parquet parsers read their columns by name and ignore a column they do not know, where the JSON inputs refuse unknown keys through deny_unknown_fields
why-now:   deliberate — the stage-volume bounds work left it as found, to keep the change to its stated capability
quadrant:  deliberate x prudent
interest:  a binary that predates a column ignores it and solves without the declared bound, and a misspelled column name is dropped silently; it compounds with each new optional column
payoff:    refuse unknown column names in each constraints/*.parquet parser from one known-column list per file
trigger:   a deck run by a binary that predates one of its columns, or the next edit to parse_hydro_bounds or parse_pumping_bounds
owner:     novomodelo-io maintainers
blast:     crates/novomodelo-io/src/constraints/bounds.rs and the other parsers under crates/novomodelo-io/src/constraints; the novomodelo-io lib tests and the test binaries that load committed decks (deterministic, cli_validate) must pass
origin:    recorded on branch feat/hydro-bounds-stage-volume while the hm³ columns were added to the existing readers; the silent ignore of unknown columns predates that work

### DEBT-007
what:      crates/novomodelo-python/tests/test_cli_python_file_set_parity.py drives no non-zero outflow-volume slack or cost through both surfaces: it runs d28-decomp-weekly-monthly, d57-fpha-zero-turbine-capacity, d59-parameter-sum-useful-volume-floor and d60-stage-outflow-volume-cap, none with hm³ bounds, so the outflow-volume slack and cost columns are 0.0 on both, and the non-zero d62-stage-outflow-volume-bounds values are asserted only in crates/novomodelo-sddp/tests/deterministic.rs
why-now:   deliberate — the stage-volume bounds work left the parity test as found, to keep the change to its stated capability; both surfaces share crates/novomodelo-io/src/output/simulation_writer.rs, so parity holds by construction today
quadrant:  deliberate x prudent
interest:  a future divergence between the CLI and Python writers of these columns passes every committed test; it compounds with each output column added without a parity case
payoff:    add a d62-stage-outflow-volume-bounds case to test_cli_python_file_set_parity.py that asserts both surfaces write the same non-zero outflow_volume_slack_above_hm3 and outflow_volume_violation_above_cost
trigger:   the next change to the outflow-volume columns in crates/novomodelo-io/src/output/simulation_writer.rs, or the next edit to test_cli_python_file_set_parity.py
owner:     novomodelo-python maintainers
blast:     crates/novomodelo-python/tests/test_cli_python_file_set_parity.py only; the novomodelo-python test suite must pass with the examples/deterministic/d62-stage-outflow-volume-bounds deck
origin:    recorded on branch feat/hydro-bounds-stage-volume; the gap came with that work, which added the outflow-volume columns and the d62-stage-outflow-volume-bounds deck

### DEBT-008
what:      crates/novomodelo-sddp/tests/conformance.rs:constraint_extraction_regression_guard lists the hydro slack families that feed the hydro violation cost decomposition and asserts the list length, but omits outflow_volume_below_slack and outflow_volume_above_slack, which now also feed hydro_violation_cost
why-now:   inadvertent — found when the outflow-volume slack family was added
quadrant:  inadvertent x prudent
interest:  the guard's overlap check never sees the outflow-volume ranges, so a layout arithmetic bug that overlaps them with another family passes; the next family added can skip the guard the same way
payoff:    add both outflow-volume slack ranges to the guard with a fixture that declares one outflow-volume pair, and update its asserted count
trigger:   the next slack family added to hydro_violation_cost, or the next edit to constraint_extraction_regression_guard
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/tests/conformance.rs only; the novomodelo-sddp conformance test binary must pass
origin:    recorded on branch feat/hydro-bounds-stage-volume; the gap came with that work, which added the outflow-volume slack family

### DEBT-009
what:      crates/novomodelo-sddp/src/simulation/extraction.rs:HydroReverseLookup documents its slots as FPHA, evaporation and filling-slack lookups while the struct also holds the plant_floor and outflow_volume slots
why-now:   inadvertent — found when the outflow-volume slack family was added
quadrant:  inadvertent x prudent
interest:  a reader looking for a sparse family's slot can miss it; the enumeration goes stale again with the next sparse family
payoff:    rewrite the doc to state the rule (one slot per sparse per-(hydro, stage) family) instead of an enumeration, per .claude/rules/comments.md
trigger:   the next sparse per-(hydro, stage) family added to HydroReverseLookup, or the next edit to its doc comment
owner:     novomodelo-sddp maintainers
blast:     the HydroReverseLookup doc comment in crates/novomodelo-sddp/src/simulation/extraction.rs only; cargo doc and cargo clippy for novomodelo-sddp must stay clean
origin:    recorded on branch feat/hydro-bounds-stage-volume; the gap came with that work, which added the outflow_volume slot without extending the doc

## Paid
