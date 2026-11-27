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
what:      crates/novomodelo-sddp/src/lp/builder/entries.rs:fill_generic_constraint_entries guards a stage-volume term (hydro_outflow_volume, hydro_diversion_volume, pumping_volume) on a per-block generic row with a debug_assert! only, so a System built through SystemBuilder in a release build gets per-block rows, each carrying the whole-stage volume, with no error
why-now:   deliberate — novomodelo-io refuses every input that would split a volume row (build_variable_ref, check_stage_volume_constraints), so no case loaded by the CLI or the Python bindings reaches the builder with one; the work kept the builder change to the stated capability
quadrant:  deliberate x prudent
interest:  a library caller that builds a System by hand and splits a volume row per block gets a model that counts the stage volume once per block, silently; it compounds only as SystemBuilder callers outside novomodelo-io appear
payoff:    refuse the composition in SystemBuilder::build in every build profile (or force a row with a volume term to stage level in the layout), with a test that pins the refusal
trigger:   a SystemBuilder caller outside novomodelo-io that builds generic constraints, or the next change to the stage-level collapse rule (constraint_collapses_to_stage_level)
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-core (SystemBuilder::build), crates/novomodelo-sddp/src/lp/builder/{entries.rs,layout.rs}; the novomodelo-io stage-volume validation tests and the template_integration generic-constraint tests must pass
origin:    recorded on branch feat/generic-constraint-stage-volume from the review of its stage-volume builder guard

## Paid

### DEBT-004
what:      crates/novomodelo-io/src/validation/semantic/electrical.rs:check_electrical_constructs — registered in the semantic driver but inert; the three checks are read only by tests
why-now:   the parsed electrical constraints are not carried by ParsedData yet, so no construct reaches the checks; they land first as pure functions over the syntax tree and a reference graph
quadrant:  deliberate x prudent
interest:  a case's electrical constructs are not validated at load until the entry point is wired; contained, does not compound
payoff:    carry the parsed electrical constraints on ParsedData, call the three checks per construct from check_electrical_constructs, and drop the module-level allow(dead_code)
trigger:   the electrical constraints loader is wired into the case load
owner:     novomodelo-io maintainers
blast:     validation/semantic/electrical.rs, validation/semantic/mod.rs, validation/schema.rs; `cargo test -p novomodelo-io validation::semantic::electrical`
origin:    native-electrical-constraints plan, epic-01 ticket-005
paid:      native-electrical-constraints plan, epic-03 ticket-012

### DEBT-005
what:      crates/novomodelo-io/src/validation/semantic/electrical.rs:call_fault — name and arguments of the generation, exchange, contract and load terms addressed by entity code, and of the reference to another restriction, are not checked
why-now:   the variable catalog resolves none of them yet, so checking them against it would reject every well-formed restriction; they pass through until their resolution lands
quadrant:  deliberate x prudent
interest:  a wrong argument list on these names surfaces at resolution instead of validation; contained, does not compound
payoff:    resolve these names in the catalog and drop the pass-through lists and the early return in call_fault
trigger:   code-to-id resolution of electrical constructs lands
owner:     novomodelo-io maintainers
blast:     validation/semantic/electrical.rs, expression/catalog.rs; `cargo test -p novomodelo-io validation::semantic::electrical`
origin:    native-electrical-constraints plan, epic-01 ticket-005
paid:      native-electrical-constraints plan, epic-03 ticket-009

### DEBT-006
what:      crates/novomodelo-io/src/expression/parser.rs:parse_condition — kept alive by allow(dead_code); read only by tests
why-now:   the activation-rule loader that parses condition strings is not wired yet, so the entry point lands first with its grammar tests
quadrant:  deliberate x prudent
interest:  an unused allow hides a genuinely dead entry point if the loader never lands; contained, does not compound
payoff:    call parse_condition from the activation-rule loader and drop the allow(dead_code)
trigger:   the activation-rule loader calls parse_condition
owner:     novomodelo-io maintainers
blast:     expression/parser.rs; `cargo test -p novomodelo-io expression`
origin:    native-electrical-constraints plan, epic-01 ticket-003
paid:      native-electrical-constraints plan, epic-03 ticket-012

### DEBT-007
what:      crates/novomodelo-sddp/src/lp/builder/generic_constraints.rs:resolve_variable_ref and policy/resolved_generic_coefficients.rs:resolve_constraint — HydroAvailability, hydro_generation by unit group, BusLoad and SystemLoad resolve to no column; a constraint using one is rejected at setup
why-now:   the variants exist so the DSL and the validators can carry electrical terms, while the builder has no column for them; rejecting beats the silent term loss a column-free arm would cause
quadrant:  deliberate x prudent
interest:  a generic constraint cannot use these four terms until the builder materializes them; contained, does not compound
payoff:    the builder materializes a column or coefficient for each term, resolve_variable_ref returns it, and the rejection in resolve_constraint (with its UnsupportedTerm error) is removed
trigger:   LP builder support for electrical terms lands
owner:     novomodelo-sddp maintainers
blast:     lp/builder/generic_constraints.rs, lp/builder/layout.rs, policy/resolved_generic_coefficients.rs, setup/mod.rs; `cargo test -p novomodelo-sddp --features test-support generic_constraint`
origin:    native-electrical-constraints plan, epic-01 ticket-001
paid:      native-electrical-constraints plan, epic-04 ticket-045 — the electrical rows materialize the four terms (tickets 013, 014); generic constraints reject them by contract

### DEBT-008
what:      crates/novomodelo-io/src/validation/referential.rs:validate_variable_ref_entity — the arm for HydroAvailability, hydro_generation by unit group, BusLoad and SystemLoad is a no-op; the entity codes are not checked for existence
why-now:   the entity arguments are entity codes until resolution translates them to ids, and the catalog resolves none of them yet, so checking them against the id sets would reject every well-formed term
quadrant:  deliberate x prudent
interest:  a term naming a non-existent plant, group or submarket surfaces at resolution instead of validation; contained, does not compound (see DEBT-005)
payoff:    validate the entity codes against the code-to-id map once the catalog resolves them, and replace the no-op arm with the existence checks
trigger:   code-to-id resolution of electrical constructs lands
owner:     novomodelo-io maintainers
blast:     validation/referential.rs, expression/catalog.rs; `cargo test -p novomodelo-io validation::referential`
origin:    native-electrical-constraints plan, epic-01 ticket-005
paid:      native-electrical-constraints plan, epic-03 ticket-009

### DEBT-009
what:      crates/novomodelo-io/src/expression/catalog.rs:check_call_shape — the reference to another restriction is not matched, so a well-formed reference is reported as an unknown variable name by the semantic checks, and `is_catalog_name` does not list it
why-now:   the electrical lowering resolves the reference itself and the semantic checks are inert until the parsed constraints are wired (DEBT-004), so the shape check could wait; a test in validation/semantic/electrical.rs still pins the reference as unknown to the catalog
quadrant:  deliberate x prudent
interest:  once the semantic checks are wired, every restriction that references another would be rejected at validation although it lowers; contained until then, does not compound
payoff:    match the reference in check_call_shape, add it to `is_catalog_name`, and replace the test that pins it as unknown with one that accepts a well-formed call and rejects a malformed one
trigger:   the electrical constraints loader is wired into the case load (DEBT-004)
owner:     novomodelo-io maintainers
blast:     expression/catalog.rs, validation/semantic/electrical.rs; `cargo test -p novomodelo-io validation::semantic::electrical`
origin:    native-electrical-constraints plan, epic-03 ticket-010
paid:      native-electrical-constraints plan, epic-03 ticket-012

### DEBT-013
what:      crates/novomodelo-io/src/constraints/electrical_resolve.rs — an inequality override whose stage_start is after its stage_end is accepted and never applies, while a formula override with the same fault is rejected as schema.3
why-now:   rejecting it changes which input files are accepted, which needs an operator decision on the file format
quadrant:  deliberate x prudent
interest:  a mistyped inequality override range is ignored silently; contained, does not compound
payoff:    report an inverted inequality override range as schema.3, like the formula override, with a test
trigger:   operator approves the format change
owner:     novomodelo-io maintainers
blast:     constraints/electrical_resolve.rs; `cargo test -p novomodelo-io constraints::electrical_resolve`
origin:    native-electrical-constraints plan, epic-04 ticket-046
paid:      native-electrical-constraints plan, epic-04 ticket-043

### DEBT-014
what:      crates/novomodelo-sddp/src/policy/resolved_electrical_rows.rs:RowResolver::bound — an electrical-row bound reading an if() selector is rejected at setup (UnevaluatedSelector) instead of being evaluated per solve
why-now:   the per-solve evaluation lands first for the scenario load values; a selector needs the data-only condition evaluator, which lands with the activation rules
quadrant:  deliberate x prudent
interest:  a case whose restriction bound reads an if() over scenario data fails the setup; contained, does not compound
payoff:    evaluate each selector per solve from its data-only condition, let the per-solve bound read its value, and remove UnevaluatedSelector
trigger:   the per-solve evaluation of activation rules and selectors lands (native-electrical-constraints plan, epic-05 ticket-017)
owner:     novomodelo-sddp maintainers
blast:     policy/resolved_electrical_rows.rs, policy/resolved_electrical_rows/per_solve.rs; `cargo test -p novomodelo-sddp --features test-support policy::resolved_electrical_rows`
origin:    native-electrical-constraints plan, epic-05 ticket-049
paid:      native-electrical-constraints plan, epic-05 ticket-017 — selectors are evaluated per solve

### DEBT-011
what:      crates/novomodelo-sddp/src/lp/builder/layout.rs:enumerate_generic_constraint_rows — every electrical-restriction row is hard (no slack column), whatever the violation treatment of its restriction
why-now:   the resolved electrical rows do not carry the violation treatments yet; the rows land first with their fixed structure, and a row given a SlackConfig gets its slack columns and their price from the existing generic-row allocation and fill with no other change
quadrant:  deliberate x prudent
interest:  a restriction a scenario cannot meet makes the stage LP infeasible instead of paying its violation cost; contained, does not compound
payoff:    carry each restriction's violation treatment on its resolved rows and set the row's slack from it: none for a hard treatment, the per-stage cost for a soft one, a soft row at 10x the deficit cost when the restriction has no treatment
trigger:   the violation treatment of the electrical restrictions lands (native-electrical-constraints plan, epic-06 tickets 020-022)
owner:     novomodelo-sddp maintainers
blast:     lp/builder/layout.rs, policy/resolved_electrical_rows.rs; `cargo test -p novomodelo-sddp --features test-support lp::builder::layout` and `cargo test -p novomodelo-sddp --features test-support --test parity --test template_permutation`
origin:    native-electrical-constraints plan, epic-04 ticket-015
paid:      native-electrical-constraints plan, epic-06 ticket-022 — each electrical row takes the slack of its violation treatment; no treatment takes 10x the largest deficit cost

### DEBT-015
what:      crates/novomodelo-sddp/src/simulation/extraction.rs:extract_generic_violations — the slack cost of the electrical rows is not reported in the simulation cost breakdown; `Σ(breakdown) == immediate_cost` does not hold when an electrical slack is used
why-now:   the electrical rows take their slack before the electrical violations have an output family of their own; reporting them as generic violations would mix the provenance of the two families
quadrant:  deliberate x prudent
interest:  a simulated stage that pays an electrical slack reports an immediate cost above the sum of its breakdown, and the slack value and cost are not reported; contained, does not compound
payoff:    extract the electrical violations into their own output family and add their cost to the breakdown, so `Σ(breakdown) == immediate_cost` holds again
trigger:   the extraction of the electrical violation values lands (native-electrical-constraints plan, epic-07 ticket-024)
owner:     novomodelo-sddp maintainers
blast:     simulation/extraction.rs, simulation/types.rs; `cargo test -p novomodelo-sddp --features test-support simulation::extraction`
origin:    native-electrical-constraints plan, epic-06 ticket-022
paid:      native-electrical-constraints plan, epic-07 ticket-024 — the slack of every electrical row is reported per row and summed into electrical_violation_cost

### DEBT-017
what:      crates/novomodelo-io/src/constraints/electrical_legacy.rs — the legacy input shape of `constraints/electrical_constraints.json` (formulas over entity codes, translated through the entity-code table), read by parse_legacy_electrical_constraints and resolved by resolve_legacy_electrical_constraints, kept beside the neutral shape
why-now:   parallel change: the external case converter still emits the legacy shape, and the legacy path is the differential oracle the neutral path is checked against, so both shapes coexist until the converter emits the neutral one
quadrant:  deliberate x prudent
interest:  two input shapes and two resolver fronts to keep in step, and every test written against the legacy shape is one more to migrate; compounds while such tests are added
payoff:    migrate the tests that still use the legacy shape, then delete the electrical_legacy module, the legacy resolver front and the dual read by file shape
trigger:   the external case converter emits the neutral shape and its contract test passes
owner:     novomodelo-io maintainers
blast:     every file `grep -rln "electrical_legacy\|legacy_electrical" crates` lists, in novomodelo-io and in the novomodelo-sddp tests that build resolved electrical constraints; `cargo test -p novomodelo-io` and `cargo test -p novomodelo-sddp --features test-support policy::resolved_electrical_rows`
origin:    electrical-constraints-neutral-input plan, epic-02 ticket-011
paid:      electrical-constraints-neutral-input plan, epic-04 ticket-034

### DEBT-018
what:      crates/novomodelo-sddp/src/policy/resolved_generic_coefficients.rs:resolve_constraint — the PerSolveRead branch refuses, per constraint, a bound or endpoint shift that reads a data atom or a selector of the scenario layer, and a layer entry with active_when, before any row is built
why-now:   the loader produces the scenario layer before the generic row owner consumes it; without the guard a constraint with active_when or an endpoint shift would be built as if it had neither
quadrant:  deliberate x prudent
interest:  a case whose generic constraints read data per scenario does not build; contained, does not compound
payoff:    the generic row owner evaluates those bounds, shifts and conditions per solve; the PerSolveRead branch and variant are deleted, and the tests that expect the refusal are rewritten to expect the rows
trigger:   the generic row owner evaluates bounds, endpoint shifts and active_when per solve
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/src/policy/resolved_generic_coefficients.rs (scenario_reads_are_a_coefficient_error_or_a_per_solve_refusal) and crates/novomodelo-sddp/src/setup/mod.rs (validate_generic_constraint_parameters_refuses_a_bound_read_per_solve); `cargo test -p novomodelo-sddp --features test-support --lib`
origin:    electrical-into-generic-constraints plan, epic-02 ticket-009
paid:      electrical-into-generic-constraints plan, epic-03 ticket-017

### DEBT-010
what:      crates/novomodelo-io/src/validation/semantic/electrical.rs:check_electrical_constructs — the restriction and expression formulas are checked here and again when the constraints are resolved at load (constraints/electrical_resolve.rs), so a faulty formula is reported by both layers; an activation condition that reads an LP variable or calls a data term with arguments it does not accept is reported by both too (its parse fault only by this layer)
why-now:   the semantic layer must cover every construct of the file, while the resolver must reject what it cannot lower; one layer owning the formula diagnostics needs the resolution to run behind the validation gate
quadrant:  deliberate x prudent
interest:  a case with a faulty formula lists the fault twice, once per layer and worded differently; contained, does not compound
payoff:    build the resolved model after the validation gate, or have the resolver hand its faults to the semantic layer, so one layer reports each formula fault, and drop the checks the other subsumes
trigger:   the resolved model is built after validation, in the study setup
owner:     novomodelo-io maintainers
blast:     validation/semantic/electrical.rs, constraints/electrical_resolve.rs, validation/schema.rs; `cargo test -p novomodelo-io validation::semantic::electrical` and `cargo test -p novomodelo-io constraints::electrical_resolve`
origin:    native-electrical-constraints plan, epic-03 ticket-012
paid:      electrical-into-generic-constraints plan, epic-05 ticket-037

### DEBT-012
what:      crates/novomodelo-io/src/validation/semantic/electrical.rs:formula_sources — the formula overrides of a restriction are not among the formulas the semantic layer checks; only the resolver (constraints/electrical_resolve.rs) reports their faults
why-now:   the resolver already rejects every fault in an override formula, and DEBT-010 is about to decide which single layer owns formula diagnostics; extending the semantic layer now would deepen the double reporting DEBT-010 removes
quadrant:  deliberate x prudent
interest:  an override formula fault is reported once while a base-formula fault is reported twice; contained, does not compound
payoff:    settle DEBT-010; if the semantic layer keeps formula checks, add the formula overrides to formula_sources with a test
trigger:   DEBT-010 is paid
owner:     novomodelo-io maintainers
blast:     validation/semantic/electrical.rs; `cargo test -p novomodelo-io validation::semantic::electrical`
origin:    native-electrical-constraints plan, epic-04 ticket-046
paid:      electrical-into-generic-constraints plan, epic-05 ticket-037

### DEBT-016
what:      crates/novomodelo-io/src/constraints/electrical_resolve.rs:pending_restrictions — two restrictions with the same id and the same name but different bodies are both kept, so the order of their later diagnostics (e.g. two different undeclared entity ids) follows declaration order
why-now:   the duplicate id is already reported as an error, so such a file never resolves; ordering the restrictions by body would need a canonical body key that nothing else uses
quadrant:  deliberate x prudent
interest:  only the order of error messages on an already-rejected file can differ between declaration orders; contained, does not compound
payoff:    order duplicate restrictions by a canonical body key (or keep only the first by that key) and extend faulty_order_reference with such a pair
trigger:   an input file or test reaches this case, or diagnostics become part of a golden comparison
owner:     novomodelo-io maintainers
blast:     constraints/electrical_resolve.rs; `cargo test -p novomodelo-io --lib constraints::electrical_resolve`
origin:    native-electrical-constraints plan, epic-10 ticket-035
paid:      electrical-into-generic-constraints plan, epic-05 ticket-037

### DEBT-019
what:      crates/novomodelo-sddp/tests/fixtures/constraint_row_golden/{template_facts.json,outputs.json} hold bytes no test reads — the `load_coupling` entry, which tests/constraint_row_oracle.rs:RETIRED_GOLDEN_CASES retires from the comparison, and the `coupling_buses` and `coupling_columns` keys of every stage, which the facts no longer carry
why-now:   deliberate — the golden was proven equal to the snapshot it replaced, and rewriting it would trade that proof for one recorded by the path under test; it is not rewritten without the owner's confirmation
quadrant:  deliberate x prudent
interest:  a reader of the golden sees facts the code no longer produces; contained by the test every_golden_entry_without_a_case_is_retired, which fails on any other entry without a case, so it does not compound
payoff:    rewrite the golden once without those bytes, proving the six remaining entries equal to the current files, then delete RETIRED_GOLDEN_CASES and make every_golden_entry_without_a_case_is_retired assert that no golden entry lacks a case
trigger:   the next approved rewrite of the golden
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/tests/fixtures/constraint_row_golden/*.json, crates/novomodelo-sddp/tests/constraint_row_oracle.rs; `cargo test -p novomodelo-sddp --features test-support --test constraint_row_oracle` must pass
origin:    electrical-into-generic-constraints plan, epic-07 ticket-048
paid:      operator-authorized rewrite of the golden, 2026-11-27, outside the plan's tickets — the bytes no test read were removed with the six remaining entries proven equal to the files they replace; RETIRED_GOLDEN_CASES is deleted and every_golden_entry_has_a_case asserts that no golden entry lacks a case

### DEBT-020
what:      crates/novomodelo-io/src/expression/lower.rs — classify_call types every name that is not a data term as an LP variable, so an unknown name in an `if()` condition or an `active_when` is refused with the LP-variable message; and a condition side that mixes an LP variable with `hydro_availability` is reported by its first term in `active_when` but by `hydro_availability` in an `if()` condition
why-now:   inadvertent — found in review; every such input is still refused, each message is true of the term it cites, and no test pins the text for an unknown name or a mixed side
quadrant:  inadvertent x prudent
interest:  a typo in a condition reads as a misuse of an LP variable, which slows the author's fix; does not compound
payoff:    check the name against the variable catalog before reporting an LP variable, and report the same term in `active_when` and in an `if()` condition, with tests that pin both messages
trigger:   the next change to the condition messages in expression/lower.rs, or the first user report of a misleading condition message
owner:     novomodelo-io maintainers
blast:     crates/novomodelo-io/src/expression/lower.rs, crates/novomodelo-io/src/expression/lower/tests.rs, crates/novomodelo-io/src/validation/semantic/constraints.rs; `cargo test -p novomodelo-io --features test-support` must pass
origin:    electrical-into-generic-constraints plan, epic-07 ticket-048
paid:      operator-authorized change, 2026-11-27, outside the plan's tickets — an unknown name in an `if()` condition or an `active_when` is refused as an unknown name, and a condition side that reads an LP variable and `hydro_availability` is refused at the first of the two in source order, in an `if()` condition and in an `active_when` alike

### DEBT-021
what:      three intra-doc links of novomodelo-sddp resolve only with the `test-support` feature — crates/novomodelo-sddp/src/lp/indexer/range_cursor.rs (module docs, `crate::lp::builder::StageLayout`), lp/indexer/state_space.rs (`Self::new`) and setup/mod.rs (`lp_build_inputs::build_stage_templates_resolving_layout`)
why-now:   inadvertent — found while documenting the crate; CI documents it with `--features test-support`, where the three links resolve
quadrant:  inadvertent x prudent
interest:  `RUSTDOCFLAGS="-Dwarnings" cargo doc -p novomodelo-sddp --no-deps --document-private-items` fails without the feature, which hides any new broken link from a local run; does not compound
payoff:    rewrite the three links to items visible without the feature
trigger:   CI documenting private items without `test-support`, or the next edit to one of the three files
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/src/{lp/indexer/range_cursor.rs,lp/indexer/state_space.rs,setup/mod.rs}; the cargo doc command above must exit 0 with and without `--features test-support`
origin:    electrical-into-generic-constraints plan, epic-07 ticket-048
paid:      operator-authorized change, 2026-11-27, outside the plan's tickets — the three links are plain code text now, since their targets are visible only with `test-support`; the cargo doc command exits 0 with and without the feature

### DEBT-022
what:      crates/novomodelo-sddp/src/simulation/extraction.rs:extract_buses — the branch that writes `spot_price` as 0.0 for a block of zero hours has no test
why-now:   inadvertent — found in review; the branch predates the work that found it
quadrant:  inadvertent x prudent
interest:  a change to the division or to the branch would go unnoticed for zero-hour blocks; does not compound
payoff:    a test with a nonzero balance dual and `block_hours = [0.0]` that requires `spot_price` to be 0.0
trigger:   the next edit to extract_buses
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/src/simulation/extraction.rs; `cargo test -p novomodelo-sddp --features test-support --lib simulation::extraction` must pass
origin:    electrical-into-generic-constraints plan, epic-07 ticket-048
paid:      operator-authorized change, 2026-11-27, outside the plan's tickets — a_block_with_no_hours_prices_its_buses_at_zero pins a zero spot_price for a block with no hours and a nonzero balance dual

### DEBT-023
what:      crates/novomodelo-sddp/tests/parity.rs, module scenario_constraint_invariance — assert_fixture_has_power checks only constraints 703 and 701, so the_constraints_file_is_invariant_to_declaration_order still passes if the `if()` of 704 or the condition of 707 becomes constant
why-now:   inadvertent — found in review; the gap predates the change under review and affects no production code
quadrant:  inadvertent x prudent
interest:  an order dependence confined to the selector of 704 or the condition of 707 would pass the test; does not compound
payoff:    extend the power check to require both branches of the `if()` of 704 and both states of 707, proven by mutating each to a constant
trigger:   the next edit to the scenario_constraint_invariance module
owner:     novomodelo-sddp maintainers
blast:     crates/novomodelo-sddp/tests/parity.rs; `cargo test -p novomodelo-sddp --features test-support --test parity scenario_constraint_invariance` must pass
origin:    electrical-into-generic-constraints plan, epic-07 ticket-048
paid:      operator-authorized change, 2026-11-27, outside the plan's tickets — assert_selector_fixture_has_power requires 704's if() to pick both branches and 707 to be both active and inactive across scenarios; constant mutations of either now fail
