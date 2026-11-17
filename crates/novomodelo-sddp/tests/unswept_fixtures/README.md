# Unswept fixtures

Decks that exercise a combination no committed deck covers. They sit outside
the `committed_decks()` scan, which reads `examples/deterministic/`,
`crates/novomodelo-sddp/tests/fixtures/`, `examples/1dtoy` and `examples/4ree`,
so the sweeps over committed decks never load them. A test that needs one names
it explicitly.

## `dual_calendar_boundary/`

A travel-time arc (H0 -> H1), a lead-time anticipated thermal whose lead reaches
past the horizon, `post_study_stages.json`, and `config.policy.boundary`.

`water_calendar_differential.rs` builds its setups from this deck; the declared
boundary keeps the terminal bucket slots live. The boundary policy is not
committed, so never run the deck in place: `source_policy` does not exist in the
tree.

## `legacy_inputs/`

One overlay per committed deck whose post-study input was migrated to
`thermal_stages`/`water_bucket_stages`: the files the deck had before
migration, plus `absent.json`, the files it did not have.
`novomodelo_sddp::test_support::legacy_inputs::materialize` applies an overlay to a
deck copy. `scripts/migrate/test_in_repo_migration.py` migrates every overlay and
compares the result with the committed deck, and the legacy-vs-migrated tests
run a deck from its overlay.
