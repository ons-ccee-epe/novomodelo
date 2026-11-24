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

## `unmigrated_inputs/`

Overlays that turn a migrated committed deck into its legacy twin:
`unmigrated_inputs/<generation>/<flat deck key>/` holds the files the deck had
before that generation's migration, plus `absent.json`, a JSON array of the paths
it did not have. The flat key is the deck's repository-relative path with each
`/` written `__`.

Each generation patches the deck as the next-newer generation left it; they are
listed newest first:

1. `initial_conditions/`: the files before `boundary_conditions.json` replaced
   `initial_conditions.json`, `stages.json` and `post_study_stages.json`.
2. `post_study_stages_key/`: the files before the post-study input moved from the
   `stages` key, or from no file, to `thermal_stages` and `water_bucket_stages`.

A deck has an overlay in a generation only when that migration changed it; a
missing generation directory means no overlays.

Every consumer goes through
`novomodelo_sddp::test_support::unmigrated_inputs::materialize_twin`. It applies
the newer generations that hold an overlay for the deck, newest first, then the
requested one, and fails unless the result holds `initial_conditions.json` and no
`boundary_conditions.json`. `deck_keys` lists the decks a generation covers.
`scripts/migrate/test_in_repo_migration.py` migrates every overlay and compares
the result with the committed deck. `tests/parity.rs` runs a deck from a
`post_study_stages_key/` overlay that only removes `post_study_stages.json`; an
overlay that writes a `stages` file is read by the migration test alone, because
the loader refuses that key.
