# Gate fixtures

Decks that exercise a combination no committed deck covers. They sit outside
the `committed_decks()` scan, which reads only `examples/deterministic/` and
`crates/novomodelo-sddp/tests/fixtures/`, so the sweeps that build a `StudySetup`
without a source checkpoint never load them. A gate sweep adds a deck from this
directory explicitly.

## `dual_calendar_boundary/`

A travel-time arc (H0 -> H1), a lead-time anticipated thermal whose lead reaches
past the horizon, `post_study_stages.json`, and `config.policy.boundary`.

The boundary policy is not committed: a checkpoint is refused when its
`software_version` differs from the running binary. The deck nests a `source/`
deck instead, and a gate run:

1. runs `source/` and keeps its `policy/`;
2. copies that `policy/` to `<tmp copy of target>/source_policy`, the path
   `config.policy.boundary.path` names;
3. runs the target from the copy.

`prepare_dual_calendar_boundary_case` in `crates/novomodelo-cli/tests/common/mod.rs`
does steps 1 and 2.

Never run the target in place: `source_policy` does not exist in the tree.

The source's pool priced at the target's last stage `end_date` is the one the
target loads, so `source/` leads with a stage that ends on that date. Its
remaining stages equal the target's post-study stages, so both decks price the
same months on the same grid.

The inputs are frozen once a baseline is captured over this deck; editing one
invalidates that baseline's input hash.
