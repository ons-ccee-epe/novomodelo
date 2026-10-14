# Wire Format Evolution

> **Status:** Living register — tracks append-only postcard discriminant
> assignments for tail-appended wire enum variants and the FlatBuffers field-id
> assignments of policy checkpoint manifest fields; re-derive against the live
> tree before acting.

`VariableRef` (`novomodelo-core`) and `BroadcastComputedParameter` (`novomodelo-io`) are
postcard-encoded enums with no version byte: postcard serializes a variant as
its positional index, so inserting a variant mid-enum silently shifts every
later discriminant and reinterprets previously serialized values as the wrong
variant. Both enums are therefore append-only — a new variant is added at the
tail only — and each tail assignment is guarded by a first-byte discriminant
pin plus a full encode → decode → equality round-trip.

## Tail assignments

| Variant                             | Enum                         | Postcard discriminant | Guarding test(s)                                                                                                 |
| ------------------------------------ | ----------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `HydroUsefulVolumeInitial`           | `VariableRef`                 | `0x18`                  | `test_variable_ref_postcard_discriminant_pin` (pin); `hydro_useful_volume_initial_postcard_roundtrip` (round-trip)  |
| `HydroUsefulVolumeFinal`             | `VariableRef`                 | `0x19`                  | `test_variable_ref_postcard_discriminant_pin` (pin); `hydro_useful_volume_final_postcard_roundtrip` (round-trip)    |
| `IntegratedEquivalentProductivity`   | `BroadcastComputedParameter`  | `0x07`                  | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_integrated_tags_round_trip` (round-trip) |
| `IntegratedAccumulatedProductivity`  | `BroadcastComputedParameter`  | `0x08`                  | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_integrated_tags_round_trip` (round-trip) |
| `MaxStoredEnergy`                    | `BroadcastComputedParameter`  | `0x09`                  | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_max_stored_energy_round_trip` (round-trip) |

## Why no reject-old-version test applies

A dual-owned wire format normally carries both a round-trip test and a
reject-old-version test (a stale version byte must fail decoding rather than
silently corrupt). Neither obligation is missing here — it does not apply, for
two reasons specific to these two enums:

- **No version byte to reject on.** Postcard encodes an enum variant purely as
  its positional index; there is no separate version field in the payload for
  a decoder to compare against. A "feed a stale version byte, assert an error"
  test has nothing to assert against — the discriminant pin already covers the
  failure mode a version byte would guard (a shifted index reinterpreting a
  byte as the wrong variant).
- **Same-version-ranks-only, never persisted.** `VariableRef` values travel
  inside the in-process constraint model built and consumed within one solver
  build. `BroadcastComputedParameter` values travel only between MPI ranks of
  a single running solve, broadcast from the same binary that constructed
  them. Neither payload is written to disk and read back by a possibly older
  or newer build, so there is no cross-version compatibility surface to test.

A future reader should not file a "missing reject-test" finding against either
enum on this basis; the discriminant pin plus the full round-trip is the
complete discipline for an append-only, non-versioned, non-persisted wire
enum.

## How to append a new variant

Add the variant at the enum's tail — never mid-enum. Add a first-byte
discriminant-pin assertion for the new variant (extending the enum's existing
pin test) and a full encode → decode → `assert_eq!` round-trip test covering
every field combination the variant's shape allows. Never insert a variant
before an existing one: doing so shifts every later discriminant and silently
reinterprets already-serialized values as the wrong variant.

## Policy checkpoint manifest fields (FlatBuffers)

The policy checkpoint's `manifest.bin` is a FlatBuffers `CheckpointManifest`
table. `crates/novomodelo-io/schemas/policy.fbs` is the canonical schema, and its
header owns the wire-level conventions, the burned slots and every field id
assigned before this register. The hand-rolled codec in
`crates/novomodelo-io/src/output/policy/codec.rs` addresses each field by a vtable
slot constant, `4 + 2 * id`. A new manifest field follows these rules:

- It takes the next explicit `id` at the table's end and never reuses a
  `deprecated` slot.
- Its codec slot constant changes together with `policy.fbs`.
- A buffer without the field decodes to the field's empty or `None` value,
  never to an error.
- The flatc conformance test
  (`crates/novomodelo-io/tests/flatbuffers_schema_conformance.rs`, feature
  `flatc-conformance`) guards both directions: the hand-rolled buffer read by
  `flatc`, and a `flatc`-built buffer read by the hand-rolled reader.
- An additive field needs no `FORMAT_VERSION` step inside a release. The policy
  load gate refuses any checkpoint whose `software`/`software_version` differ
  from the running build's, so every checkpoint a run loads was written by that
  same build.

### Field assignments

| Field                 | Table                | `id` | Codec slot | Guarding test(s)                                                                                                                                                                                                  |
| --------------------- | -------------------- | ---- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lower_bound_history` | `CheckpointManifest` | 21   | 46         | `checkpoint_manifest_round_trip` (flatc, both directions); `checkpoint_manifest_round_trips_the_lower_bound_history_bitwise` (round-trip); `checkpoint_manifest_without_the_lower_bound_history_reads_an_empty_series` (absent field) |

### Format version 4

Format version 4 adds fields to four tables and burns the fields they replace.
A checkpoint of another version is refused before any payload is parsed
(`deserialize_checkpoint_manifest`), so a burned field is never read; a
`flatc`-built buffer that still carries one decodes as if it did not.

Every piece's `intercept` and `coefficients` are stored in the writing study's
scaled units, and the pool's `StageCuts.cost_scale_factor` names that scale.
A load whose own factor is bit-equal to the pool's returns the stored bits
unchanged, so a resume at the same factor reproduces the written pool exactly.
A load at another factor converts every value as
`(value * cost_scale_factor) / loading_factor`, multiplying first. A pool whose
factor is absent, not finite or not positive is refused, naming the pool.

| Field                   | Table                | `id` | Codec slot | Change | Guarding test(s)                                                                                                                                       |
| ----------------------- | -------------------- | ---- | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `piece_id`              | `AffinePiece`        | 0    | 4          | burned | `stage_cuts_writer_matches_schema` (not written); `flatc_written_burned_piece_fields_are_ignored` (ignored on read)                                    |
| `scored`                | `AffinePiece`        | 8    | 20         | new    | `stage_cuts_writer_matches_schema`; `stage_cuts_reader_consumes_flatc_buffer`; `deserialize_stage_cuts_single_cut_all_fields`                          |
| `active_count`          | `AffinePiece`        | 9    | 22         | new    | `stage_cuts_writer_matches_schema`; `stage_cuts_reader_consumes_flatc_buffer`; `a_flatc_piece_without_activity_counters_reads_them_as_absent`           |
| `last_active_iteration` | `AffinePiece`        | 10   | 24         | new    | `stage_cuts_writer_matches_schema`; `stage_cuts_reader_consumes_flatc_buffer`; `a_flatc_piece_without_activity_counters_reads_them_as_absent`           |
| `active_cut_indices`    | `StageCuts`          | 5    | 14         | burned | `stage_cuts_writer_matches_schema` (not written); `flatc_written_burned_piece_fields_are_ignored` (ignored on read)                                    |
| `slots`                 | `StageStates`        | 6    | 16         | new    | `stage_states_writer_matches_schema`; `stage_states_reader_consumes_flatc_buffer`; `stage_states_entity_manifest_round_trip`                           |
| `num_columns`           | `StageBasis`         | 2    | 8          | burned | `stage_basis_writer_matches_schema` (not written); `flatc_written_burned_basis_fields_are_ignored` (ignored on read)                                   |
| `num_rows`              | `StageBasis`         | 3    | 10         | burned | `stage_basis_writer_matches_schema` (not written); `flatc_written_burned_basis_fields_are_ignored` (ignored on read)                                   |
| `column_status`         | `StageBasis`         | 4    | 12         | burned | `stage_basis_writer_matches_schema` (not written); `flatc_written_burned_basis_fields_are_ignored` (ignored on read)                                   |
| `row_status`            | `StageBasis`         | 5    | 14         | burned | `stage_basis_writer_matches_schema` (not written); `flatc_written_burned_basis_fields_are_ignored` (ignored on read)                                   |
| `num_cut_rows`          | `StageBasis`         | 6    | 16         | burned | `stage_basis_writer_matches_schema` (not written); `flatc_written_burned_basis_fields_are_ignored` (ignored on read)                                   |
| `entries`               | `StageBasis`         | 7    | 18         | new    | `stage_basis_writer_matches_schema`; `stage_basis_reader_consumes_flatc_buffer`                                                                         |
| `scenario`              | `BasisEntry`         | 0    | 4          | new    | `stage_basis_writer_matches_schema`; `stage_basis_reader_consumes_flatc_buffer`                                                                         |
| `column_status`         | `BasisEntry`         | 1    | 6          | new    | `stage_basis_writer_matches_schema`; `stage_basis_reader_consumes_flatc_buffer`                                                                         |
| `row_status`            | `BasisEntry`         | 2    | 8          | new    | `stage_basis_writer_matches_schema`; `stage_basis_reader_consumes_flatc_buffer`                                                                         |
| `piece_row_slots`       | `BasisEntry`         | 3    | 10         | new    | `stage_basis_writer_matches_schema`; `stage_basis_reader_consumes_flatc_buffer`                                                                         |
| `state_history_scope`   | `CheckpointManifest` | 22   | 48         | new    | `checkpoint_manifest_round_trip`; `checkpoint_manifest_state_history_scopes_read_from_flatc` (every scope, and the absent field)                       |
| `state_history_window`  | `CheckpointManifest` | 23   | 50         | new    | `checkpoint_manifest_round_trip`; `checkpoint_manifest_state_history_scopes_read_from_flatc`                                                            |
| `activity_counters`     | `CheckpointManifest` | 24   | 52         | new    | `checkpoint_manifest_round_trip`; `checkpoint_manifest_state_history_scopes_read_from_flatc`                                                            |
| `basis_store_complete`  | `CheckpointManifest` | 25   | 54         | new    | `checkpoint_manifest_round_trip`; `checkpoint_manifest_state_history_scopes_read_from_flatc`                                                            |

The version step itself is guarded by `a_hand_built_version_three_manifest_is_refused`
and `a_flatc_built_version_three_manifest_is_refused`, and the
stored-value rule by `export_then_load_same_factor_round_trips_bit_for_bit`,
`a_marked_checkpoint_converts_by_its_own_factor` and
`a_cross_factor_load_multiplies_by_the_source_before_dividing`
(`crates/novomodelo-sddp/src/policy/policy_load.rs`).
