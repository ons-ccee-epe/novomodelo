# Wire Format Evolution

> **Status:** Living register — tracks append-only postcard discriminant
> assignments for tail-appended wire enum variants and the FlatBuffers field-id
> assignments of policy checkpoint manifest fields; re-derive against the live
> tree before acting.

Every postcard-encoded enum this register tracks — every enum named in the
Enum column of the table below — has no version byte: postcard serializes a
variant as its positional index, so inserting a variant mid-enum silently
shifts every later discriminant and reinterprets previously serialized values
as the wrong variant. Each such enum is therefore append-only — a new variant
is added at the tail only — and each tail assignment is guarded by a
first-byte discriminant pin plus a full encode → decode → equality round-trip.

A `ScalarExpr` serializes as the bare sequence of its `ScalarOp`s and
deserializes by replaying that sequence through its constructors, so a
malformed op list (empty, an operator with too few operands, or more than one
value left) is a decoding error, never a value.

## Tail assignments

| Variant                             | Enum                         | Postcard discriminant | Guarding test(s)                                                                                                 |
| ------------------------------------ | ----------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `HydroUsefulVolumeInitial`           | `VariableRef`                 | `0x18`                  | `test_variable_ref_postcard_discriminant_pin` (pin); `hydro_useful_volume_initial_postcard_roundtrip` (round-trip)  |
| `HydroUsefulVolumeFinal`             | `VariableRef`                 | `0x19`                  | `test_variable_ref_postcard_discriminant_pin` (pin); `hydro_useful_volume_final_postcard_roundtrip` (round-trip)    |
| `IntegratedEquivalentProductivity`   | `BroadcastComputedParameter`  | `0x07`                  | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_integrated_tags_round_trip` (round-trip) |
| `IntegratedAccumulatedProductivity`  | `BroadcastComputedParameter`  | `0x08`                  | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_integrated_tags_round_trip` (round-trip) |
| `MaxStoredEnergy`                    | `BroadcastComputedParameter`  | `0x09`                  | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_max_stored_energy_round_trip` (round-trip) |
| `IntegratedAccumulatedProductivityMwhPerHm3` | `BroadcastComputedParameter` | `0x0A` | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_mwh_tags_round_trip` (round-trip) |
| `MaxStoredEnergyMwh` | `BroadcastComputedParameter` | `0x0B` | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_mwh_tags_round_trip` (round-trip) |
| `IntegratedEquivalentProductivityMwhPerHm3` | `BroadcastComputedParameter` | `0x0C` | `broadcast_computed_parameter_postcard_discriminant_pin` (pin); `broadcast_computed_parameter_mwh_tags_round_trip` (round-trip) |
| `Literal` | `ScalarOp` | `0x00` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Param` | `ScalarOp` | `0x01` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Neg` | `ScalarOp` | `0x02` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Add` | `ScalarOp` | `0x03` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Sub` | `ScalarOp` | `0x04` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Mul` | `ScalarOp` | `0x05` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Div` | `ScalarOp` | `0x06` | `scalar_op_postcard_discriminant_pin` (pin); `scalar_expr_postcard_round_trip` (round-trip) |
| `Expression` | `BroadcastParameterKind` | `0x05` | `broadcast_parameter_kind_postcard_discriminant_pin` (pin); `broadcast_parameter_kind_expression_round_trip` (round-trip) |

## Why no reject-old-version test applies

A dual-owned wire format normally carries both a round-trip test and a
reject-old-version test (a stale version byte must fail decoding rather than
silently corrupt). Neither obligation is missing here — it does not apply, for
two reasons that hold for every enum this register tracks:

- **No version byte to reject on.** Postcard encodes an enum variant purely as
  its positional index; there is no separate version field in the payload for
  a decoder to compare against. A "feed a stale version byte, assert an error"
  test has nothing to assert against — the discriminant pin already covers the
  failure mode a version byte would guard (a shifted index reinterpreting a
  byte as the wrong variant).
- **Same-version-ranks-only, never persisted.** Every tracked enum travels
  only within one build, either inside the in-process model or between the
  MPI ranks of one solve broadcast from the same binary, and none is written
  to disk and read back by a possibly older or newer build, so there is no
  cross-version compatibility surface to test.

A future reader should not file a "missing reject-test" finding against any
tracked enum on this basis; the discriminant pin plus the full round-trip is
the complete discipline for an append-only, non-versioned, non-persisted wire
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
