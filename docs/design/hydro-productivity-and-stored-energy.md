# Hydro productivity and stored energy

> **Status:** Live spec — reflects shipped behavior. Verify the cited symbols against the tree before acting.

This is the contract-of-record for how novomodelo computes hydro productivity and
stored energy: which primitives exist, what inputs each one reads, and how a
security-curve constraint is authored against them. It consolidates facts that
are otherwise scattered across several crates' rustdoc; a caller integrating
against novomodelo (a bridge writing input decks, or a reader of simulation output)
should need only this page plus the symbols it points at.

## The 2x2 scope x evaluator model

Hydro productivity is computed along two independent axes. **Scope** is `own`
(the plant's own contribution) or `cascade` (the plant's own contribution plus
every downstream plant to the sea). **Evaluator** is `reference point` (head
evaluated at a single reference volume) or `useful-range mean` (head averaged
over a volume range). Each of the four cells is a `computed` scalar-parameter
tag (`ComputedParameter` in `novomodelo-core`'s parameter model), serialized in
`snake_case` and carrying only a `hydro_id`:

| | reference point | useful-range mean |
|---|---|---|
| **own** | `equivalent_productivity` | `integrated_equivalent_productivity` |
| **cascade** | `accumulated_productivity` | `integrated_accumulated_productivity` |

The two reference-point tags are unchanged: their JSON shape and their wire
payload carry no head field and no basis field, byte-identical to how they
existed before the mean evaluator was added. Scope is expressed entirely by
the tag name, not by a field on the tag. The cascade mean-evaluator cell also
feeds a fifth computed tag, `max_stored_energy`, described below.

Two simulation output columns mirror the mean-evaluator pair, tail-appended
after the existing reference-point pair, so the cascade recurrence is
checkable on disk for both evaluators:
`integrated_equivalent_productivity_mw_per_m3s` and
`integrated_accumulated_productivity_mw_per_m3s` (unit `MW/(m3/s)`, same unit
as the existing `equivalent_productivity_mw_per_m3s` /
`accumulated_productivity_mw_per_m3s` pair). Every output file carrying these
columns is written by both the CLI and the Python bindings.

## The quadrature invariant

The mean evaluator's own term, when it integrates rather than copies (see
"Geometry gates the mean evaluator" below), is `rho_esp * (mean_head - cf -
losses)`, where `cf`/`losses` are the same tailrace and hydraulic-loss terms
the reference-point evaluator already applies, and `mean_head` is
`ForebayTable::mean_height(v_lo, v_hi)` — the exact integral of the
piecewise-linear forebay-height curve over the plant's VHA breakpoints,
computed by composite trapezoid quadrature. `mean_height` clamps `v_lo` and
`v_hi` to the table's own breakpoint extent before integrating, so a VHA table
that does not span the plant's physical range integrates over the narrower
extent it does cover; authoring the table to span
`[Hydro.min_storage_hm3, Hydro.max_storage_hm3]` is the input author's
responsibility. This is exact for the
piecewise-linear table the composite trapezoid rule integrates, never an
approximation via Simpson's rule or any other quadrature scheme, and never a
read from fitted polynomial coefficients — the VHA table's own breakpoints are
the sole geometry source. A consumer that wants to reproduce this value
analytically (to compare against its own computation of the same integral)
should sample its own head curve at a resolution fine enough that a
piecewise-linear reconstruction matches the composite-trapezoid result to the
tolerance it needs; the two converge as sample count grows, since a
piecewise-linear curve is exactly what the trapezoid rule integrates.

## The physical/operative split

A hydro plant carries two distinct storage ranges. The **physical** range is
`Hydro.min_storage_hm3` / `Hydro.max_storage_hm3` — an entity-level field,
stage-invariant for the plant's lifetime. The **operative** range is
`HydroStageBounds.{min,max}_storage_hm3`, read per `(hydro, stage)` via
`ResolvedBounds::hydro_bounds` — the LP storage variable's actual bounds at
that stage, which may be tightened below the physical range by a flood-control
ceiling, a per-stage modifier, or any other operative constraint.

Every stored-energy consumer reads the **physical** range, never the
operative one: the two mean-evaluator grids' integration range
(`v_lo, v_hi` in the quadrature above), `max_stored_energy`'s `(V_hi - V_lo)`
factor, the `hydro_useful_volume_*` fold's `V_lo` term, and the
`stored_energy_*_mwh` output columns' `V_lo` term all read
`Hydro.{min,max}_storage_hm3`. `ResolvedBounds::hydro_bounds` is not called by
any of these four consumers. This is a modeled input distinction, not an
authoring obligation on any one input source: a plant's physical capacity and
its per-stage operative envelope are two different facts, so an active
flood-control ceiling never truncates a reported stored-energy value or the
security-curve integration range. A per-stage physical range (as opposed to
the stage-invariant entity field) is a reserved, not-built extension point,
for the case of a genuine mid-study physical capacity enlargement.

## The per-column override table

`system/hydro_energy_productivity.parquet` (parsed by
`crates/novomodelo-io/src/extensions/hydro_energy_productivity.rs`) carries three
optional per-`(hydro, stage)` override columns, each reaching a distinct,
non-overlapping set of consumers:

| Column | Symbol | Reaches |
|---|---|---|
| `equivalent_productivity_mw_per_m3s` (rho_eq) | direct productivity override | Both evaluators' own term, outright — see "geometry gates the mean evaluator" below — AND the `equivalent_productivity` output column. |
| `specific_productivity_mw_per_m3s_per_m` (rho_esp) | head-derived productivity coefficient | Both evaluators' head-derived productivity (the reference-point FPHA derivation and the mean own term above) AND the `specific_productivity` computed tag, through one shared resolver — `HydroEnergyProductivityOverride::resolve_specific_productivity` — so the tag and the builder can never disagree. |
| `reference_outflow_m3s` (Q_ref) | reference turbine flow override | The `reference_turbine` computed tag only. The head evaluation — both evaluators — always reads `Hydro.max_turbined_m3s`, never this column. |

An unset column resolves to the entity-level default: `rho_esp` falls back to
`Hydro.specific_productivity_mw_per_m3s_per_m`; `Q_ref` is not read from the
entity at all (`derive_conversion_for_hydro` hard-codes
`reference_outflow_m3s = hydro.max_turbined_m3s`). A study that leaves every
row of this parquet's `rho_esp` column NULL is byte-neutral: the resolved
`rho_esp` equals the entity value on every `(hydro, stage)`.

## `max_stored_energy`'s unit

`max_stored_energy(h)` is `integrated_accumulated_productivity(h, t) * (V_hi -
V_lo)` over the physical range. Its unit is the raw `productivity * volume`
product — not MWh. A security-curve constraint built from this coefficient
and the matching mean-evaluator tag (below) keeps both sides in this same raw
unit, so the per-hydro coefficient cancels out of the ratio the constraint
expresses; converting either side to MWh independently would reintroduce a
unit mismatch the raw form avoids.

## Stored-energy output units

`stored_energy_{initial,final}_mwh` is stage-length independent: it reads the
`integrated_accumulated_productivity` grid and the physical `V_lo`, converting
`hm3 * MW/(m3/s)` directly to MWh with a fixed hm3-to-seconds factor — no
stage duration enters the computation. `stored_energy_{initial,final}_mw` is
derived from it: `_mwh` divided by the stage's total block hours. The source
models this feature's inputs originate from (NEWAVE, DECOMP) report a
comparable quantity on a month-based unit; reconciling that unit against
novomodelo's stage-hours-based `_mw` column, when the two differ, is a consumer-side
concern — novomodelo reports the stage's own duration-normalized value and takes no
position on a month-based convention.

## The collapsed-range rule

When the physical range is collapsed (`V_lo == V_hi`), the mean evaluator's
own term equals the reference-point value at that same pinned volume — a
zero-width integration range reduces to a point evaluation. A plant whose VHA
table has only one row reaches the same outcome by a different route: the
table reports one constant height for every volume, so the mean over any
range equals the point value. This is how a source model's
run-of-river-versus-reservoir distinction falls out of the data: a plant
whose physical range is authored as collapsed behaves identically under
either evaluator, with no separate per-plant regulation-type concept and no
per-plant evaluator-selection field anywhere in novomodelo's input surface.

## Geometry gates the mean evaluator, not the generation model

The mean-evaluator own term is computed for any hydro that has VHA geometry
rows and a resolvable `rho_esp` at one or more stages — regardless of that
plant's generation model (FPHA, `constant_productivity`, or any other). The
resolution order per `(hydro, stage)` is: an explicit `rho_eq` override wins
outright and is copied bit-for-bit into both evaluators; otherwise, a
collapsed range or missing geometry/`rho_esp` copies the reference-point own
value bit-for-bit (the collapsed-range rule above); otherwise the quadrature
above computes `rho_esp * (mean_head - cf - losses)`. A VHA table that fails
to build is an error for every hydro it would otherwise cover — never a
silent fall-back to the reference-point value. The reference-point path
itself is unchanged: whether that value derives from FPHA geometry or from a
non-FPHA production model still depends solely on the plant's generation
model, as it always has.

## The `hydro_useful_volume_*` fold

`hydro_useful_volume_{initial,final}(id[, block])` are authorable generic
constraint variable references, parsed in
`crates/novomodelo-io/src/constraints/generic.rs` and resolving to
`VariableRef::HydroUsefulVolumeInitial` / `HydroUsefulVolumeFinal`. Each
resolves to the exact same LP column as the corresponding
`hydro_storage_{initial,final}` reference (coefficient multiplier `1.0`), and
receives the same referential and per-block validation the storage pair
already receives — it names no new LP variable. What differs is the bound: at
LP build time, `useful_volume_bound_shift` (in
`crates/novomodelo-sddp/src/lp/builder/layout.rs`) moves each term's dead volume
onto the bound: a term `coef * hydro_useful_volume_*(h)` stands for
`coef * (storage - V_lo)`, so `coef * V_lo` — with `V_lo` the entity physical
`Hydro.min_storage_hm3` — is ADDED to every resolved bound endpoint the
constraint carries, one such term per `hydro_useful_volume_*` reference. The
constraint reads as a useful-volume (above-dead-storage) quantity while the
underlying LP column still carries the absolute storage value; a negative
coefficient lowers the bound by the same rule.

The folded bound is publicly observable: `build_generic_constraint_echo_rows`
(`crates/novomodelo-sddp/src/generic_constraint_echo.rs`) reports the
post-fold value on `GenericConstraintEchoRow.bound_lower`, written to the same
`generic_constraint_echo` output by both the CLI and the Python bindings. This
echo output is the intended way to observe the folded bound — reading the
declared bound directly from the input constraint would show the pre-fold
value.

## The security-curve authoring recipe

A security curve is authored as a generic constraint with three ingredients:
a `computed` scalar parameter with `"tag": "integrated_accumulated_productivity"`
as the per-hydro coefficient, a `computed` scalar parameter with `"tag":
"max_stored_energy"` as the right-hand-side reference, and
`hydro_useful_volume_final(id)` as the left-hand-side variable — for example,
a constraint of the shape `coeff * hydro_useful_volume_final(id) >= pct *
max_energy`. Because `max_stored_energy` and
`integrated_accumulated_productivity` ride the same raw `productivity *
volume` unit (see above), and `hydro_useful_volume_final` reads as an
above-dead-storage quantity (see the fold above), this constraint expresses "at
least `pct` of the plant's useful volume, in energy terms" without any unit
conversion in the input.

The bridge's hand-folded equivalent — the form used before this feature
existed, still valid today — expresses the same constraint as a literal
coefficient on `hydro_storage_final(id)` with the plant's dead volume
pre-subtracted into the coefficient by hand. The two forms are equivalent by
construction; `crates/novomodelo-sddp/tests/deterministic.rs`'s
`security_curve_integrated_productivity_equivalence` module builds an
end-to-end deck exercising both forms side by side (non-uniform VHA geometry,
an entity `rho_esp` plus a stage-specific override, an operative ceiling below
the entity's physical maximum, and a non-FPHA `rho_eq` override) and owns the
resulting worked numbers. This page points at that module rather than
reproducing any of its figures; a reader who needs the actual resolved
values should read the module directly.

## The pairing contract and the two validation rules

Pairing `max_stored_energy(h)` on one side of a constraint with the
reference-point `accumulated_productivity(h)` tag for the same hydro on the
other is a mismatch: the two ride different evaluators and would not cancel
the way the authoring recipe above relies on. The matching (cancelling)
coefficient for `max_stored_energy` is `integrated_accumulated_productivity`,
never `accumulated_productivity`. This mismatch is what rule 51
(`crates/novomodelo-io/src/validation/semantic/constraints.rs`) detects, reporting
it as a non-blocking warning — it never rejects the constraint outright,
since a study author may have a reason to combine them that the checker
cannot see.

Separately, rule 52 (`crates/novomodelo-io/src/validation/semantic/stages.rs`)
requires every study stage to declare at least one block, and every declared
block's duration to be finite and strictly positive. This is the invariant
the `stored_energy_*_mw` division (`_mwh` divided by the stage's summed block
hours) relies on to never divide by zero or a non-finite value.

## The declared stored-energy value change

`stored_energy_{initial,final}_mwh` now reads the
`integrated_accumulated_productivity` grid and the physical `V_lo`, in place
of what it read before this feature. This is a declared, intentional value
change for any study whose mean and reference-point evaluators diverge (any
plant with a genuine, non-collapsed physical range and resolvable geometry) —
not a silent rebaseline. A study where every hydro's physical range is
collapsed, or where no hydro has resolvable VHA geometry, sees the two
evaluators coincide and the column is byte-neutral.

## Bridge-facing deviations

A bridge authoring a security curve against this contract writes a `computed`
scalar parameter with `"tag": "integrated_accumulated_productivity"` as
described above — never a `"basis"` field on `computed_spec`, and never a
`"head"` field on the existing `equivalent_productivity` /
`accumulated_productivity` tags (see the 2x2 table: scope and evaluator are
both expressed by tag name, not by a field). Two designs considered earlier in
this feature's development are retired and should not be re-attempted: moving
the flood-control ceiling and similar per-stage operative limits out of
`hydro_bounds` into a separate physical-range-purity channel (see "The
physical/operative split" above — the two ranges stay exactly where they
are); and a per-plant basis or participation field selecting which evaluator
a plant uses (the collapsed-range rule above already makes this unnecessary —
a plant with a collapsed range gets point-equal-to-mean behavior with no
field to set).

## Corner cases

A run-of-river plant whose physical range the input author pins to a single
volume (its declared minimum and maximum storage coincide) falls under the
collapsed-range rule: its mean evaluator equals its reference-point evaluator
exactly, with no separate code path for "this plant does not regulate."

A reservoir whose source model pins only its tailrace elevation (its forebay
still varies with volume, so its reference-point and mean productivities
genuinely differ) has no exact single-value representation in the override
table: an explicit `equivalent_productivity_mw_per_m3s` override wins outright
over both evaluators with the same value, which is exact for a collapsed range
and an approximation for such a reservoir. A per-stage forebay/tailrace
elevation override applied before the head derivation would make both
evaluators exact; no such input exists today.

A genuine mid-study enlargement of a plant's physical storage capacity — as
opposed to an operative, per-stage tightening — is a reserved, not-built
extension point (see "The physical/operative split" above); no input surface
exists for it today.

## See also

`docs/design/wire-format-evolution.md` owns the postcard discriminant ledger
for the tail-appended `VariableRef` and `BroadcastComputedParameter` variants
this feature adds. This page does not restate that ledger; consult it
directly for the wire-format assignments.
